/* period_height_check.cu
 *
 * 用 end_surface_period_scan 的命中 CSV，检查「周期循环点」附近是否真的存在高地表。
 *
 *   1) 读 CSV，按种子取最大 envelope；保留 envelope > --envelope-thr 的种子
 *      （默认 148；合法区间 137..152）。
 *   2) 对每个种子，以 peak 为基准、按**真实末地地表噪声周期**
 *      T = 245133.2823 格（= 5 × 49026.65646，即扫描程序 main 噪声周期的 5 倍，
 *      含 mod-5 相位类）枚举 2-D 周期循环点 k·T，|k·T| ≤ --range
 *      （默认 ±30000000，合法区间 122566..30000000）。
 *   3) 在每个循环点的 ±--window 格窗口内（默认 ±16，步长 --window-step 默认 1），
 *      判定**真实末地地表高度**是否 ≥ 73；出现 74/75 立刻打屏。
 *
 * 为什么只要查 celly 18（见 end_island_noise.cuh 的推导，cubiomes 注释同结论）：
 *   cy=18 的 upper_drop u = (78-18)/64 = 15/16 是「还能成实心」的最高 cell，
 *   所以 高度≥73 ⟺ cy=18 内存在实心块；高度 74 ⟺ y=3 不成、y=2 成。
 *
 * 高度口径 = cubiomes-end getEndSurfaceHeight（含内岛 + 25×25 外环外岛 +
 * 1.14+ overflow void），不是 Official JAR。
 *
 * host-only 构建（不需要 CUDA）：
 *   g++ -O3 -std=c++17 -o period_height_check period_height_check.cu -lm
 *
 *   period_height_check --hits end-hits-2B.csv
 *     [--envelope-thr 148] [--range 30000000] [--window 16] [--window-step 1]
 *     [--period 245133.2823] [--loop 2d|cross|1d] [--threads N]
 *     [--max-seeds N] [--max-loop-points N] [--out hits73.csv] [--quiet]
 *     [--selftest]
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "end_island_noise.cuh"
#include "end_surface_noise.cuh"

#define ES_TRUE_PERIOD 245133.2823 /* = 5 * ES_PERIOD */
#define ES_ENV_THR_LO 137.0
#define ES_ENV_THR_HI 152.0
#define ES_RANGE_LO 122566
#define ES_RANGE_HI 30000000

enum { LOOP_2D = 0, LOOP_CROSS = 1, LOOP_1D = 2 };

typedef struct Cfg {
    const char *hits_path;
    const char *out_path;
    double env_thr;
    int range;
    int window;
    int window_step;
    double period;
    int loop_mode;
    int threads;
    int max_seeds;
    long long max_loop_points; /* 0 = 不限 */
    int quiet;
    double depth_thr; /* D_eff <= 该值的列直接跳过（-1e30 = 关闭过滤） */
} Cfg;

typedef struct SeedRec {
    int64_t seed;
    double env;
    int px, pz;
    int is_max;
} SeedRec;

typedef struct Job {
    int seed_idx;
    int kx;
    int kz;
} Job;

/* 每个种子的统计（每个种子一把锁，避免全局串行） */
typedef struct SeedStat {
    std::mutex mtx;
    long long loops = 0;
    long long loops_ge73 = 0;
    long long cols_ge73 = 0;
    long long c73 = 0, c74 = 0, c75 = 0;
    long long cols_none = 0;  /* 通过 D_eff 过滤但 <73 */
    long long cols_filtered = 0; /* 被 D_eff 过滤掉的列 */
    long long cache_miss = 0; /* 理论不该出现（窗口缓存未覆盖） */
    int max_h = -1;
    int max_x = 0, max_z = 0;
    int best_kx = 0, best_kz = 0;
    int best_abs = 1 << 30; /* 同高度时取 |kx|+|kz| 最小（离峰值最近）的循环点 */
} SeedStat;

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --hits <scan.csv> [options]\n"
            "  --envelope-thr F   只查该阈值以上的种子 (%.0f..%.0f, 默认 148)\n"
            "  --range N          循环点搜索半径 ±N 格 (%d..%d, 默认 %d)\n"
            "  --window W         每个循环点检查 ±W 格 (默认 16)\n"
            "  --window-step S    窗口步长 (默认 1)\n"
            "  --period F         真实周期 (默认 %.4f = 5 x %.5f)\n"
            "  --loop MODE        2d | cross | 1d (默认 2d)\n"
            "  --threads N        线程数 (默认 硬件并发)\n"
            "  --max-seeds N      只取 envelope 最高的 N 个种子\n"
            "  --max-loop-points N  每种子最多枚举 N 个循环点 (调试用)\n"
            "  --depth-thr F      D_eff 过滤阈值 (默认 62; 见 README)\n"
            "  --no-depth-filter  关闭 D_eff 过滤 (慢, 作对照)\n"
            "  --out FILE         把所有 ≥73 的列写 CSV\n"
            "  --quiet            不打进度\n"
            "  --selftest         内部一致性自检 (缓存版 vs 直算版 vs 完整高度)\n"
            "\n"
            "高度口径: cubiomes-end getEndSurfaceHeight (内岛 + 25x25 外环外岛\n"
            "+ 1.14+ overflow void)。出现 74/75 立即打屏。\n",
            argv0, ES_ENV_THR_LO, ES_ENV_THR_HI, ES_RANGE_LO, ES_RANGE_HI,
            ES_RANGE_HI, ES_TRUE_PERIOD, (double)ES_PERIOD);
}

/* ---------------- CSV 读取 ---------------- */

static int parse_fmt_d(const char *s, double *out)
{
    char *end = NULL;
    double v;
    if (!s || !*s) return 0;
    v = strtod(s, &end);
    if (!end || (*end && *end != '\n' && *end != '\r')) return 0;
    *out = v;
    return 1;
}

static int parse_fmt_i64(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v;
    if (!s || !*s) return 0;
    v = strtoll(s, &end, 10);
    if (!end || (*end && *end != '\n' && *end != '\r')) return 0;
    *out = (int64_t)v;
    return 1;
}

/* 取第 idx 个逗号分隔字段（就地不修改，返回起始指针与长度） */
static int field_at(const char *line, int idx, const char **beg, size_t *len)
{
    const char *p = line;
    int i = 0;
    while (i < idx) {
        const char *c = strchr(p, ',');
        if (!c) return 0;
        p = c + 1;
        i++;
    }
    {
        const char *c = strchr(p, ',');
        *beg = p;
        *len = c ? (size_t)(c - p) : strlen(p);
        while (*len && ((*beg)[*len - 1] == '\n' || (*beg)[*len - 1] == '\r'))
            (*len)--;
    }
    return 1;
}

static int field_i64(const char *line, int idx, int64_t *out)
{
    const char *beg;
    size_t len;
    char buf[64];
    if (!field_at(line, idx, &beg, &len) || len == 0 || len >= sizeof(buf)) return 0;
    memcpy(buf, beg, len);
    buf[len] = 0;
    return parse_fmt_i64(buf, out);
}

static int field_f64(const char *line, int idx, double *out)
{
    const char *beg;
    size_t len;
    char buf[64];
    if (!field_at(line, idx, &beg, &len) || len == 0 || len >= sizeof(buf)) return 0;
    memcpy(buf, beg, len);
    buf[len] = 0;
    return parse_fmt_d(buf, out);
}

/* 按种子归并：取 envelope 最大的那一行（同一 seed 的 peak/branch 应当一致） */
static int load_seeds(const char *path, std::vector<SeedRec> *out)
{
    FILE *fp = fopen(path, "r");
    char line[512];
    std::unordered_map<int64_t, size_t> index;
    long long rows = 0, bad = 0, inconsistent = 0;
    int header_seen = 0;

    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    while (fgets(line, sizeof(line), fp)) {
        int64_t seed;
        double env;
        int64_t px, pz;
        char brbuf[16];
        const char *bbeg;
        size_t blen;
        int is_max;

        if (!field_i64(line, 0, &seed)) {
            if (!header_seen) {
                header_seen = 1;
                continue; /* 表头 */
            }
            bad++;
            continue;
        }
        header_seen = 1;
        if (!field_f64(line, 4, &env) || !field_i64(line, 5, &px)
            || !field_i64(line, 6, &pz)) {
            bad++;
            continue;
        }
        is_max = 0;
        if (field_at(line, 1, &bbeg, &blen) && blen < sizeof(brbuf)) {
            memcpy(brbuf, bbeg, blen);
            brbuf[blen] = 0;
            is_max = (strcmp(brbuf, "max") == 0);
        }
        rows++;
        {
            std::unordered_map<int64_t, size_t>::iterator it = index.find(seed);
            if (it == index.end()) {
                SeedRec r;
                r.seed = seed;
                r.env = env;
                r.px = (int)px;
                r.pz = (int)pz;
                r.is_max = is_max;
                index[seed] = out->size();
                out->push_back(r);
            } else {
                SeedRec *r = &(*out)[it->second];
                if (r->px != (int)px || r->pz != (int)pz) inconsistent++;
                if (env > r->env) {
                    r->env = env;
                    r->px = (int)px;
                    r->pz = (int)pz;
                    r->is_max = is_max;
                }
            }
        }
    }
    fclose(fp);
    fprintf(stderr,
            "csv: %lld rows, %zu distinct seeds (%lld bad lines, %lld inconsistent "
            "peak)\n",
            rows, out->size(), bad, inconsistent);
    return 1;
}

/* ---------------- 单点检查 ---------------- */

typedef struct HitSink {
    std::mutex *io_mtx;
    FILE *out;
    long long printed74;
} HitSink;

static void report_hit(HitSink *sink, const SeedRec *rec, int kx, int kz, int base_x,
                       int base_z, int x, int z, int h)
{
    std::lock_guard<std::mutex> lk(*sink->io_mtx);
    if (h >= 74) {
        printf("[%d] seed=%lld x=%d z=%d  (peak=%d,%d  k=(%d,%d) base=(%d,%d))\n", h,
               (long long)rec->seed, x, z, rec->px, rec->pz, kx, kz, base_x, base_z);
        fflush(stdout);
        sink->printed74++;
    }
    if (sink->out) {
        fprintf(sink->out, "%lld,%s,%.6f,%d,%d,%d,%d,%d,%d,%d\n", (long long)rec->seed,
                rec->is_max ? "max" : "min", rec->env, kx, kz, base_x, base_z, x, z, h);
    }
}

/* ---------------- 主流程 ---------------- */

static int run_scan(const Cfg *cfg, std::vector<SeedRec> *seeds, std::vector<Job> *jobs)
{
    const int nseeds = (int)seeds->size();
    const int K = (int)floor(cfg->period > 0.0 ? (double)cfg->range / cfg->period : 0.0);
    std::vector<SeedStat> stats((size_t)nseeds);
    std::mutex io_mtx;
    HitSink sink;
    std::atomic<size_t> next(0);
    std::vector<std::thread> pool;
    int t;

    sink.io_mtx = &io_mtx;
    sink.out = NULL;
    sink.printed74 = 0;

    if (cfg->out_path) {
        sink.out = fopen(cfg->out_path, "w");
        if (!sink.out) {
            fprintf(stderr, "cannot open %s\n", cfg->out_path);
            return 0;
        }
        fprintf(sink.out,
                "seed,branch,envelope,loop_kx,loop_kz,base_x,base_z,x,z,height\n");
    }

    fprintf(stderr,
            "scan: seeds=%d  loop=%s  K=%d (per axis %d points)  jobs=%zu  "
            "window=+-%d step=%d  threads=%d\n",
            nseeds,
            cfg->loop_mode == LOOP_2D ? "2d"
                                      : (cfg->loop_mode == LOOP_CROSS ? "cross" : "1d"),
            K, 2 * K + 1, jobs->size(), cfg->window, cfg->window_step, cfg->threads);
    if (cfg->depth_thr > -1e29)
        fprintf(stderr,
                "depth filter: 跳过 D_eff < %.1f  (该点高度73 需加权噪声 > %.2f,"
                " 74 需 > %.2f)\n",
                cfg->depth_thr, 213.3891 - cfg->depth_thr,
                226.8908 - cfg->depth_thr);
    else
        fprintf(stderr, "depth filter: off\n");

    for (t = 0; t < cfg->threads; t++) {
        pool.push_back(std::thread([&]() {
            std::vector<EsSurfaceNoise> sn_cache((size_t)nseeds);
            std::vector<EsPerlin> is_cache((size_t)nseeds);
            std::vector<char> ready((size_t)nseeds, 0);
            const size_t side =
                (size_t)(2 * cfg->window / cfg->window_step + 1);
            std::vector<int> sel_x(side * side), sel_z(side * side);
            EsEndWin win;
            long long done = 0;

            for (;;) {
                const size_t j = next.fetch_add(1);
                Job job;
                SeedRec rec;
                SeedStat *st;
                int x0, z0, bx, bz;

                if (j >= jobs->size()) break;
                job = (*jobs)[j];
                rec = (*seeds)[(size_t)job.seed_idx];
                st = &stats[(size_t)job.seed_idx];

                if (!ready[(size_t)job.seed_idx]) {
                    es_init_surface_noise_end(&sn_cache[(size_t)job.seed_idx], rec.seed);
                    es_end_island_init(&is_cache[(size_t)job.seed_idx], rec.seed);
                    ready[(size_t)job.seed_idx] = 1;
                }

                x0 = rec.px + (int)llround((double)job.kx * cfg->period);
                z0 = rec.pz + (int)llround((double)job.kz * cfg->period);

                if (!es_end_win_prepare_depth(&is_cache[(size_t)job.seed_idx], x0, z0,
                                              cfg->window, &win)) {
                    std::lock_guard<std::mutex> lk(st->mtx);
                    st->cache_miss++;
                    continue;
                }

                {
                    long long n73 = 0, none = 0, miss = 0, filt = 0;
                    int lo_max = -1, lo_x = 0, lo_z = 0;
                    int nsel = 0, k;

                    /* pass 1: D_eff 过滤 + 标记需要的 cell（此时还没算任何
                     * SurfaceNoise —— 这是 depth 过滤唯一能真正省下开销的位置） */
                    for (bz = z0 - cfg->window; bz <= z0 + cfg->window;
                         bz += cfg->window_step) {
                        for (bx = x0 - cfg->window; bx <= x0 + cfg->window;
                             bx += cfg->window_step) {
                            if (es_end_win_deff(&win, bx, bz) < cfg->depth_thr) {
                                filt++;
                                continue;
                            }
                            sel_x[(size_t)nsel] = bx;
                            sel_z[(size_t)nsel] = bz;
                            nsel++;
                            es_end_win_mark(&win, bx, bz, &win);
                        }
                    }

                    /* pass 2: 只为被选中的 cell 算 v18/v19 */
                    es_end_win_fill_noise(&sn_cache[(size_t)job.seed_idx], &win);

                    /* pass 3: 求值 */
                    for (k = 0; k < nsel; k++) {
                        const int h = es_end_height73_cached(&win, sel_x[(size_t)k],
                                                             sel_z[(size_t)k]);
                        if (h == -2) {
                            miss++;
                            continue;
                        }
                        if (h < 0) {
                            none++;
                            continue;
                        }
                        n73++;
                        if (h > lo_max) {
                            lo_max = h;
                            lo_x = sel_x[(size_t)k];
                            lo_z = sel_z[(size_t)k];
                        }
                        report_hit(&sink, &rec, job.kx, job.kz, x0, z0,
                                   sel_x[(size_t)k], sel_z[(size_t)k], h);
                    }
                    {
                        std::lock_guard<std::mutex> lk(st->mtx);
                        st->loops++;
                        st->cols_none += none;
                        st->cols_filtered += filt;
                        st->cache_miss += miss;
                        if (n73) {
                            st->loops_ge73++;
                            st->cols_ge73 += n73;
                        }
                        {
                            const int absk = (job.kx < 0 ? -job.kx : job.kx)
                                           + (job.kz < 0 ? -job.kz : job.kz);
                            if (lo_max > st->max_h
                                || (lo_max >= 73 && lo_max == st->max_h
                                    && absk < st->best_abs)) {
                                st->max_h = lo_max;
                                st->max_x = lo_x;
                                st->max_z = lo_z;
                                st->best_kx = job.kx;
                                st->best_kz = job.kz;
                                st->best_abs = absk;
                            }
                        }
                        if (lo_max == 73) st->c73++;
                        else if (lo_max == 74) st->c74++;
                        else if (lo_max >= 75) st->c75++;
                    }
                }

                done++;
                if (!cfg->quiet && (done % 2000) == 0) {
                    std::lock_guard<std::mutex> lk(io_mtx);
                    fprintf(stderr, "  ... %lld jobs done\n", done);
                }
            }
        }));
    }
    for (t = 0; t < (int)pool.size(); t++) pool[t].join();

    printf("\n=== 结果 (envelope > %.6f, range +-%d, %s, window +-%d/%d) ===\n",
           cfg->env_thr, cfg->range,
           cfg->loop_mode == LOOP_2D ? "2d"
                                     : (cfg->loop_mode == LOOP_CROSS ? "cross" : "1d"),
           cfg->window, cfg->window_step);
    printf("%-12s %11s %8s %10s %9s %5s\n", "seed", "envelope", "loops", "loops>=73",
           "cols>=73", "maxH");
    {
        long long tot_loops = 0, tot_ge = 0, tot_cols = 0, tot74 = 0, tot75 = 0, tot73 = 0;
        long long tot_miss = 0, tot_filt = 0, tot_none = 0;
        int any73 = 0;
        int i;
        for (i = 0; i < nseeds; i++) {
            SeedStat *st = &stats[(size_t)i];
            printf("%-12lld %11.6f %8lld %10lld %9lld %5d %s\n",
                   (long long)(*seeds)[(size_t)i].seed, (*seeds)[(size_t)i].env,
                   st->loops, st->loops_ge73, st->cols_ge73, st->max_h,
                   st->max_h >= 73 ? "  <= 有 >=73" : "");
            if (st->max_h >= 73) {
                any73 = 1;
                printf("%-12s %11s %8s %10s %9s      -> 最近 >=73 的点 (%d,%d) "
                       "k=(%d,%d)  [loop pts: 73=%lld 74=%lld 75=%lld]\n",
                       "", "", "", "", "", st->max_x, st->max_z, st->best_kx,
                       st->best_kz, st->c73, st->c74, st->c75);
            }
            tot_loops += st->loops;
            tot_ge += st->loops_ge73;
            tot_cols += st->cols_ge73;
            tot73 += st->c73;
            tot74 += st->c74;
            tot75 += st->c75;
            tot_miss += st->cache_miss;
            tot_filt += st->cols_filtered;
            tot_none += st->cols_none;
        }
        printf("\n合计: 循环点 %lld, 其中含 >=73 的循环点 %lld, >=73 的列 %lld\n",
               tot_loops, tot_ge, tot_cols);
        printf("      含 73 的循环点 %lld, 含 74 的 %lld, 含 75 的 %lld"
               "  (74/75 已即时打屏)\n",
               tot73, tot74, tot75);
        if (cfg->depth_thr > -1e29) {
            printf("      D_eff 过滤: 跳过 %lld 列, 求值 %lld 列 (保留 %.3f%%)\n",
                   tot_filt, tot_none + tot_cols,
                   100.0 * (double)(tot_none + tot_cols)
                       / (double)(tot_filt + tot_none + tot_cols));
            printf("        阈值: 跳过 D_eff < %.1f ⟹ 该点高度73 需加权噪声 > %.2f"
                   " (74 需 > %.2f)\n",
                   cfg->depth_thr, 213.3891 - cfg->depth_thr,
                   226.8908 - cfg->depth_thr);
        }
        if (tot_miss) printf("      !! 窗口缓存未覆盖的循环点 %lld\n", tot_miss);
        printf("结论: envelope > %.6f 的 %d 个种子里,%s出现 >=73 的真实地表高度\n",
               cfg->env_thr, nseeds, any73 ? "" : "没有");
    }
    if (sink.out) {
        fclose(sink.out);
        printf("所有 >=73 的列已写入 %s\n", cfg->out_path);
    }
    return 1;
}

/* ---------------- 自检 ---------------- */

static int selftest(void)
{
    static const uint64_t seeds[] = {0ULL, 305ULL, 81604415554ULL, 1400753836ULL,
                                     694195937ULL};
    static const int offs[] = {0, 1, 7, 8, -1, 16, -16, 1000, -122566, 122566};
    int fail = 0, n = 0, s, o;

    for (s = 0; s < (int)(sizeof(seeds) / sizeof(seeds[0])); s++) {
        EsSurfaceNoise sn;
        EsPerlin island;
        es_init_surface_noise_end(&sn, seeds[s]);
        es_end_island_init(&island, seeds[s]);
        for (o = 0; o < (int)(sizeof(offs) / sizeof(offs[0])); o++) {
            const int bx = offs[o], bz = -offs[o] + 3;
            const int h_ref = es_end_height73(&sn, &island, bx, bz);
            const int h_ex = es_end_height_exact(&sn, &island, bx, bz);
            EsEndWin win;
            int h_win;
            n++;
            if (!es_end_win_prepare(&sn, &island, bx, bz, 16, &win)) {
                printf("FAIL: win_prepare seed=%llu (%d,%d)\n",
                       (unsigned long long)seeds[s], bx, bz);
                fail++;
                continue;
            }
            h_win = es_end_height73_cached(&win, bx, bz);
            if (h_win != h_ref) {
                printf("FAIL: cached=%d direct=%d seed=%llu (%d,%d)\n", h_win, h_ref,
                       (unsigned long long)seeds[s], bx, bz);
                fail++;
            }
            /* h73 的定义：>=73 的高度；与完整高度必须一致 */
            if ((h_ref >= 73) != (h_ex >= 73)) {
                printf("FAIL: h73=%d exact=%d seed=%llu (%d,%d)\n", h_ref, h_ex,
                       (unsigned long long)seeds[s], bx, bz);
                fail++;
            }
            if (h_ref >= 73 && h_ref != h_ex) {
                printf("FAIL: h73=%d exact=%d (应相等) seed=%llu (%d,%d)\n", h_ref,
                       h_ex, (unsigned long long)seeds[s], bx, bz);
                fail++;
            }
        }
    }
    printf("selftest: %d cases, %d failures -> %s\n", n, fail, fail ? "FAIL" : "PASS");

    /* --- 共享 simplex 网格 / D_eff 过滤的等价性 --- */
    {
        static const int pos[][2] = {{0, 0},       {12345, -6789}, {1225660, 0},
                                     {-777, 65536}, {245133, -245133}};
        int p, f2 = 0, np = 0;
        for (p = 0; p < (int)(sizeof(pos) / sizeof(pos[0])); p++) {
            EsSurfaceNoise sn;
            EsPerlin island;
            EsEndWin wd, wf, wm;
            const int xc = pos[p][0], zc = pos[p][1];
            int i, j, bx, bz;

            es_init_surface_noise_end(&sn, 694195937ULL);
            es_end_island_init(&island, 694195937ULL);

            /* (1) 网格 depth == 未缓存 depth */
            if (!es_end_win_prepare_depth(&island, xc, zc, 16, &wd)) { f2++; continue; }
            for (j = 0; j < wd.ncz; j++)
                for (i = 0; i < wd.ncx; i++) {
                    double dref;
                    np++;
                    if (wd.isvoid[j][i]) continue;
                    dref = (double)es_end_height_noise(&island, wd.cx0 + i,
                                                       wd.cz0 + j, 0)
                         - 8.0;
                    if (fabs(dref - wd.d[j][i]) > 1e-9) {
                        printf("FAIL: grid depth %.6f != uncached %.6f at (%d,%d)\n",
                               wd.d[j][i], dref, wd.cx0 + i, wd.cz0 + j);
                        f2++;
                    }
                }

            /* (2) 全量路径 */
            if (!es_end_win_prepare(&sn, &island, xc, zc, 16, &wf)) { f2++; continue; }
            /* (3) 过滤路径：只标记 D_eff > 62 的列 */
            if (!es_end_win_prepare_depth(&island, xc, zc, 16, &wm)) { f2++; continue; }
            for (bz = zc - 16; bz <= zc + 16; bz++)
                for (bx = xc - 16; bx <= xc + 16; bx++)
                    if (es_end_win_deff(&wm, bx, bz) >= 62.0)
                        es_end_win_mark(&wm, bx, bz, &wm);
            es_end_win_fill_noise(&sn, &wm);

            for (bz = zc - 16; bz <= zc + 16; bz++)
                for (bx = xc - 16; bx <= xc + 16; bx++) {
                    const int hf = es_end_height73_cached(&wf, bx, bz);
                    const double deff = es_end_win_deff(&wf, bx, bz);
                    np++;
                    /* 过滤不得丢掉任何 >=73 的列（阈值语义：跳过 D_eff < 62） */
                    if (hf >= 73 && deff < 62.0) {
                        printf("FAIL: 过滤会丢命中 h=%d D_eff=%.4f at (%d,%d)\n", hf,
                               deff, bx, bz);
                        f2++;
                    }
                    if (deff >= 62.0) {
                        const int hm = es_end_height73_cached(&wm, bx, bz);
                        if (hm != hf) {
                            printf("FAIL: 过滤路径 %d != 全量 %d at (%d,%d)\n", hm, hf,
                                   bx, bz);
                            f2++;
                        }
                    }
                }
        }
        printf("selftest(depth grid + D_eff filter): %d checks, %d failures -> %s\n",
               np, f2, f2 ? "FAIL" : "PASS");
        fail += f2;
    }
    return fail ? 0 : 1;
}

int main(int argc, char **argv)
{
    Cfg cfg;
    std::vector<SeedRec> seeds;
    std::vector<Job> jobs;
    int argi;

    memset(&cfg, 0, sizeof(cfg));
    cfg.env_thr = 148.0;
    cfg.range = ES_RANGE_HI;
    cfg.window = 16;
    cfg.window_step = 1;
    cfg.period = ES_TRUE_PERIOD;
    cfg.loop_mode = LOOP_2D;
    cfg.threads = (int)std::thread::hardware_concurrency();
    if (cfg.threads < 1) cfg.threads = 1;
    cfg.max_seeds = 0;
    cfg.max_loop_points = 0;
    cfg.quiet = 0;
    cfg.depth_thr = 62.0;

    for (argi = 1; argi < argc; argi++) {
        if (!strcmp(argv[argi], "--hits") && argi + 1 < argc)
            cfg.hits_path = argv[++argi];
        else if (!strcmp(argv[argi], "--out") && argi + 1 < argc)
            cfg.out_path = argv[++argi];
        else if (!strcmp(argv[argi], "--envelope-thr") && argi + 1 < argc)
            cfg.env_thr = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--range") && argi + 1 < argc)
            cfg.range = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--window") && argi + 1 < argc)
            cfg.window = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--window-step") && argi + 1 < argc)
            cfg.window_step = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--period") && argi + 1 < argc)
            cfg.period = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--threads") && argi + 1 < argc)
            cfg.threads = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--max-seeds") && argi + 1 < argc)
            cfg.max_seeds = atoi(argv[++argi]);
        else if (!strcmp(argv[argi], "--max-loop-points") && argi + 1 < argc)
            cfg.max_loop_points = atoll(argv[++argi]);
        else if (!strcmp(argv[argi], "--depth-thr") && argi + 1 < argc)
            cfg.depth_thr = atof(argv[++argi]);
        else if (!strcmp(argv[argi], "--no-depth-filter"))
            cfg.depth_thr = -1e30;
        else if (!strcmp(argv[argi], "--quiet"))
            cfg.quiet = 1;
        else if (!strcmp(argv[argi], "--selftest")) return selftest() ? 0 : 1;
        else if (!strcmp(argv[argi], "--loop") && argi + 1 < argc) {
            argi++;
            if (!strcmp(argv[argi], "2d")) cfg.loop_mode = LOOP_2D;
            else if (!strcmp(argv[argi], "cross")) cfg.loop_mode = LOOP_CROSS;
            else if (!strcmp(argv[argi], "1d")) cfg.loop_mode = LOOP_1D;
            else {
                fprintf(stderr, "bad --loop (want 2d|cross|1d)\n");
                return 2;
            }
        } else if (!strcmp(argv[argi], "--help") || !strcmp(argv[argi], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[argi]);
            usage(argv[0]);
            return 2;
        }
    }

    if (!cfg.hits_path) {
        usage(argv[0]);
        return 2;
    }
    if (cfg.env_thr < ES_ENV_THR_LO || cfg.env_thr > ES_ENV_THR_HI) {
        fprintf(stderr, "--envelope-thr 需在 %.0f..%.0f 之间 (收到 %.3f)\n",
                ES_ENV_THR_LO, ES_ENV_THR_HI, cfg.env_thr);
        return 2;
    }
    if (cfg.range < ES_RANGE_LO || cfg.range > ES_RANGE_HI) {
        fprintf(stderr, "--range 需在 %d..%d 之间 (收到 %d)\n", ES_RANGE_LO,
                ES_RANGE_HI, cfg.range);
        return 2;
    }
    if (cfg.window < 0 || cfg.window > 64) {
        fprintf(stderr, "--window 需在 0..64 之间\n");
        return 2;
    }
    if (cfg.window_step < 1 || cfg.window_step > cfg.window + 1) {
        fprintf(stderr, "--window-step 需在 1..window+1 之间\n");
        return 2;
    }
    if (cfg.threads < 1 || cfg.threads > 1024) {
        fprintf(stderr, "--threads 非法\n");
        return 2;
    }
    if (!(cfg.period > 0.0)) {
        fprintf(stderr, "--period 非法\n");
        return 2;
    }
    if (!(cfg.depth_thr <= 73.0 && (cfg.depth_thr >= -110.0 || cfg.depth_thr < -1e29))) {
        fprintf(stderr, "--depth-thr 需在 -110..73 之间（或用 --no-depth-filter）\n");
        return 2;
    }

    if (!load_seeds(cfg.hits_path, &seeds)) return 1;

    {
        /* envelope 过滤 + 按 envelope 降序 */
        std::vector<SeedRec> keep;
        size_t i;
        for (i = 0; i < seeds.size(); i++)
            if (seeds[i].env > cfg.env_thr) keep.push_back(seeds[i]);
        std::sort(keep.begin(), keep.end(),
                  [](const SeedRec &a, const SeedRec &b) { return a.env > b.env; });
        if (cfg.max_seeds > 0 && (int)keep.size() > cfg.max_seeds)
            keep.resize((size_t)cfg.max_seeds);
        seeds.swap(keep);
    }
    if (seeds.empty()) {
        fprintf(stderr, "没有种子的 envelope > %.6f\n", cfg.env_thr);
        return 1;
    }
    fprintf(stderr, "seeds kept (envelope > %.6f): %zu\n", cfg.env_thr, seeds.size());

    {
        const int K = (int)floor((double)cfg.range / cfg.period);
        size_t si;
        int kx, kz;
        long long per_seed = 0;
        for (kx = -K; kx <= K; kx++) {
            for (kz = -K; kz <= K; kz++) {
                if (cfg.loop_mode == LOOP_1D && kz != 0) continue;
                if (cfg.loop_mode == LOOP_CROSS && !(kx == 0 || kz == 0)) continue;
                per_seed++;
            }
        }
        if (cfg.max_loop_points > 0 && per_seed > cfg.max_loop_points) {
            fprintf(stderr, "limit: each seed capped at %lld loop points\n",
                    cfg.max_loop_points);
        }
        for (si = 0; si < seeds.size(); si++) {
            long long cnt = 0;
            for (kx = -K; kx <= K; kx++) {
                for (kz = -K; kz <= K; kz++) {
                    Job j;
                    if (cfg.loop_mode == LOOP_1D && kz != 0) continue;
                    if (cfg.loop_mode == LOOP_CROSS && !(kx == 0 || kz == 0)) continue;
                    if (cfg.max_loop_points > 0 && cnt >= cfg.max_loop_points) break;
                    j.seed_idx = (int)si;
                    j.kx = kx;
                    j.kz = kz;
                    jobs.push_back(j);
                    cnt++;
                }
                if (cfg.max_loop_points > 0 && cnt >= cfg.max_loop_points) break;
            }
        }
    }

    return run_scan(&cfg, &seeds, &jobs) ? 0 : 1;
}
