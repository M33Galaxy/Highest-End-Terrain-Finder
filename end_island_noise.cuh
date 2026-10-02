/* end_island_noise.cuh
 *
 * 末地「真实地表高度」原语（host + device 通用）。逐字移植自 cubiomes-end：
 *   third_party/cubiomes-end/biomenoise.c : setEndSeed / getEndHeightNoise /
 *                                           sampleSurfaceNoiseBetween /
 *                                           sampleNoiseColumnEnd / getSurfaceHeight
 *   third_party/cubiomes-end/noise.c      : sampleSimplex2D / simplexGrad
 *   third_party/cubiomes-end/rng.h        : skipNextN
 *
 * ── 为什么需要这个文件 ────────────────────────────────────────────────────
 * end_surface_period_scan 只做 SurfaceNoise 包络 + 周期晶格判据，不含高度。
 * 而仓库里其它高度实现在远距离是**错的**：
 *   front_dragon_hsum.cu 的 end_depth_simple / stage2_hotpath.cuh 的 end_island_depth
 *   = 100 - sqrt(64*(cx^2+cz^2))，clamp[-100,80] 再 -8
 * 它只是「内岛」项（其注释即写明"此处邻域 rsq≪4096，与种子无关"）。|cell| 超过约 25
 * 后该项被 clamp 到 -108，此时 cy=18 需要 noise > 308（不可能），高度上限掉到 cy=14
 * （Y≈59）。±3000 万格处的地形完全来自 getEndHeightNoise 的 25×25 外环项，
 * 所以必须用这个真实实现。
 *
 * ── 关键结论（cubiomes biomenoise.c:602-609 的注释给出同一结论）──────────
 *   depth ∈ [-108, +72]，noise ∈ [-128, +128]（实测有 >128 的尾巴，约到 152）
 *   cy=18 的 upper_drop u = (78-18)/64 = 15/16，是「还能成实心」的最高 cell：
 *   实心要求 u·(noise+depth+3000) > 3000，cy≥19 时 u<15/16 已不够。
 *   => 高度 ≥ 73  ⟺  cy=18 内存在实心块（y=1/2/3 任一 noise>0）
 *      高度 74    ⟺  y=3 不成立且 y=2 成立
 *      高度 75    ⟺  y=3 成立
 *   所以只要 4 个 cell × celly{18,19} 共 8 个噪声值即可判定 ≥73，
 *   不必算 4 cell × celly 0..32 的 132 个值。
 *
 * ── 另一个必须保留的真实行为：1.14+ 的 overflow void ──────────────────────
 *   sampleNoiseColumnEnd 里 `rsq = x*x + z*z` 用 32 位溢出语义((int)rsq < 0)判空。
 *   远距离 cell 坐标会让它成立，该列整列变为 void（无地形）。这是真实生成行为，
 *   在 ±10^7 量级会否掉相当一部分坐标，不能省。
 *
 * 证据口径：相对 cubiomes-end `getEndSurfaceHeight`，不是 Official JAR。
 * 已用 tools 侧的 ground-truth 探针对拍（见 period_height_check --selftest）。
 */
#pragma once

#include <math.h>
#include <stdint.h>

#include "end_surface_noise.cuh"

/* ---- Java LCG 闭式跳进（cubiomes rng.h skipNextN 逐字移植）---- */
ES_INLINE void es_skip_next_n(uint64_t *seed, uint64_t n)
{
    uint64_t m = 1;
    uint64_t a = 0;
    uint64_t im = 0x5deece66dULL;
    uint64_t ia = 0xb;
    uint64_t k;
    for (k = n; k; k >>= 1) {
        if (k & 1) {
            m *= im;
            a = im * a + ia;
        }
        ia = (im + 1) * ia;
        im *= im;
    }
    *seed = *seed * m + a;
    *seed &= 0xffffffffffffULL;
}

/* ---- setEndSeed：外岛 simplex 噪声（cubiomes biomenoise.c:370）----
 * setSeed(seed) → consumeCount(17292) → perlinInit
 * 注意：这里初始化出来的是 PerlinNoise 结构，但只被当作 SimplexNoise 用
 * （sampleSimplex2D 只读 d[] 置换表）。 */
ES_FN void es_end_island_init(EsPerlin *island, uint64_t seed)
{
    uint64_t s;
    es_set_seed(&s, seed);
    es_skip_next_n(&s, 17292);
    es_perlin_init(island, &s);
}

/* ---- simplexGrad（cubiomes noise.c:294）---- */
ES_INLINE double es_simplex_grad(int idx, double x, double y, double z, double d)
{
    double con = d - x * x - y * y - z * z;
    if (con < 0) return 0;
    con *= con;
    return con * con * es_indexed_lerp((uint8_t)idx, x, y, z);
}

/* ---- sampleSimplex2D（cubiomes noise.c:303 逐字移植）---- */
ES_FN double es_sample_simplex_2d(const EsPerlin *noise, double x, double y)
{
    const double SKEW = 0.5 * (sqrt(3.0) - 1.0);
    const double UNSKEW = (3.0 - sqrt(3.0)) / 6.0;

    double hf = (x + y) * SKEW;
    int hx = (int)floor(x + hf);
    int hz = (int)floor(y + hf);
    double mhxz = (hx + hz) * UNSKEW;
    double x0 = x - (hx - mhxz);
    double y0 = y - (hz - mhxz);
    int offx = (x0 > y0);
    int offz = !offx;
    double x1 = x0 - offx + UNSKEW;
    double y1 = y0 - offz + UNSKEW;
    double x2 = x0 - 1.0 + 2.0 * UNSKEW;
    double y2 = y0 - 1.0 + 2.0 * UNSKEW;
    int gi0 = noise->d[0xff & (hz)];
    int gi1 = noise->d[0xff & (hz + offz)];
    int gi2 = noise->d[0xff & (hz + 1)];
    double t = 0;

    gi0 = noise->d[0xff & (gi0 + hx)];
    gi1 = noise->d[0xff & (gi1 + hx + offx)];
    gi2 = noise->d[0xff & (gi2 + hx + 1)];

    t += es_simplex_grad(gi0 % 12, x0, y0, 0.0, 0.5);
    t += es_simplex_grad(gi1 % 12, x1, y1, 0.0, 0.5);
    t += es_simplex_grad(gi2 % 12, x2, y2, 0.0, 0.5);
    return 70.0 * t;
}

/* ---- 外岛候选中心（getEndHeightNoise 内层条件的因式提取）----
 * 返回 1 表示 (rx,rz) 是一个合格的外岛中心，并给出 v^2。
 * v = (|rx|*3439 + |rz|*147) % 13 + 9，逐字保留 float32 运算与 unsigned 截断。 */
ES_FN int es_end_island_center(
    const EsPerlin *island, int64_t rx, int64_t rz, uint64_t *v2_out)
{
    int64_t rsq_s = rx * rx + rz * rz;
    uint64_t rsq = (uint64_t)rsq_s;
    unsigned int v;

    if (!(rsq > 4096)) return 0;
    if (!(es_sample_simplex_2d(island, (double)rx, (double)rz) < -0.9f)) return 0;

    v = (unsigned int)(fabsf((float)rx) * 3439.0f + fabsf((float)rz) * 147.0f) % 13 + 9;
    *v2_out = (uint64_t)v * (uint64_t)v;
    return 1;
}

/* ---- getEndHeightNoise（cubiomes biomenoise.c:527 逐字移植）----
 * x,z 为 **cell** 坐标（8 格/cell）。range=0 → 12（25×25 邻域）。
 * 外层 64*(x^2+z^2) 是内岛项，内层 25×25 的 min 是外岛项。 */
ES_FN float es_end_height_noise(const EsPerlin *island, int x, int z, int range)
{
    int hx = x / 2;
    int hz = z / 2;
    int oddx = x % 2;
    int oddz = z % 2;
    int i, j;
    int64_t h = 64 * (x * (int64_t)x + z * (int64_t)z);
    float ret;

    if (range == 0) range = 12;

    for (j = -range; j <= range; j++) {
        for (i = -range; i <= range; i++) {
            int64_t rx = hx + i;
            int64_t rz = hz + j;
            uint64_t v2;
            if (!es_end_island_center(island, rx, rz, &v2)) continue;
            {
                int64_t ox = oddx - i * 2;
                int64_t oz = oddz - j * 2;
                int64_t rsq2 = ox * ox + oz * oz;
                int64_t noise = rsq2 * (int64_t)v2;
                if (noise < h) h = noise;
            }
        }
    }

    ret = 100 - sqrtf((float)h);
    if (ret < -100.0f) ret = -100.0f;
    if (ret > 80.0f) ret = 80.0f;
    return ret;
}

/* ---- 1.14+ overflow void：cell 坐标下 (int)(x^2+z^2) < 0 → 整列 void ---- */
ES_INLINE int es_end_column_void(int cx, int cz)
{
    uint64_t rsq = (uint64_t)cx * (uint64_t)cx + (uint64_t)cz * (uint64_t)cz;
    return ((uint32_t)rsq >= 0x80000000u) ? 1 : 0;
}

/* upper_drop[y] = clamp((32+46-y)/64) = clamp((78-y)/64)；y 为 cell-y
 * lower_drop[y] = clamp((y-1)/7)                        （biomenoise.c:574/582） */
ES_INLINE double es_upper_drop(int y)
{
    double u = (78.0 - (double)y) / 64.0;
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    return u;
}

ES_INLINE double es_lower_drop(int y)
{
    double l = ((double)y - 1.0) / 7.0;
    if (l < 0.0) l = 0.0;
    if (l > 1.0) l = 1.0;
    return l;
}

/* ---- sampleSurfaceNoiseBetween(sn, cx, cy, cz, -128, +128)（biomenoise.c:60）---- */
ES_FN double es_end_surface_noise_at(const EsSurfaceNoise *sn, int cx, int cy, int cz)
{
    const double xzScale = ES_BASE_FREQ * ES_XZ_SCALE;
    const double yScale = ES_BASE_FREQ * ES_Y_SCALE;
    double vmin = 0.0;
    double vmax = 0.0;
    double persist = 1.0 / 32768.0;
    double amp = 64.0;
    int i;

    for (i = 15; i >= 0; i--) {
        const double dx = (double)cx * xzScale * persist;
        const double dz = (double)cz * xzScale * persist;
        const double sy = yScale * persist;
        const double dy = (double)cy * sy;

        vmin += es_sample_perlin(&sn->octmin[i], dx, dy, dz, sy, dy) * amp;
        vmax += es_sample_perlin(&sn->octmax[i], dx, dy, dz, sy, dy) * amp;
        if (vmin - amp > 128.0 && vmax - amp > 128.0) return 128.0;
        if (vmin + amp < -128.0 && vmax + amp < -128.0) return -128.0;

        amp *= 0.5;
        persist *= 2.0;
    }

    {
        const double xzStep = xzScale / ES_XZ_FACTOR;
        const double yStep = yScale / ES_Y_FACTOR;
        double vmain = 0.5;

        persist = 1.0 / 128.0;
        amp = 0.05 * 128.0;

        for (i = 7; i >= 0; i--) {
            const double dx = (double)cx * xzStep * persist;
            const double dz = (double)cz * xzStep * persist;
            const double sy = yStep * persist;
            const double dy = (double)cy * sy;

            vmain += es_sample_perlin(&sn->octmain[i], dx, dy, dz, sy, dy) * amp;
            if (vmain - amp > 1.0) return vmax;
            if (vmain + amp < 0.0) return vmin;

            amp *= 0.5;
            persist *= 2.0;
        }
        return es_clamped_lerp(vmain, vmin, vmax);
    }
}

/* ---- sampleNoiseColumnEnd 的单 cell 值（biomenoise.c:569）----
 * depth 由调用方给出（= es_end_height_noise(cx,cz,0) - 8），因为它是整个函数里
 * 最贵的部分（25×25 simplex），同一 cell 的 cy=18/19 两次取样必须复用。
 * 返回该 cell 在 cell-y = cy 处的列密度；*is_void 置 1 表示整列 void。 */
ES_FN double es_end_column_cell_depth(const EsSurfaceNoise *sn, int cx, int cy,
                                      int cz, double depth, int *is_void)
{
    double noise;
    double clamped;

    if (is_void) *is_void = 0;
    if (es_end_column_void(cx, cz)) {
        if (is_void) *is_void = 1;
        return 0.0;
    }
    if (es_lower_drop(cy) == 0.0) return -30.0;

    noise = es_end_surface_noise_at(sn, cx, cy, cz);
    clamped = noise + depth;
    clamped = es_lerp(es_upper_drop(cy), -3000.0, clamped);
    clamped = es_lerp(es_lower_drop(cy), -30.0, clamped);
    return clamped;
}

ES_FN double es_end_column_cell(const EsSurfaceNoise *sn, const EsPerlin *island,
                                int cx, int cy, int cz, int *is_void)
{
    const double depth = (double)es_end_height_noise(island, cx, cz, 0) - 8.0;
    return es_end_column_cell_depth(sn, cx, cy, cz, depth, is_void);
}

/* ---- lerp2/lerp3（cubiomes rng.h:459/465 逐字移植，参数序保持原样）---- */
ES_INLINE double es_cub_lerp2(
    double dx, double dy, double v00, double v10, double v01, double v11)
{
    return es_lerp(dy, es_lerp(dx, v00, v10), es_lerp(dx, v01, v11));
}

ES_INLINE double es_cub_lerp3(
    double dx, double dy, double dz, double v000, double v100, double v010,
    double v110, double v001, double v101, double v011, double v111)
{
    v000 = es_cub_lerp2(dx, dy, v000, v100, v010, v110);
    v001 = es_cub_lerp2(dx, dy, v001, v101, v011, v111);
    return es_lerp(dz, v000, v001);
}

/* ---- getSurfaceHeight 的插值核心（biomenoise.c:630）----
 * n00=(cx,cz) n01=(cx,cz+1) n10=(cx+1,cz) n11=(cx+1,cz+1)，各自 [celly] 与 [celly+1]。
 * 抽出来以便在任意 celly 上求值（本文件只用 18）。 */
ES_INLINE double es_end_interp_celly(double a00, double a01, double a10, double a11,
                                     double b00, double b01, double b10, double b11,
                                     double dx, double dz, double dy)
{
    /* 对应 lerp3(dy, dx, dz, v000,v010,v100,v110, v001,v011,v101,v111) */
    return es_cub_lerp3(dy, dx, dz, a00, b00, a10, b10, a01, b01, a11, b11);
}

/* ---- 快速判定：仅 cy=18（= 地表高度 ≥ 73 的充要条件）----
 * 返回 73/74/75；无实心块（高度 < 73，含 void）返回 -1。
 * 4 个 cell × celly{18,19} = 8 个列值（区间缓存由调用方负责，这里只算一次）。 */
ES_FN int es_end_height73(const EsSurfaceNoise *sn, const EsPerlin *island,
                          int bx, int bz)
{
    const int cellx = bx >> 3;
    const int cellz = bz >> 3;
    const double dx = (double)(bx & 7) / 8.0;
    const double dz = (double)(bz & 7) / 8.0;
    /* ncol[c][celly]，c: 0=(cx,cz) 1=(cx,cz+1) 2=(cx+1,cz) 3=(cx+1,cz+1) */
    double a[4], b[4];
    static const int dcx[4] = {0, 0, 1, 1};
    static const int dcz[4] = {0, 1, 0, 1};
    int c, y;

    for (c = 0; c < 4; c++) {
        int v18 = 0, v19 = 0;
        a[c] = es_end_column_cell(sn, island, cellx + dcx[c], 18, cellz + dcz[c], &v18);
        b[c] = es_end_column_cell(sn, island, cellx + dcx[c], 19, cellz + dcz[c], &v19);
        if (v18 || v19) return -1; /* 任一 cell void → 整列 void（NaN 传播等价） */
    }

    for (y = 3; y >= 1; y--) {
        const double dy = (double)y / 4.0;
        /* v000=a[0] v001=a[1] v100=a[2] v101=a[3]，下一层 v010=b[0] ... */
        const double noise = es_cub_lerp3(dy, dx, dz, a[0], b[0], a[2], b[2], a[1],
                                          b[1], a[3], b[3]);
        if (noise > 0.0) return 18 * 4 + y;
    }
    return -1;
}

/* ---- 窗口 cell 缓存 ----------------------------------------------------
 * 扫描热路径必需：一个 ±W 的方块窗口只覆盖约 (2W/8+2)^2 个 cell，
 * 而每个 cell 的 depth 要花 25×25 次 simplex。逐列直接调 es_end_height73
 * 会对同一 cell 重复算 4 次 depth，慢 ~30 倍。
 * W<=64 → ncx,ncz <= 19 < 24。 */
#define ES_END_WIN_MAXC 24

typedef struct EsEndWin {
    int cx0, cz0, ncx, ncz;
    double v18[ES_END_WIN_MAXC][ES_END_WIN_MAXC]; /* [j][i] j=cz 方向 */
    double v19[ES_END_WIN_MAXC][ES_END_WIN_MAXC];
    unsigned char isvoid[ES_END_WIN_MAXC][ES_END_WIN_MAXC];
} EsEndWin;

ES_FN int es_end_win_prepare(const EsSurfaceNoise *sn, const EsPerlin *island,
                            int xc, int zc, int w, EsEndWin *win)
{
    const int cx_lo = (xc - w) >> 3;
    const int cx_hi = ((xc + w) >> 3) + 1;
    const int cz_lo = (zc - w) >> 3;
    const int cz_hi = ((zc + w) >> 3) + 1;
    int i, j;

    win->cx0 = cx_lo;
    win->cz0 = cz_lo;
    win->ncx = cx_hi - cx_lo + 1;
    win->ncz = cz_hi - cz_lo + 1;
    if (win->ncx > ES_END_WIN_MAXC || win->ncz > ES_END_WIN_MAXC) return 0;

    for (j = 0; j < win->ncz; j++) {
        for (i = 0; i < win->ncx; i++) {
            const int cx = cx_lo + i;
            const int cz = cz_lo + j;
            if (es_end_column_void(cx, cz)) {
                win->isvoid[j][i] = 1;
                win->v18[j][i] = 0.0;
                win->v19[j][i] = 0.0;
                continue;
            }
            win->isvoid[j][i] = 0;
            {
                const double depth =
                    (double)es_end_height_noise(island, cx, cz, 0) - 8.0;
                win->v18[j][i] =
                    es_end_column_cell_depth(sn, cx, 18, cz, depth, NULL);
                win->v19[j][i] =
                    es_end_column_cell_depth(sn, cx, 19, cz, depth, NULL);
            }
        }
    }
    return 1;
}

/* 用缓存判定方块 (bx,bz)：返回 73/74/75；<73 或 void 返回 -1；缓存未覆盖返回 -2。 */
ES_FN int es_end_height73_cached(const EsEndWin *win, int bx, int bz)
{
    const int i0 = (bx >> 3) - win->cx0;
    const int j0 = (bz >> 3) - win->cz0;
    const double dx = (double)(bx & 7) / 8.0;
    const double dz = (double)(bz & 7) / 8.0;
    double a[4], b[4];
    int y;

    if (i0 < 0 || j0 < 0 || i0 + 1 >= win->ncx || j0 + 1 >= win->ncz) return -2;
    if (win->isvoid[j0][i0] || win->isvoid[j0][i0 + 1] || win->isvoid[j0 + 1][i0]
        || win->isvoid[j0 + 1][i0 + 1])
        return -1;

    a[0] = win->v18[j0][i0];         /* (cx,   cz  ) */
    b[0] = win->v19[j0][i0];
    a[1] = win->v18[j0 + 1][i0];     /* (cx,   cz+1) */
    b[1] = win->v19[j0 + 1][i0];
    a[2] = win->v18[j0][i0 + 1];     /* (cx+1, cz  ) */
    b[2] = win->v19[j0][i0 + 1];
    a[3] = win->v18[j0 + 1][i0 + 1]; /* (cx+1, cz+1) */
    b[3] = win->v19[j0 + 1][i0 + 1];

    for (y = 3; y >= 1; y--) {
        const double dy = (double)y / 4.0;
        const double noise = es_cub_lerp3(dy, dx, dz, a[0], b[0], a[2], b[2], a[1],
                                          b[1], a[3], b[3]);
        if (noise > 0.0) return 18 * 4 + y;
    }
    return -1;
}

/* ---- 完整高度（cubiomes getSurfaceHeight 等价，celly 31→0，blockspercell=4）----
 * 仅用于与 cubiomes 对拍；扫描热路径用 es_end_height73。 */
ES_FN int es_end_height_exact(const EsSurfaceNoise *sn, const EsPerlin *island,
                              int bx, int bz)
{
    const int cellx = bx >> 3;
    const int cellz = bz >> 3;
    const double dx = (double)(bx & 7) / 8.0;
    const double dz = (double)(bz & 7) / 8.0;
    static const int dcx[4] = {0, 0, 1, 1};
    static const int dcz[4] = {0, 1, 0, 1};
    /* col[c][cy]，cy = 0..32 */
    double col[4][33];
    int c, y, celly;

    for (c = 0; c < 4; c++) {
        int isv = 0;
        for (y = 0; y <= 32; y++) {
            col[c][y] = es_end_column_cell(sn, island, cellx + dcx[c], y,
                                           cellz + dcz[c], &isv);
            if (isv) return 0; /* void 列 → getSurfaceHeight 返回 0 */
        }
    }

    for (celly = 31; celly >= 0; celly--) {
        for (y = 3; y >= 0; y--) {
            const double dy = (double)y / 4.0;
            const double noise =
                es_cub_lerp3(dy, dx, dz, col[0][celly], col[0][celly + 1],
                             col[2][celly], col[2][celly + 1], col[1][celly],
                             col[1][celly + 1], col[3][celly], col[3][celly + 1]);
            if (noise > 0.0) return celly * 4 + y;
        }
    }
    return 0;
}
