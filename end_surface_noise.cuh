/* End SurfaceNoise primitives for CUDA / host parity.
 * Aligned with cubiomes initSurfaceNoise(DIM_END) + C++learning end_period_common.
 * Fidelity: relative to vendored cubiomes / C++learning period predicates — not Official JAR.
 */
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
/* Shared Perlin/init math: host upload + device kernels.
 * Device-only (clock64 / __constant__ LUT): use ES_DEVICE. */
#define ES_HD     __host__ __device__
#define ES_DEVICE __device__
#define ES_INLINE __host__ __device__ __forceinline__
#define ES_FN     ES_HD
#else
#define ES_HD     static
#define ES_DEVICE static
#define ES_INLINE static inline
#define ES_FN     static
#endif

/* Locked defaults (plan / C++learning). Y=72 → celly=18 (same cell as Y=73). */
#ifndef ES_WORLD_Y
#define ES_WORLD_Y 72
#endif
#define ES_CELLY (ES_WORLD_Y / 4)

/* Locked stage gates (Y=72 / celly=18). Chain: 55 / 60 / 100 / 110 / 130 / 137.
 * Coarse 57 rejected: ~10% FN among env>140 on 10× seed sample. */
#define ES_STEP1_MAX0 55.0
#define ES_STEP2_SUM2 100.0
#define ES_STEP3_SUM3 110.0
#define ES_STAGE2_SUM_MIN 130.0
#define ES_STAGE3_SUM_MIN 137.0

/* Default n55: ±64 9-pt oct15 max confirm before 14/13. */
#define ES_S1_NEIGH_RADIUS 64
#define ES_S1_NEIGH_THR 60.0

#define ES_STAGE1_RANGE 24320
#define ES_STAGE1_STEP 128
/* Optional stage1: step-256 oct15 screen → refine step-128 near hits.
 * Default off on GPU (enable --s1-hier256). thr=42 host: ~9% FN / ~3.4× samples. */
#define ES_S1_HIER256_STEP 256
#define ES_S1_HIER256_THR 42.0
/* Stage1: oct15 allow gradvec (min OR max) before full sample.
 * Masks fit win-side on end-hits-1000w env>140 (n=158).
 * TEMP default in end_surface_period_scan: allow1 (max_fail=1) topk=4. */
#define ES_S1_GRADVEC_MAX_FAIL 1
#ifdef __CUDACC__
#define ES_GRAD_MASK_ATTR __device__ __constant__
#else
#define ES_GRAD_MASK_ATTR static
#endif
ES_GRAD_MASK_ATTR uint16_t ES_GRAD_ALLOW1_K4[8] = {
    0x131, 0x126, 0x286, 0x22a, 0x443, 0x486, 0x84c, 0x88a};
ES_GRAD_MASK_ATTR uint16_t ES_GRAD_ALLOW1_K5[8] = {
    0x133, 0x127, 0x296, 0xa2a, 0x4c3, 0x4a6, 0x84d, 0x8ca};
ES_GRAD_MASK_ATTR uint16_t ES_GRAD_ALLOW1_K6[8] = {
    0x333, 0x927, 0x29e, 0xaaa, 0x4c7, 0x4ae, 0x84f, 0x8ce};
ES_GRAD_MASK_ATTR uint16_t ES_GRAD_ALLOW1_K7[8] = {
    0x337, 0x92f, 0x29f, 0xaae, 0x4cf, 0x4af, 0xa4f, 0x8cf};
ES_GRAD_MASK_ATTR uint16_t ES_GRAD_ALLOW1_K8[8] = {
    0x3b7, 0x9af, 0xa9f, 0xbae, 0x5cf, 0xcaf, 0xb4f, 0xccf};
#define ES_STAGE2_RANGE 48
#define ES_STAGE2_STEP 16
#define ES_STAGE3_RANGE 16
#define ES_STAGE3_STEP 2

#define ES_PERIOD 49026.65646
/* 周期平移半径。默认 = 一个**总周期**：T = 5*P = 245133.2823，±T/2 = ±122566
 * ⟹ kMax = floor(122566/P) = 2，5x5 = 25 个晶格点，正好覆盖全部 5 个相位类一次。
 * （上游原值 30000000；本仓库按"主程序只走一个总周期、远处交给 period_height_check
 * 按整周期 T 平移去验真实高度"的分工改为此值。改动会同时影响 CLI 默认与 usage 文本。） */
#define ES_PERIOD_RANGE 122566
#define ES_MAIN_PHASE_MOD 5
#define ES_YOFFSET_SCORE_MIN 1.5

#define ES_BRANCH_MIN 1
#define ES_BRANCH_MAX 2

#define ES_XZ_SCALE 2.0
#define ES_Y_SCALE 1.0
#define ES_XZ_FACTOR 80.0
#define ES_Y_FACTOR 160.0
#define ES_BASE_FREQ 684.412

typedef struct EsPerlin {
    uint8_t d[256 + 1];
    uint8_t h2;
    double a, b, c;
    double d2, t2;
} EsPerlin;

typedef struct EsSurfaceNoise {
    EsPerlin octmin[16];
    EsPerlin octmax[16];
    EsPerlin octmain[8];
} EsSurfaceNoise;

/* ---- Java Random (48-bit LCG) ---- */

ES_INLINE void es_set_seed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66dULL) & ((1ULL << 48) - 1);
}

ES_INLINE int es_next(uint64_t *seed, int bits)
{
    *seed = (*seed * 0x5deece66dULL + 0xbULL) & ((1ULL << 48) - 1);
    return (int)((int64_t)*seed >> (48 - bits));
}

ES_INLINE int es_next_int(uint64_t *seed, int n)
{
    int bits, val;
    const int m = n - 1;
    if ((m & n) == 0) {
        uint64_t x = (uint64_t)n * (uint64_t)es_next(seed, 31);
        return (int)((int64_t)x >> 31);
    }
    do {
        bits = es_next(seed, 31);
        val = bits % n;
    } while ((int32_t)((uint32_t)bits - val + m) < 0);
    return val;
}

ES_INLINE double es_next_double(uint64_t *seed)
{
    uint64_t x = (uint64_t)es_next(seed, 26);
    x <<= 27;
    x += (uint64_t)es_next(seed, 27);
    return (double)x / (double)(1ULL << 53);
}

/* ---- Perlin ---- */

ES_INLINE double es_lerp(double part, double a, double b)
{
    return a + part * (b - a);
}

ES_INLINE double es_clamped_lerp(double part, double from, double to)
{
    if (part <= 0.0) return from;
    if (part >= 1.0) return to;
    return es_lerp(part, from, to);
}

ES_INLINE double es_indexed_lerp(uint8_t idx, double a, double b, double c)
{
    switch (idx & 0xf) {
    case 0:  return a + b;
    case 1:  return -a + b;
    case 2:  return a - b;
    case 3:  return -a - b;
    case 4:  return a + c;
    case 5:  return -a + c;
    case 6:  return a - c;
    case 7:  return -a - c;
    case 8:  return b + c;
    case 9:  return -b + c;
    case 10: return b - c;
    case 11: return -b - c;
    case 12: return a + b;
    case 13: return -b + c;
    case 14: return -a + b;
    case 15: return -b - c;
    }
    return 0;
}

ES_FN void es_perlin_init(EsPerlin *noise, uint64_t *seed)
{
    int i;
    noise->a = es_next_double(seed) * 256.0;
    noise->b = es_next_double(seed) * 256.0;
    noise->c = es_next_double(seed) * 256.0;
    for (i = 0; i < 256; i++) noise->d[i] = (uint8_t)i;
    for (i = 0; i < 256; i++) {
        int j = es_next_int(seed, 256 - i) + i;
        uint8_t n = noise->d[i];
        noise->d[i] = noise->d[j];
        noise->d[j] = n;
    }
    noise->d[256] = noise->d[0];
    {
        double i2 = floor(noise->b);
        double d2 = noise->b - i2;
        noise->h2 = (uint8_t)(int)i2;
        noise->d2 = d2;
        noise->t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
    }
}

ES_FN double es_sample_perlin(
    const EsPerlin *noise, double d1, double d2, double d3, double yamp, double ymax)
{
    uint8_t h1, h2, h3;
    double t1, t2, t3;

    if (d2 == 0.0) {
        d2 = noise->d2;
        h2 = noise->h2;
        t2 = noise->t2;
    } else {
        d2 += noise->b;
        double i2 = floor(d2);
        d2 -= i2;
        h2 = (uint8_t)(int)i2;
        t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
    }

    d1 += noise->a;
    d3 += noise->c;
    {
        double i1 = floor(d1);
        double i3 = floor(d3);
        d1 -= i1;
        d3 -= i3;
        h1 = (uint8_t)(int)i1;
        h3 = (uint8_t)(int)i3;
    }
    t1 = d1 * d1 * d1 * (d1 * (d1 * 6.0 - 15.0) + 10.0);
    t3 = d3 * d3 * d3 * (d3 * (d3 * 6.0 - 15.0) + 10.0);

    if (yamp != 0.0) {
        double yclamp = ymax >= 0.0 && ymax < d2 ? ymax : d2;
        d2 -= floor(yclamp / yamp) * yamp;
    }

    {
        const uint8_t *idx = noise->d;
        uint8_t a1 = (uint8_t)(idx[h1] + h2);
        uint8_t b1 = (uint8_t)(idx[h1 + 1] + h2);
        uint8_t a2 = (uint8_t)(idx[a1] + h3);
        uint8_t b2 = (uint8_t)(idx[b1] + h3);
        uint8_t a3 = (uint8_t)(idx[a1 + 1] + h3);
        uint8_t b3 = (uint8_t)(idx[b1 + 1] + h3);

        double l1 = es_indexed_lerp(idx[a2], d1, d2, d3);
        double l2 = es_indexed_lerp(idx[b2], d1 - 1, d2, d3);
        double l3 = es_indexed_lerp(idx[a3], d1, d2 - 1, d3);
        double l4 = es_indexed_lerp(idx[b3], d1 - 1, d2 - 1, d3);
        double l5 = es_indexed_lerp(idx[a2 + 1], d1, d2, d3 - 1);
        double l6 = es_indexed_lerp(idx[b2 + 1], d1 - 1, d2, d3 - 1);
        double l7 = es_indexed_lerp(idx[a3 + 1], d1, d2 - 1, d3 - 1);
        double l8 = es_indexed_lerp(idx[b3 + 1], d1 - 1, d2 - 1, d3 - 1);

        l1 = es_lerp(t1, l1, l2);
        l3 = es_lerp(t1, l3, l4);
        l5 = es_lerp(t1, l5, l6);
        l7 = es_lerp(t1, l7, l8);
        l1 = es_lerp(t2, l1, l3);
        l5 = es_lerp(t2, l5, l7);
        return es_lerp(t3, l1, l5);
    }
}

ES_FN void es_init_surface_noise_end(EsSurfaceNoise *sn, uint64_t world_seed)
{
    uint64_t s;
    int i;
    es_set_seed(&s, world_seed);
    /* octaveInit omin=-15 len=16 → end=0: init [0] then [1..15] */
    for (i = 0; i < 16; i++) es_perlin_init(&sn->octmin[i], &s);
    for (i = 0; i < 16; i++) es_perlin_init(&sn->octmax[i], &s);
    for (i = 0; i < 8; i++) es_perlin_init(&sn->octmain[i], &s);
}

ES_FN void es_sample_minmax_octave(
    const EsSurfaceNoise *sn, int cellx, int celly, int cellz, int i,
    double amp, double persist, double *cmin, double *cmax)
{
    const double xzScale = ES_BASE_FREQ * ES_XZ_SCALE;
    const double yScale = ES_BASE_FREQ * ES_Y_SCALE;
    const double dx = (double)cellx * xzScale * persist;
    const double dz = (double)cellz * xzScale * persist;
    const double sy = yScale * persist;
    const double dy_n = (double)celly * sy;
    *cmin = es_sample_perlin(&sn->octmin[i], dx, dy_n, dz, sy, dy_n) * amp;
    *cmax = es_sample_perlin(&sn->octmax[i], dx, dy_n, dz, sy, dy_n) * amp;
}

/* weirdfinder-style dedup: map raw 0..15 → 0..11 */
ES_INLINE uint8_t es_grad_dedup(uint8_t g)
{
    return (uint8_t)((g < 12) ? g : (g - 12));
}

ES_FN void es_corner_grad_ids(
    const EsPerlin *noise, double d1, double d2, double d3, double yamp, double ymax,
    uint8_t out8[8])
{
    uint8_t h1, h2, h3;
    const uint8_t *idx;

    if (d2 == 0.0) {
        d2 = noise->d2;
        h2 = noise->h2;
    } else {
        d2 += noise->b;
        {
            double i2 = floor(d2);
            d2 -= i2;
            h2 = (uint8_t)(int)i2;
        }
    }
    d1 += noise->a;
    d3 += noise->c;
    {
        double i1 = floor(d1);
        double i3 = floor(d3);
        d1 -= i1;
        d3 -= i3;
        h1 = (uint8_t)(int)i1;
        h3 = (uint8_t)(int)i3;
    }
    if (yamp != 0.0) {
        double yclamp = ymax >= 0.0 && ymax < d2 ? ymax : d2;
        d2 -= floor(yclamp / yamp) * yamp;
        (void)d1;
        (void)d3;
    }
    idx = noise->d;
    {
        uint8_t a1 = (uint8_t)(idx[h1] + h2);
        uint8_t b1 = (uint8_t)(idx[h1 + 1] + h2);
        uint8_t a2 = (uint8_t)(idx[a1] + h3);
        uint8_t b2 = (uint8_t)(idx[b1] + h3);
        uint8_t a3 = (uint8_t)(idx[a1 + 1] + h3);
        uint8_t b3 = (uint8_t)(idx[b1 + 1] + h3);
        out8[0] = es_grad_dedup((uint8_t)(idx[a2] & 15));
        out8[1] = es_grad_dedup((uint8_t)(idx[b2] & 15));
        out8[2] = es_grad_dedup((uint8_t)(idx[a3] & 15));
        out8[3] = es_grad_dedup((uint8_t)(idx[b3] & 15));
        out8[4] = es_grad_dedup((uint8_t)(idx[a2 + 1] & 15));
        out8[5] = es_grad_dedup((uint8_t)(idx[b2 + 1] & 15));
        out8[6] = es_grad_dedup((uint8_t)(idx[a3 + 1] & 15));
        out8[7] = es_grad_dedup((uint8_t)(idx[b3 + 1] & 15));
    }
}

ES_INLINE int es_gradvec_allow(
    const uint16_t masks[8], const uint8_t gs[8], int max_fail)
{
    int fail = 0;
    int c;
    for (c = 0; c < 8; c++) {
        if (((unsigned)masks[c] & (1u << gs[c])) == 0u) {
            fail++;
            if (fail > max_fail) return 0;
        }
    }
    return 1;
}

/* Pass if octmin[15] OR octmax[15] satisfies allow (max_fail=0 → allow0). */
ES_DEVICE int es_oct15_gradvec_or(
    const EsSurfaceNoise *sn, int cellx, int celly, int cellz,
    const uint16_t *masks, int max_fail)
{
    const double xzScale = ES_BASE_FREQ * ES_XZ_SCALE;
    const double yScale = ES_BASE_FREQ * ES_Y_SCALE;
    const double persist = 1.0 / 32768.0;
    const double dx = (double)cellx * xzScale * persist;
    const double dz = (double)cellz * xzScale * persist;
    const double sy = yScale * persist;
    const double dy_n = (double)celly * sy;
    uint8_t gs[8];
    if (max_fail < 0) max_fail = 0;
    es_corner_grad_ids(&sn->octmin[15], dx, dy_n, dz, sy, dy_n, gs);
    if (es_gradvec_allow(masks, gs, max_fail)) return 1;
    es_corner_grad_ids(&sn->octmax[15], dx, dy_n, dz, sy, dy_n, gs);
    return es_gradvec_allow(masks, gs, max_fail);
}

ES_DEVICE const uint16_t *es_grad_allow1_topk(int top_k)
{
    if (top_k <= 4) return ES_GRAD_ALLOW1_K4;
    if (top_k == 5) return ES_GRAD_ALLOW1_K5;
    if (top_k == 6) return ES_GRAD_ALLOW1_K6;
    if (top_k == 7) return ES_GRAD_ALLOW1_K7;
    return ES_GRAD_ALLOW1_K8;
}

ES_FN double es_sample_vmain(const EsSurfaceNoise *sn, int cellx, int celly, int cellz)
{
    const double xzScale = ES_BASE_FREQ * ES_XZ_SCALE;
    const double yScale = ES_BASE_FREQ * ES_Y_SCALE;
    const double xzStep = xzScale / ES_XZ_FACTOR;
    const double yStep = yScale / ES_Y_FACTOR;
    double vmain = 0.5;
    double persist = 1.0 / 128.0;
    double amp = 6.4;
    int i;
    for (i = 7; i >= 0; i--) {
        const double dx = (double)cellx * xzStep * persist;
        const double dz = (double)cellz * xzStep * persist;
        const double sy = yStep * persist;
        const double dy_n = (double)celly * sy;
        vmain += es_sample_perlin(&sn->octmain[i], dx, dy_n, dz, sy, dy_n) * amp;
        amp *= 0.5;
        persist *= 2.0;
    }
    return vmain;
}

ES_FN void es_sample_envelope(
    const EsSurfaceNoise *sn, int worldX, int worldZ, int celly,
    double *vmin, double *vmax)
{
    const int cellx = worldX >> 3;
    const int cellz = worldZ >> 3;
    double persist = 1.0 / 32768.0;
    double amp = 64.0;
    int i;
    *vmin = 0.0;
    *vmax = 0.0;
    for (i = 15; i >= 0; i--) {
        double cmin, cmax;
        es_sample_minmax_octave(sn, cellx, celly, cellz, i, amp, persist, &cmin, &cmax);
        *vmin += cmin;
        *vmax += cmax;
        amp *= 0.5;
        persist *= 2.0;
    }
}

ES_INLINE double es_branch_value(int branch, double vmin, double vmax)
{
    return branch == ES_BRANCH_MIN ? vmin : vmax;
}

/* Stage1 oct15→14→13. If neigh_radius>0: after center max0>coarse_thr,
 * sample 3×3 at ±neigh_radius, require max(max0)>neigh_thr, then continue
 * 14/13 at the argmax of those 9 (updates worldX/worldZ).
 * fail_step (optional): 15 / 16=neigh / 14 / 13 / 1=branch.
 * grad_allow1: optional oct15 allow masks (min OR max); NULL disables.
 * grad_max_fail: 0=allow0, 1=allow1 (default), 2=soft.
 * On device, time_steps may accumulate clock64 into cyc15/14/13 (neigh probes → cyc15).
 */
ES_DEVICE int es_stage1_three_step_ex(
    const EsSurfaceNoise *sn, int *worldX, int *worldZ, int celly,
    double coarse_thr, int neigh_radius, double neigh_thr,
    const uint16_t *grad_allow1, int grad_max_fail,
    int *branchMin, int *branchMax, double *sum3Max, int *fail_step
#ifdef __CUDACC__
    , unsigned long long *cyc15, unsigned long long *cyc14, unsigned long long *cyc13,
    int time_steps
#endif
    )
{
    int wx = *worldX;
    int wz = *worldZ;
    int cellx = wx >> 3;
    int cellz = wz >> 3;
    double persist = 1.0 / 32768.0;
    double amp = 64.0;
    double cmin, cmax;
    double sum_max = 0.0;
    double vmin_partial = 0.0;
    double vmax_partial = 0.0;
    double max0;
#ifdef __CUDACC__
    unsigned long long t0 = 0, t1 = 0;
#endif

#ifdef __CUDA_ARCH__
    if (time_steps) t0 = clock64();
#endif
    if (grad_allow1
        && !es_oct15_gradvec_or(
               sn, cellx, celly, cellz, grad_allow1, grad_max_fail)) {
#ifdef __CUDA_ARCH__
        if (time_steps && cyc15) {
            t1 = clock64();
            *cyc15 += t1 - t0;
        }
#endif
        if (fail_step) *fail_step = 15;
        return 0;
    }
    es_sample_minmax_octave(sn, cellx, celly, cellz, 15, amp, persist, &cmin, &cmax);
    max0 = cmin > cmax ? cmin : cmax;
    if (max0 <= coarse_thr) {
#ifdef __CUDA_ARCH__
        if (time_steps && cyc15) {
            t1 = clock64();
            *cyc15 += t1 - t0;
        }
#endif
        if (fail_step) *fail_step = 15;
        return 0;
    }
    sum_max = max0;
    vmin_partial = cmin;
    vmax_partial = cmax;

    if (neigh_radius > 0) {
        int ox, oz;
        int best_x = wx, best_z = wz;
        double best_max0 = max0;
        double best_cmin = cmin, best_cmax = cmax;
        for (oz = -neigh_radius; oz <= neigh_radius; oz += neigh_radius) {
            for (ox = -neigh_radius; ox <= neigh_radius; ox += neigh_radius) {
                double m;
                int ncx, ncz;
                if (ox == 0 && oz == 0) continue;
                ncx = (wx + ox) >> 3;
                ncz = (wz + oz) >> 3;
                if (grad_allow1
                    && !es_oct15_gradvec_or(
                           sn, ncx, celly, ncz, grad_allow1, grad_max_fail))
                    continue;
                es_sample_minmax_octave(
                    sn, ncx, celly, ncz, 15, amp, persist, &cmin, &cmax);
                m = cmin > cmax ? cmin : cmax;
                if (m > best_max0) {
                    best_max0 = m;
                    best_cmin = cmin;
                    best_cmax = cmax;
                    best_x = wx + ox;
                    best_z = wz + oz;
                }
            }
        }
#ifdef __CUDA_ARCH__
        if (time_steps && cyc15) {
            t1 = clock64();
            *cyc15 += t1 - t0;
        }
#endif
        if (best_max0 <= neigh_thr) {
            if (fail_step) *fail_step = 16;
            return 0;
        }
        wx = best_x;
        wz = best_z;
        *worldX = wx;
        *worldZ = wz;
        sum_max = best_max0;
        vmin_partial = best_cmin;
        vmax_partial = best_cmax;
        cellx = wx >> 3;
        cellz = wz >> 3;
    } else {
#ifdef __CUDA_ARCH__
        if (time_steps && cyc15) {
            t1 = clock64();
            *cyc15 += t1 - t0;
        }
#endif
    }

    persist *= 2.0;
    amp = 32.0;
#ifdef __CUDA_ARCH__
    if (time_steps) t0 = clock64();
#endif
    es_sample_minmax_octave(sn, cellx, celly, cellz, 14, amp, persist, &cmin, &cmax);
#ifdef __CUDA_ARCH__
    if (time_steps && cyc14) {
        t1 = clock64();
        *cyc14 += t1 - t0;
    }
#endif
    sum_max += (cmin > cmax ? cmin : cmax);
    if (sum_max <= ES_STEP2_SUM2) {
        if (fail_step) *fail_step = 14;
        return 0;
    }
    vmin_partial += cmin;
    vmax_partial += cmax;

    persist *= 2.0;
    amp = 16.0;
#ifdef __CUDA_ARCH__
    if (time_steps) t0 = clock64();
#endif
    es_sample_minmax_octave(sn, cellx, celly, cellz, 13, amp, persist, &cmin, &cmax);
#ifdef __CUDA_ARCH__
    if (time_steps && cyc13) {
        t1 = clock64();
        *cyc13 += t1 - t0;
    }
#endif
    sum_max += (cmin > cmax ? cmin : cmax);
    if (sum_max <= ES_STEP3_SUM3) {
        if (fail_step) *fail_step = 13;
        return 0;
    }

    *branchMin = (vmin_partial + cmin > ES_STEP3_SUM3) ? 1 : 0;
    *branchMax = (vmax_partial + cmax > ES_STEP3_SUM3) ? 1 : 0;
    *sum3Max = sum_max;
    if (!*branchMin && !*branchMax) {
        if (fail_step) *fail_step = 1;
        return 0;
    }
    if (fail_step) *fail_step = 0;
    return 1;
}

ES_DEVICE int es_stage1_three_step(
    const EsSurfaceNoise *sn, int worldX, int worldZ, int celly,
    int *branchMin, int *branchMax, double *sum3Max)
{
    int wx = worldX, wz = worldZ;
    return es_stage1_three_step_ex(
        sn, &wx, &wz, celly, ES_STEP1_MAX0, 0, 0.0, NULL, ES_S1_GRADVEC_MAX_FAIL,
        branchMin, branchMax, sum3Max, NULL
#ifdef __CUDACC__
        , NULL, NULL, NULL, 0
#endif
        );
}

/* ---- y_offset prefilter (no full SurfaceNoise) ---- */

ES_FN void es_rng_burn_perlin(uint64_t *rnd)
{
    int i;
    es_next_double(rnd);
    es_next_double(rnd);
    es_next_double(rnd);
    for (i = 0; i < 256; i++) es_next_int(rnd, 256 - i);
}

ES_FN double es_rng_read_perlin_yoffset(uint64_t *rnd)
{
    int i;
    es_next_double(rnd);
    {
        double b = es_next_double(rnd) * 256.0;
        es_next_double(rnd);
        for (i = 0; i < 256; i++) es_next_int(rnd, 256 - i);
        return b;
    }
}

ES_FN double es_rng_peek_perlin_yoffset(uint64_t *rnd)
{
    es_next_double(rnd);
    return es_next_double(rnd) * 256.0;
}

ES_INLINE double es_yoffset_ef_end(double yOffset, double persist, int celly)
{
    const double sy = ES_BASE_FREQ * persist;
    const double t = (double)celly * sy + yOffset;
    return t - floor(t);
}

/* Legacy gain score (kept for comparison). Prefer es_seed_phase_logit. */
ES_INLINE double es_yoffset_gain(double ef)
{
    if (ef >= 0.58 && ef <= 0.88) return 1.0;
    if ((ef >= 0.48 && ef < 0.58) || (ef > 0.88 && ef <= 0.93)) return 0.6;
    if (ef >= 0.38 && ef < 0.48) return 0.3;
    return 0.0;
}

ES_FN double es_seed_yoffset_score(uint64_t seed, int celly)
{
    uint64_t rnd;
    int i;
    double bmin14, bmin15, bmax14, bmax15;
    double p15 = 1.0 / 32768.0;
    double p14 = 1.0 / 16384.0;
    double g15, g14;

    es_set_seed(&rnd, seed);
    for (i = 0; i < 14; i++) es_rng_burn_perlin(&rnd);
    bmin14 = es_rng_read_perlin_yoffset(&rnd);
    bmin15 = es_rng_read_perlin_yoffset(&rnd);
    for (i = 0; i < 14; i++) es_rng_burn_perlin(&rnd);
    bmax14 = es_rng_read_perlin_yoffset(&rnd);
    bmax15 = es_rng_peek_perlin_yoffset(&rnd);

    {
        double g15a = es_yoffset_gain(es_yoffset_ef_end(bmin15, p15, celly));
        double g15b = es_yoffset_gain(es_yoffset_ef_end(bmax15, p15, celly));
        g15 = g15a > g15b ? g15a : g15b;
    }
    {
        double g14a = es_yoffset_gain(es_yoffset_ef_end(bmin14, p14, celly));
        double g14b = es_yoffset_gain(es_yoffset_ef_end(bmax14, p14, celly));
        g14 = g14a > g14b ? g14a : g14b;
    }
    return g15 + 0.5 * g14;
}

#if defined(__cplusplus)
#include "end_phase_lut.cuh"

/* True y_offset phase LUT: ef bins; gate = min OR max. */
ES_INLINE int es_phase_bin(double ef)
{
    int b = (int)(ef * (double)end_phase_lut::kBins);
    if (b < 0) return 0;
    if (b >= end_phase_lut::kBins) return end_phase_lut::kBins - 1;
    return b;
}

/* Returns max(score_min, score_max). Pass iff either side clears threshold. */
ES_DEVICE float es_seed_phase_logit(uint64_t seed, int celly)
{
    uint64_t rnd;
    int i;
    double bmin14, bmin15, bmax14, bmax15;
    const double p15 = 1.0 / 32768.0;
    const double p14 = 1.0 / 16384.0;
    int b0, b1, b2, b3;
    float smin, smax;

    es_set_seed(&rnd, seed);
    for (i = 0; i < 14; i++) es_rng_burn_perlin(&rnd);
    bmin14 = es_rng_read_perlin_yoffset(&rnd);
    bmin15 = es_rng_read_perlin_yoffset(&rnd);
    for (i = 0; i < 14; i++) es_rng_burn_perlin(&rnd);
    bmax14 = es_rng_read_perlin_yoffset(&rnd);
    bmax15 = es_rng_peek_perlin_yoffset(&rnd);

    b0 = es_phase_bin(es_yoffset_ef_end(bmin15, p15, celly));
    b1 = es_phase_bin(es_yoffset_ef_end(bmax15, p15, celly));
    b2 = es_phase_bin(es_yoffset_ef_end(bmin14, p14, celly));
    b3 = es_phase_bin(es_yoffset_ef_end(bmax14, p14, celly));
    smin = end_phase_lut::kInterceptMin
         + end_phase_lut::kWeight[0][b0]
         + end_phase_lut::kWeight[2][b2];
    smax = end_phase_lut::kInterceptMax
         + end_phase_lut::kWeight[1][b1]
         + end_phase_lut::kWeight[3][b3];
    return smin > smax ? smin : smax;
}

ES_DEVICE int es_seed_phase_pass(uint64_t seed, int celly, float threshold)
{
    return es_seed_phase_logit(seed, celly) + 1e-6f >= threshold;
}
#endif /* __cplusplus */


ES_INLINE int es_mod_phase(int k)
{
    int r = k % ES_MAIN_PHASE_MOD;
    return r < 0 ? r + ES_MAIN_PHASE_MOD : r;
}

ES_INLINE int es_vmain_pass(int branch, double vmain)
{
    if (branch == ES_BRANCH_MIN) return vmain < 0.0;
    return vmain > 1.0;
}

ES_INLINE int es_skip_parity(int blockX, int blockZ)
{
    const int64_t x = blockX;
    const int64_t z = blockZ;
    const int64_t q = (x * x + z * z) >> 37;
    return (q & 1) != 0;
}
