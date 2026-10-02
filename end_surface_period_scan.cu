/*
 * end_surface_period_scan.cu
 *
 * End SurfaceNoise period lattice + multi-stage envelope filter (C++learning 1C+2B).
 * Y=72 → celly=18. Stage1 default: step 128 + n55 + block-par.
 *
 * Fidelity: relative to cubiomes initSurfaceNoise(DIM_END) + C++learning
 * end_period_common predicates — not Official JAR. Islands/height omitted (v1).
 *
 *   end_surface_period_scan --start-seed A --end-seed B --out hits.csv
 *     [--stage1-step 128] [--period-range N] [--no-yoffset]
 *     [--s1-neigh|--no-s1-neigh] [--s1-coarse-thr 55]
 *     [--s1-hier256|--no-s1-hier256] [--s1-hier256-thr 42]
 *     [--celly 18] [--max-hits N] [--profile] [--profile-s1]
 *
 * Default stage1 (n55 + grid-par + hier256@42): step-256 oct15 screen, then
 *   step-128 n55 three_step only near screen hits; sum2/sum3 100/110; stage2/3 130/137.
 * TEMP: --profile coarse stages + stage1 fail-step counts.
 * TEMP: --profile-s1 also subsampled clock64 for oct15/14/13 (~1/16 pts).
 */

#include <cuda_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "end_surface_noise.cuh"

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct HitRow {
    int64_t seed;
    int branch; /* ES_BRANCH_MIN or ES_BRANCH_MAX */
    int stage1_x, stage1_z;
    double envelope;
    int peak_x, peak_z;
    int tx, tz;
    double vmain;
} HitRow;

typedef struct ScanConfig {
    int celly;
    int stage1_range;
    int stage1_step;
    int stage2_range;
    int stage2_step;
    int stage3_range;
    int stage3_step;
    double period;
    int period_range;
    float phase_thr; /* <= -1e8f disables phase prefilter */
    int max_hits;
    int profile; /* TEMP: 1=coarse+s1 counts, 2=+s1 step clocks (subsampled) */
    /* stage1 oct15: coarse center thr; optional ±r 9-pt max confirm before 14/13 */
    double s1_coarse_thr;
    int s1_neigh_radius; /* 0 = off */
    double s1_neigh_thr;
    /* 1 = phase-compact + block-per-seed grid-parallel stage1 (default). */
    int s1_grid_par;
    int s1_grid_par_threads; /* threads per seed-block */
    /* 1 = step-256 oct15 screen then step-128 refine near marks (default off). */
    int s1_hier256;
    int s1_hier256_step;
    double s1_hier256_thr;
    /* Refine half = ring * hier_step (1→±256 5×5, 2→±512 9×9). */
    int s1_hier256_ring;
    /* 1 = only three_step at screen hot centers (no neighborhood). */
    int s1_hier256_centers;
    /* 1 = oct15 allow1 gradvec (min OR max) before sample; topk selects mask. */
    int s1_gradvec;
    int s1_gradvec_topk; /* 4..8; masks from end-hits-1000w env>140 */
    int s1_gradvec_max_fail; /* 0=allow0, 1=allow1 */
} ScanConfig;

/* TEMP profiling — thread-cycle sums (clock64), not wall time. */
enum {
    PROF_PHASE = 0,
    PROF_INIT,
    PROF_STAGE1,
    PROF_STAGE2,
    PROF_STAGE3,
    PROF_REFINE,
    PROF_PERIOD,
    PROF_N
};

enum {
    PROF_S1_OCT15 = 0,
    PROF_S1_OCT14,
    PROF_S1_OCT13,
    PROF_S1_N
};

typedef struct StageProf {
    unsigned long long cycles[PROF_N];
    unsigned long long s1_cycles[PROF_S1_N]; /* only when profile>=2; subsampled */
    unsigned long long n_seed;
    unsigned long long n_phase_fail;
    unsigned long long n_phase_pass;
    unsigned long long n_stage1_fail;
    unsigned long long n_stage1_pass;
    unsigned long long n_stage2_try;
    unsigned long long n_stage2_pass;
    unsigned long long n_stage3_pass;
    unsigned long long n_refine;
    unsigned long long n_period_call;
    /* stage1 grid-point funnel (all points; cheap integer adds) */
    unsigned long long n_s1_pts;
    unsigned long long n_s1_fail15;
    unsigned long long n_s1_fail_neigh; /* fail_step==16 */
    unsigned long long n_s1_fail14;
    unsigned long long n_s1_fail13;
    unsigned long long n_s1_fail_br;
    unsigned long long n_s1_pts_pass;
    unsigned long long n_s1_timed_pts; /* subsampled points with clocks */
} StageProf;

__device__ __forceinline__ void prof_add_cycles(
    StageProf *prof, int slot, unsigned long long dt)
{
    if (prof) atomicAdd(&prof->cycles[slot], dt);
}

/* fail_step: 15/16=neigh/14/13/1=branch; return 1 if pass.
 * Relocates *worldX/*worldZ when neighborhood confirm selects a better corner.
 */
__device__ int es_stage1_three_step_detail(
    const EsSurfaceNoise *sn, int *worldX, int *worldZ, int celly,
    double coarse_thr, int neigh_radius, double neigh_thr,
    const uint16_t *grad_allow1, int grad_max_fail,
    int *branchMin, int *branchMax, double *sum3Max, int *fail_step,
    unsigned long long *cyc15, unsigned long long *cyc14, unsigned long long *cyc13,
    int time_steps)
{
    return es_stage1_three_step_ex(
        sn, worldX, worldZ, celly, coarse_thr, neigh_radius, neigh_thr,
        grad_allow1, grad_max_fail, branchMin, branchMax, sum3Max, fail_step,
        cyc15, cyc14, cyc13, time_steps);
}

__device__ __forceinline__ int es_hier256_n_axis(int stage1_range, int hstep)
{
    return (2 * (stage1_range / hstep)) + 1;
}

__device__ __forceinline__ int es_hier256_bit_index(
    int wx, int wz, int stage1_range, int hstep, int n_axis)
{
    const int ix = (wx + stage1_range) / hstep;
    const int iz = (wz + stage1_range) / hstep;
    return iz * n_axis + ix;
}

__device__ __forceinline__ void es_hier256_bit_set(unsigned *bits, int idx)
{
    atomicOr(&bits[idx >> 5], 1u << (idx & 31));
}

__device__ __forceinline__ int es_hier256_bit_test(const unsigned *bits, int idx)
{
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}

/* P covered if some hot H within chebyshev≤2*hstep induces a mark M with
 * chebyshev(P,M)≤hstep (matches host probe_hier256 geometry). */
__device__ int es_hier256_covers(
    int wx, int wz, int stage1_range, int hstep, const unsigned *hot, int n_axis)
{
    int ax = (wx >= 0) ? ((wx + hstep / 2) / hstep) * hstep
                       : -(((-wx) + hstep / 2) / hstep) * hstep;
    int az = (wz >= 0) ? ((wz + hstep / 2) / hstep) * hstep
                       : -(((-wz) + hstep / 2) / hstep) * hstep;
    int hx, hz, dx, dz;

    for (hz = az - 2 * hstep; hz <= az + 2 * hstep; hz += hstep) {
        for (hx = ax - 2 * hstep; hx <= ax + 2 * hstep; hx += hstep) {
            int idx;
            if (hx < -stage1_range || hx > stage1_range || hz < -stage1_range
                || hz > stage1_range)
                continue;
            if (hx > wx) {
                if (hx - wx > 2 * hstep) continue;
            } else if (wx - hx > 2 * hstep)
                continue;
            if (hz > wz) {
                if (hz - wz > 2 * hstep) continue;
            } else if (wz - hz > 2 * hstep)
                continue;
            idx = es_hier256_bit_index(hx, hz, stage1_range, hstep, n_axis);
            if (!es_hier256_bit_test(hot, idx)) continue;
            for (dz = -hstep; dz <= hstep; dz += hstep) {
                for (dx = -hstep; dx <= hstep; dx += hstep) {
                    const int mx = hx + dx;
                    const int mz = hz + dz;
                    const int ex = wx > mx ? wx - mx : mx - wx;
                    const int ez = wz > mz ? wz - mz : mz - wz;
                    if (ex <= hstep && ez <= hstep) return 1;
                }
            }
        }
    }
    return 0;
}

__device__ int es_oct15_max0_gt(
    const EsSurfaceNoise *sn, int worldX, int worldZ, int celly, double thr)
{
    double cmin, cmax, m;
    es_sample_minmax_octave(
        sn, worldX >> 3, celly, worldZ >> 3, 15, 64.0, 1.0 / 32768.0, &cmin, &cmax);
    m = cmin > cmax ? cmin : cmax;
    return m > thr;
}

__device__ int es_scan_stage1_d(
    const EsSurfaceNoise *sn, int celly, int stage1_range, int stage1_step,
    double coarse_thr, int neigh_radius, double neigh_thr,
    int hier256, int hier_step, double hier_thr,
    const uint16_t *grad_allow1, int grad_max_fail,
    int *bestX, int *bestZ, int *branchMin, int *branchMax, double *bestSum3,
    StageProf *prof, int profile_level)
{
    int found = 0;
    double bestMinSum3 = -1.0;
    double bestMaxSum3 = -1.0;
    int bestMinX = 0, bestMinZ = 0;
    int bestMaxX = 0, bestMaxZ = 0;
    int dz, dx;
    unsigned long long loc_pts = 0, loc_f15 = 0, loc_fneigh = 0, loc_f14 = 0;
    unsigned long long loc_f13 = 0, loc_fbr = 0, loc_pass = 0, loc_timed = 0;
    unsigned long long loc_c15 = 0, loc_c14 = 0, loc_c13 = 0;
    const int want_time = profile_level >= 2;
    /* Max axis for default range/step: 191 → 1141 words. */
    unsigned hot_bits[2048];
    int n_axis_h = 0;
    int n_words = 0;
    int use_hier = hier256 && hier_step > 0
        && (stage1_range % hier_step) == 0
        && (stage1_step > 0);

    if (use_hier) {
        int i;
        n_axis_h = es_hier256_n_axis(stage1_range, hier_step);
        n_words = (n_axis_h * n_axis_h + 31) >> 5;
        if (n_words > (int)(sizeof(hot_bits) / sizeof(hot_bits[0])))
            use_hier = 0;
        else {
            for (i = 0; i < n_words; i++) hot_bits[i] = 0u;
            for (dz = -stage1_range; dz <= stage1_range; dz += hier_step) {
                for (dx = -stage1_range; dx <= stage1_range; dx += hier_step) {
                    if (es_oct15_max0_gt(sn, dx, dz, celly, hier_thr)) {
                        int idx = es_hier256_bit_index(
                            dx, dz, stage1_range, hier_step, n_axis_h);
                        hot_bits[idx >> 5] |= 1u << (idx & 31);
                    }
                    if (prof) loc_pts++; /* count screen samples as pts */
                }
            }
        }
    }

    for (dz = -stage1_range; dz <= stage1_range; dz += stage1_step) {
        for (dx = -stage1_range; dx <= stage1_range; dx += stage1_step) {
            int bm = 0, bM = 0;
            double sum3 = 0.0;
            int fail_step = 0;
            int wx = dx, wz = dz;
            int time_this = 0;
            if (use_hier
                && !es_hier256_covers(
                       dx, dz, stage1_range, hier_step, hot_bits, n_axis_h))
                continue;
            if (prof) loc_pts++;
            if (want_time) {
                const int ix = (dx + stage1_range) / stage1_step;
                const int iz = (dz + stage1_range) / stage1_step;
                time_this = ((ix + iz * 3) & 15) == 0;
                if (time_this) loc_timed++;
            }
            if (prof || want_time) {
                if (!es_stage1_three_step_detail(
                        sn, &wx, &wz, celly, coarse_thr, neigh_radius, neigh_thr,
                        grad_allow1, grad_max_fail, &bm, &bM, &sum3, &fail_step,
                        &loc_c15, &loc_c14, &loc_c13, time_this)) {
                    if (prof) {
                        if (fail_step == 15) loc_f15++;
                        else if (fail_step == 16) loc_fneigh++;
                        else if (fail_step == 14) loc_f14++;
                        else if (fail_step == 13) loc_f13++;
                        else loc_fbr++;
                    }
                    continue;
                }
                if (prof) loc_pass++;
            } else {
                if (!es_stage1_three_step_ex(
                        sn, &wx, &wz, celly, coarse_thr, neigh_radius, neigh_thr,
                        grad_allow1, grad_max_fail, &bm, &bM, &sum3, NULL, NULL,
                        NULL, NULL, 0))
                    continue;
            }
            found = 1;
            if (bm && sum3 > bestMinSum3) {
                bestMinSum3 = sum3;
                bestMinX = wx;
                bestMinZ = wz;
            }
            if (bM && sum3 > bestMaxSum3) {
                bestMaxSum3 = sum3;
                bestMaxX = wx;
                bestMaxZ = wz;
            }
        }
    }
    if (prof) {
        atomicAdd(&prof->n_s1_pts, loc_pts);
        atomicAdd(&prof->n_s1_fail15, loc_f15);
        atomicAdd(&prof->n_s1_fail_neigh, loc_fneigh);
        atomicAdd(&prof->n_s1_fail14, loc_f14);
        atomicAdd(&prof->n_s1_fail13, loc_f13);
        atomicAdd(&prof->n_s1_fail_br, loc_fbr);
        atomicAdd(&prof->n_s1_pts_pass, loc_pass);
        if (want_time) {
            atomicAdd(&prof->n_s1_timed_pts, loc_timed);
            atomicAdd(&prof->s1_cycles[PROF_S1_OCT15], loc_c15);
            atomicAdd(&prof->s1_cycles[PROF_S1_OCT14], loc_c14);
            atomicAdd(&prof->s1_cycles[PROF_S1_OCT13], loc_c13);
        }
    }
    *branchMin = bestMinSum3 >= 0.0;
    *branchMax = bestMaxSum3 >= 0.0;
    if (*branchMin) {
        bestX[0] = bestMinX;
        bestZ[0] = bestMinZ;
        bestSum3[0] = bestMinSum3;
    }
    if (*branchMax) {
        bestX[1] = bestMaxX;
        bestZ[1] = bestMaxZ;
        bestSum3[1] = bestMaxSum3;
    }
    return found;
}

__device__ int es_scan_branch_max_d(
    const EsSurfaceNoise *sn, int centerX, int centerZ, int celly,
    int range, int step, int branch, double threshold,
    int *outX, int *outZ, double *outMax)
{
    int found = 0;
    int dz, dx;
    *outMax = -1e9;
    for (dz = -range; dz <= range; dz += step) {
        for (dx = -range; dx <= range; dx += step) {
            const int wx = centerX + dx;
            const int wz = centerZ + dz;
            double vmin, vmax;
            es_sample_envelope(sn, wx, wz, celly, &vmin, &vmax);
            const double val = es_branch_value(branch, vmin, vmax);
            if (val > *outMax) {
                *outMax = val;
                *outX = wx;
                *outZ = wz;
                found = 1;
            }
        }
    }
    if (!found || *outMax <= threshold) return 0;
    return 1;
}

__device__ void es_refine_peak_d(
    const EsSurfaceNoise *sn, int centerX, int centerZ, int celly, int branch,
    int *peakX, int *peakZ, double *peakVal)
{
    es_scan_branch_max_d(sn, centerX, centerZ, celly, 4, 1, branch, -1e9,
                         peakX, peakZ, peakVal);
}

__device__ int es_emit_period_hits_d(
    uint64_t seed, int peakX, int peakZ, int branch, double envelope,
    int stage1X, int stage1Z, const EsSurfaceNoise *sn, int celly,
    double period, int period_range, HitRow *out, int *out_count, int max_hits)
{
    const int kMax = (int)floor((double)period_range / period);
    int classOk[ES_MAIN_PHASE_MOD][ES_MAIN_PHASE_MOD];
    int anyClass = 0;
    int dx, dz, kx, kz;
    int rows = 0;

    for (dx = 0; dx < ES_MAIN_PHASE_MOD; dx++) {
        const int offX = (int)round((double)dx * period);
        for (dz = 0; dz < ES_MAIN_PHASE_MOD; dz++) {
            const int offZ = (int)round((double)dz * period);
            const double vmain =
                es_sample_vmain(sn, (peakX + offX) >> 3, celly, (peakZ + offZ) >> 3);
            classOk[dx][dz] = es_vmain_pass(branch, vmain);
            if (classOk[dx][dz]) anyClass = 1;
        }
    }
    if (!anyClass) return 0;

    for (kx = -kMax; kx <= kMax; kx++) {
        const int px = es_mod_phase(kx);
        const int offX = (int)round((double)kx * period);
        if (offX < -period_range || offX > period_range) continue;
        for (kz = -kMax; kz <= kMax; kz++) {
            const int pz = es_mod_phase(kz);
            int slot;
            double vmain;
            if (!classOk[px][pz]) continue;
            {
                const int offZ = (int)round((double)kz * period);
                const int tx = peakX + offX;
                const int tz = peakZ + offZ;
                if (offZ < -period_range || offZ > period_range) continue;
                if (es_skip_parity(tx, tz)) continue;
                vmain = es_sample_vmain(sn, tx >> 3, celly, tz >> 3);
                if (!es_vmain_pass(branch, vmain)) continue;

                slot = atomicAdd(out_count, 1);
                if (slot >= max_hits) {
                    atomicAdd(out_count, -1);
                    return rows;
                }
                out[slot].seed = (int64_t)seed;
                out[slot].branch = branch;
                out[slot].stage1_x = stage1X;
                out[slot].stage1_z = stage1Z;
                out[slot].envelope = envelope;
                out[slot].peak_x = peakX;
                out[slot].peak_z = peakZ;
                out[slot].tx = tx;
                out[slot].tz = tz;
                out[slot].vmain = vmain;
                rows++;
            }
        }
    }
    return rows;
}

__device__ void es_process_seed_d(
    uint64_t seed, const ScanConfig *cfg, HitRow *out, int *out_count,
    StageProf *prof)
{
    EsSurfaceNoise sn;
    int s1X[2] = {0, 0};
    int s1Z[2] = {0, 0};
    int branchMin = 0, branchMax = 0;
    double sum3[2] = {0.0, 0.0};
    int b;
    unsigned long long t0, t1;

    if (prof) atomicAdd(&prof->n_seed, 1ULL);

    if (cfg->phase_thr > -1e8f) {
        int pass;
        t0 = prof ? clock64() : 0ULL;
        pass = es_seed_phase_pass(seed, cfg->celly, cfg->phase_thr);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_PHASE, t1 - t0);
        }
        if (!pass) {
            if (prof) atomicAdd(&prof->n_phase_fail, 1ULL);
            return;
        }
        if (prof) atomicAdd(&prof->n_phase_pass, 1ULL);
    } else if (prof) {
        atomicAdd(&prof->n_phase_pass, 1ULL);
    }

    t0 = prof ? clock64() : 0ULL;
    es_init_surface_noise_end(&sn, seed);
    if (prof) {
        t1 = clock64();
        prof_add_cycles(prof, PROF_INIT, t1 - t0);
    }

    t0 = prof ? clock64() : 0ULL;
    {
        int ok = es_scan_stage1_d(
            &sn, cfg->celly, cfg->stage1_range, cfg->stage1_step,
            cfg->s1_coarse_thr, cfg->s1_neigh_radius, cfg->s1_neigh_thr,
            cfg->s1_hier256, cfg->s1_hier256_step, cfg->s1_hier256_thr,
            cfg->s1_gradvec ? es_grad_allow1_topk(cfg->s1_gradvec_topk) : NULL,
            cfg->s1_gradvec_max_fail,
            s1X, s1Z, &branchMin, &branchMax, sum3, prof, cfg->profile);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_STAGE1, t1 - t0);
        }
        if (!ok) {
            if (prof) atomicAdd(&prof->n_stage1_fail, 1ULL);
            return;
        }
    }
    if (prof) atomicAdd(&prof->n_stage1_pass, 1ULL);

    for (b = 0; b < 2; b++) {
        const int branch = (b == 0) ? ES_BRANCH_MIN : ES_BRANCH_MAX;
        const int flag = (b == 0) ? branchMin : branchMax;
        int s2X = 0, s2Z = 0;
        double s2Max = 0.0;
        int s3X = 0, s3Z = 0;
        double prePeriodMax = 0.0;
        int peakX, peakZ;
        if (!flag) continue;
        if (prof) atomicAdd(&prof->n_stage2_try, 1ULL);

        t0 = prof ? clock64() : 0ULL;
        {
            int ok = es_scan_branch_max_d(
                &sn, s1X[b], s1Z[b], cfg->celly, cfg->stage2_range,
                cfg->stage2_step, branch, ES_STAGE2_SUM_MIN, &s2X, &s2Z, &s2Max);
            if (prof) {
                t1 = clock64();
                prof_add_cycles(prof, PROF_STAGE2, t1 - t0);
            }
            if (!ok) continue;
        }
        if (prof) atomicAdd(&prof->n_stage2_pass, 1ULL);

        t0 = prof ? clock64() : 0ULL;
        {
            int ok = es_scan_branch_max_d(
                &sn, s2X, s2Z, cfg->celly, cfg->stage3_range, cfg->stage3_step,
                branch, ES_STAGE3_SUM_MIN, &s3X, &s3Z, &prePeriodMax);
            if (prof) {
                t1 = clock64();
                prof_add_cycles(prof, PROF_STAGE3, t1 - t0);
            }
            if (!ok) continue;
        }
        if (prof) atomicAdd(&prof->n_stage3_pass, 1ULL);

        peakX = s3X;
        peakZ = s3Z;
        t0 = prof ? clock64() : 0ULL;
        es_refine_peak_d(&sn, s3X, s3Z, cfg->celly, branch, &peakX, &peakZ,
                         &prePeriodMax);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_REFINE, t1 - t0);
            atomicAdd(&prof->n_refine, 1ULL);
        }

        t0 = prof ? clock64() : 0ULL;
        es_emit_period_hits_d(seed, peakX, peakZ, branch, prePeriodMax, s1X[b],
                              s1Z[b], &sn, cfg->celly, cfg->period,
                              cfg->period_range, out, out_count, cfg->max_hits);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_PERIOD, t1 - t0);
            atomicAdd(&prof->n_period_call, 1ULL);
        }
    }
}

__global__ void es_scan_seeds_kernel(
    uint64_t start_seed, int n, ScanConfig cfg, HitRow *out, int *out_count,
    StageProf *prof)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    es_process_seed_d(start_seed + (uint64_t)idx, &cfg, out, out_count,
                      cfg.profile ? prof : NULL);
}

/* =====================================================================
 * Grid-parallel stage1 (default): phase compact → one block / passing seed →
 * shared sn → grid-stride stage1 → thread0 stage2..period.
 * Disable: --no-s1-grid-par (seed-serial stage1; A/B rollback).
 * Evidence (50万种子, period_range=122566, n55): same 409 hits / 40 seeds,
 * wall ~42.3s serial → ~21.8s grid-par (~1.94×).
 * ===================================================================== */

__global__ void es_temp_phase_compact_kernel(
    uint64_t start_seed, int n, float phase_thr, int celly, uint64_t *out_seeds,
    int *out_n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t seed;
    int slot;
    if (idx >= n) return;
    seed = start_seed + (uint64_t)idx;
    if (phase_thr > -1e8f && !es_seed_phase_pass(seed, celly, phase_thr)) return;
    slot = atomicAdd(out_n, 1);
    out_seeds[slot] = seed;
}

__device__ void es_temp_tail_after_stage1_d(
    uint64_t seed, const EsSurfaceNoise *sn, const ScanConfig *cfg, int branchMin,
    int branchMax, const int *s1X, const int *s1Z, HitRow *out, int *out_count,
    StageProf *prof)
{
    int b;
    for (b = 0; b < 2; b++) {
        const int branch = (b == 0) ? ES_BRANCH_MIN : ES_BRANCH_MAX;
        const int flag = (b == 0) ? branchMin : branchMax;
        int s2X = 0, s2Z = 0;
        double s2Max = 0.0;
        int s3X = 0, s3Z = 0;
        double prePeriodMax = 0.0;
        int peakX, peakZ;
        unsigned long long t0, t1;
        if (!flag) continue;
        if (prof) atomicAdd(&prof->n_stage2_try, 1ULL);

        t0 = prof ? clock64() : 0ULL;
        if (!es_scan_branch_max_d(sn, s1X[b], s1Z[b], cfg->celly, cfg->stage2_range,
                                  cfg->stage2_step, branch, ES_STAGE2_SUM_MIN, &s2X,
                                  &s2Z, &s2Max)) {
            if (prof) {
                t1 = clock64();
                prof_add_cycles(prof, PROF_STAGE2, t1 - t0);
            }
            continue;
        }
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_STAGE2, t1 - t0);
            atomicAdd(&prof->n_stage2_pass, 1ULL);
        }

        t0 = prof ? clock64() : 0ULL;
        if (!es_scan_branch_max_d(sn, s2X, s2Z, cfg->celly, cfg->stage3_range,
                                  cfg->stage3_step, branch, ES_STAGE3_SUM_MIN, &s3X,
                                  &s3Z, &prePeriodMax)) {
            if (prof) {
                t1 = clock64();
                prof_add_cycles(prof, PROF_STAGE3, t1 - t0);
            }
            continue;
        }
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_STAGE3, t1 - t0);
            atomicAdd(&prof->n_stage3_pass, 1ULL);
        }

        peakX = s3X;
        peakZ = s3Z;
        t0 = prof ? clock64() : 0ULL;
        es_refine_peak_d(sn, s3X, s3Z, cfg->celly, branch, &peakX, &peakZ,
                         &prePeriodMax);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_REFINE, t1 - t0);
            atomicAdd(&prof->n_refine, 1ULL);
        }

        t0 = prof ? clock64() : 0ULL;
        es_emit_period_hits_d(seed, peakX, peakZ, branch, prePeriodMax, s1X[b],
                              s1Z[b], sn, cfg->celly, cfg->period, cfg->period_range,
                              out, out_count, cfg->max_hits);
        if (prof) {
            t1 = clock64();
            prof_add_cycles(prof, PROF_PERIOD, t1 - t0);
            atomicAdd(&prof->n_period_call, 1ULL);
        }
    }
}

/* One block = one phase-passing seed. Dynamic shared: EsSurfaceNoise
 * [+ optional hier256 hot bitset] + reduce scratch.
 * Profile cycles (when prof!=NULL): tid0 records block wall for init/stage1
 * (not sum-of-threads), then times stage2..period in the tail.
 */
__global__ void es_temp_grid_par_seed_kernel(
    const uint64_t *seeds, int n_seeds, ScanConfig cfg, HitRow *out, int *out_count,
    StageProf *prof)
{
    extern __shared__ unsigned char smem[];
    EsSurfaceNoise *sn = (EsSurfaceNoise *)smem;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int bid = (int)blockIdx.x;
    uint64_t seed;
    int n_axis, n_pts, p;
    double bestMinSum3 = -1.0, bestMaxSum3 = -1.0;
    int bestMinX = 0, bestMinZ = 0, bestMaxX = 0, bestMaxZ = 0;
    int branchMin = 0, branchMax = 0;
    int s1X[2], s1Z[2];
    __shared__ unsigned long long sh_t0;
    const size_t sn_bytes =
        (sizeof(EsSurfaceNoise) + (size_t)15) & ~(size_t)15;
    int use_hier = 0;
    int n_axis_h = 0, n_words = 0, n_hpts = 0;
    unsigned *hot = NULL;

    if (bid >= n_seeds) return;
    seed = seeds[bid];

    if (tid == 0 && prof) sh_t0 = clock64();
    __syncthreads();
    if (tid == 0) es_init_surface_noise_end(sn, seed);
    __syncthreads();
    if (tid == 0 && prof) {
        prof_add_cycles(prof, PROF_INIT, clock64() - sh_t0);
        sh_t0 = clock64();
    }
    __syncthreads();

    use_hier = cfg.s1_hier256 && cfg.s1_hier256_step > 0
        && (cfg.stage1_range % cfg.s1_hier256_step) == 0
        && (cfg.stage1_range % cfg.stage1_step) == 0;
    {
        const uint16_t *grad_allow1 =
            cfg.s1_gradvec ? es_grad_allow1_topk(cfg.s1_gradvec_topk) : NULL;
        const int grad_max_fail = cfg.s1_gradvec_max_fail;

    if (use_hier) {
        /* Shared layout: sn | hot_bits | hot_x[] | hot_z[] | n_hot | reduce */
        const int max_hot = 2048;
        int *n_hot_ptr;
        int *hot_x;
        int *hot_z;
        int n_hot;
        int half, n_side, per_hot, n_jobs;

        n_axis_h = es_hier256_n_axis(cfg.stage1_range, cfg.s1_hier256_step);
        n_hpts = n_axis_h * n_axis_h;
        n_words = (n_hpts + 31) >> 5;
        hot = (unsigned *)(smem + sn_bytes);
        hot_x = (int *)(smem + sn_bytes
            + (((size_t)n_words * sizeof(unsigned) + (size_t)15) & ~(size_t)15));
        hot_z = hot_x + max_hot;
        n_hot_ptr = hot_z + max_hot;

        for (p = tid; p < n_words; p += nthreads) hot[p] = 0u;
        if (tid == 0) *n_hot_ptr = 0;
        __syncthreads();

        for (p = tid; p < n_hpts; p += nthreads) {
            const int ix = p % n_axis_h;
            const int iz = p / n_axis_h;
            const int wx = -cfg.stage1_range + ix * cfg.s1_hier256_step;
            const int wz = -cfg.stage1_range + iz * cfg.s1_hier256_step;
            if (!es_oct15_max0_gt(sn, wx, wz, cfg.celly, cfg.s1_hier256_thr))
                continue;
            es_hier256_bit_set(hot, p);
            {
                int slot = atomicAdd(n_hot_ptr, 1);
                if (slot < max_hot) {
                    hot_x[slot] = wx;
                    hot_z[slot] = wz;
                }
            }
        }
        __syncthreads();
        n_hot = *n_hot_ptr;
        if (n_hot > max_hot) {
            /* Too many screen hits — fall back to full 128 lattice. */
            use_hier = 0;
        } else if (n_hot > 0) {
            int ring = cfg.s1_hier256_ring;
            if (ring < 1) ring = 1;
            if (ring > 2) ring = 2;
            if (cfg.s1_hier256_centers) {
                /* Most aggressive: only the 256-lattice hot centers. */
                for (p = tid; p < n_hot; p += nthreads) {
                    int wx = hot_x[p];
                    int wz = hot_z[p];
                    int bm = 0, bM = 0;
                    double sum3 = 0.0;
                    if (!es_stage1_three_step_ex(
                            sn, &wx, &wz, cfg.celly, cfg.s1_coarse_thr,
                            cfg.s1_neigh_radius, cfg.s1_neigh_thr, grad_allow1,
                            grad_max_fail, &bm, &bM, &sum3, NULL, NULL, NULL,
                            NULL, 0))
                        continue;
                    if (bm && sum3 > bestMinSum3) {
                        bestMinSum3 = sum3;
                        bestMinX = wx;
                        bestMinZ = wz;
                    }
                    if (bM && sum3 > bestMaxSum3) {
                        bestMaxSum3 = sum3;
                        bestMaxX = wx;
                        bestMaxZ = wz;
                    }
                }
            } else {
                /* ring=1 → ±hier_step (5×5); ring=2 → ±2*hier_step (9×9). */
                half = ring * cfg.s1_hier256_step;
                n_side = (2 * half) / cfg.stage1_step + 1;
                per_hot = n_side * n_side;
                n_jobs = n_hot * per_hot;
                for (p = tid; p < n_jobs; p += nthreads) {
                    const int hi = p / per_hot;
                    const int li = p % per_hot;
                    const int ox = (li % n_side) - (n_side / 2);
                    const int oz = (li / n_side) - (n_side / 2);
                    int wx = hot_x[hi] + ox * cfg.stage1_step;
                    int wz = hot_z[hi] + oz * cfg.stage1_step;
                    int bm = 0, bM = 0;
                    double sum3 = 0.0;
                    if (wx < -cfg.stage1_range || wx > cfg.stage1_range
                        || wz < -cfg.stage1_range || wz > cfg.stage1_range)
                        continue;
                    if (!es_stage1_three_step_ex(
                            sn, &wx, &wz, cfg.celly, cfg.s1_coarse_thr,
                            cfg.s1_neigh_radius, cfg.s1_neigh_thr, grad_allow1,
                            grad_max_fail, &bm, &bM, &sum3, NULL, NULL, NULL,
                            NULL, 0))
                        continue;
                    if (bm && sum3 > bestMinSum3) {
                        bestMinSum3 = sum3;
                        bestMinX = wx;
                        bestMinZ = wz;
                    }
                    if (bM && sum3 > bestMaxSum3) {
                        bestMaxSum3 = sum3;
                        bestMaxX = wx;
                        bestMaxZ = wz;
                    }
                }
            }
        }
    }

    if (!use_hier) {
        n_axis = (2 * (cfg.stage1_range / cfg.stage1_step)) + 1;
        n_pts = n_axis * n_axis;

        for (p = tid; p < n_pts; p += nthreads) {
            const int ix = p % n_axis;
            const int iz = p / n_axis;
            int wx = -cfg.stage1_range + ix * cfg.stage1_step;
            int wz = -cfg.stage1_range + iz * cfg.stage1_step;
            int bm = 0, bM = 0;
            double sum3 = 0.0;
            if (!es_stage1_three_step_ex(
                    sn, &wx, &wz, cfg.celly, cfg.s1_coarse_thr,
                    cfg.s1_neigh_radius, cfg.s1_neigh_thr, grad_allow1,
                    grad_max_fail, &bm, &bM, &sum3, NULL, NULL, NULL, NULL, 0))
                continue;
            if (bm && sum3 > bestMinSum3) {
                bestMinSum3 = sum3;
                bestMinX = wx;
                bestMinZ = wz;
            }
            if (bM && sum3 > bestMaxSum3) {
                bestMaxSum3 = sum3;
                bestMaxX = wx;
                bestMaxZ = wz;
            }
        }
    }
    } /* grad_allow1 scope */

    /* Block reduce: pack thread locals into shared after sn (+hier scratch). */
    {
        unsigned char *base = smem + sn_bytes;
        double *red_min;
        double *red_max;
        int *red_min_x, *red_min_z, *red_max_x, *red_max_z;
        int i;
        if (cfg.s1_hier256 && cfg.s1_hier256_step > 0
            && (cfg.stage1_range % cfg.s1_hier256_step) == 0) {
            const int max_hot = 2048;
            const int n_axis_h2 =
                es_hier256_n_axis(cfg.stage1_range, cfg.s1_hier256_step);
            const int n_words2 = (n_axis_h2 * n_axis_h2 + 31) >> 5;
            size_t hier_bytes =
                ((size_t)n_words2 * sizeof(unsigned) + (size_t)15) & ~(size_t)15;
            hier_bytes += (size_t)max_hot * 2 * sizeof(int) + sizeof(int);
            hier_bytes = (hier_bytes + (size_t)15) & ~(size_t)15;
            base += hier_bytes;
        }
        red_min = (double *)base;
        red_max = red_min + nthreads;
        red_min_x = (int *)(red_max + nthreads);
        red_min_z = red_min_x + nthreads;
        red_max_x = red_min_z + nthreads;
        red_max_z = red_max_x + nthreads;

        red_min[tid] = bestMinSum3;
        red_max[tid] = bestMaxSum3;
        red_min_x[tid] = bestMinX;
        red_min_z[tid] = bestMinZ;
        red_max_x[tid] = bestMaxX;
        red_max_z[tid] = bestMaxZ;
        __syncthreads();

        if (tid == 0) {
            bestMinSum3 = -1.0;
            bestMaxSum3 = -1.0;
            for (i = 0; i < nthreads; i++) {
                if (red_min[i] > bestMinSum3) {
                    bestMinSum3 = red_min[i];
                    bestMinX = red_min_x[i];
                    bestMinZ = red_min_z[i];
                }
                if (red_max[i] > bestMaxSum3) {
                    bestMaxSum3 = red_max[i];
                    bestMaxX = red_max_x[i];
                    bestMaxZ = red_max_z[i];
                }
            }
            branchMin = bestMinSum3 >= 0.0;
            branchMax = bestMaxSum3 >= 0.0;
            s1X[0] = bestMinX;
            s1Z[0] = bestMinZ;
            s1X[1] = bestMaxX;
            s1Z[1] = bestMaxZ;
            if (prof) {
                prof_add_cycles(prof, PROF_STAGE1, clock64() - sh_t0);
                if (branchMin || branchMax)
                    atomicAdd(&prof->n_stage1_pass, 1ULL);
                else
                    atomicAdd(&prof->n_stage1_fail, 1ULL);
            }
            if (branchMin || branchMax) {
                es_temp_tail_after_stage1_d(seed, sn, &cfg, branchMin, branchMax, s1X,
                                            s1Z, out, out_count, prof);
            }
        }
    }
}

static void es_print_stage_prof(const StageProf *h_prof, float wall_ms)
{
    static const char *names[PROF_N] = {
        "phase", "init", "stage1", "stage2", "stage3", "refine", "period"};
    unsigned long long total = 0;
    int s;
    for (s = 0; s < PROF_N; s++) total += h_prof->cycles[s];
    fprintf(stderr, "\n=== profile (thread/block-cycles; wall=%.3f ms) ===\n",
            (double)wall_ms);
    fprintf(stderr,
            "funnel: seed=%llu phase_fail=%llu phase_pass=%llu "
            "s1_fail=%llu s1_pass=%llu s2_try=%llu s2_pass=%llu "
            "s3_pass=%llu refine=%llu period=%llu\n",
            (unsigned long long)h_prof->n_seed,
            (unsigned long long)h_prof->n_phase_fail,
            (unsigned long long)h_prof->n_phase_pass,
            (unsigned long long)h_prof->n_stage1_fail,
            (unsigned long long)h_prof->n_stage1_pass,
            (unsigned long long)h_prof->n_stage2_try,
            (unsigned long long)h_prof->n_stage2_pass,
            (unsigned long long)h_prof->n_stage3_pass,
            (unsigned long long)h_prof->n_refine,
            (unsigned long long)h_prof->n_period_call);
    for (s = 0; s < PROF_N; s++) {
        double pct =
            total ? (100.0 * (double)h_prof->cycles[s] / (double)total) : 0.0;
        fprintf(stderr, "  %-7s cycles=%llu  share=%.2f%%\n", names[s],
                (unsigned long long)h_prof->cycles[s], pct);
    }
    fprintf(stderr, "  TOTAL   cycles=%llu\n", (unsigned long long)total);
    if (h_prof->n_s1_pts) {
        const double np = (double)h_prof->n_s1_pts;
        fprintf(stderr,
                "stage1 pts: total=%llu fail15=%.2f%% fail_neigh=%.2f%% "
                "fail14=%.2f%% fail13=%.2f%% fail_br=%.2f%% pass=%.2f%%\n",
                (unsigned long long)h_prof->n_s1_pts,
                100.0 * (double)h_prof->n_s1_fail15 / np,
                100.0 * (double)h_prof->n_s1_fail_neigh / np,
                100.0 * (double)h_prof->n_s1_fail14 / np,
                100.0 * (double)h_prof->n_s1_fail13 / np,
                100.0 * (double)h_prof->n_s1_fail_br / np,
                100.0 * (double)h_prof->n_s1_pts_pass / np);
    }
    fprintf(stderr, "\n");
}

static int es_temp_run_grid_par(
    uint64_t start_seed, int nseeds, ScanConfig cfg, HitRow *d_hits, int *d_count,
    float *wall_ms_out, int do_profile, StageProf *d_prof)
{
    uint64_t *d_pass = NULL;
    int *d_npass = NULL;
    int h_npass = 0;
    int threads = 128;
    int blocks;
    cudaEvent_t ev0, ev1, ev_phase1, ev_kern0;
    float ms_phase = 0.0f, ms_kern = 0.0f;
    size_t shmem;

    if (cfg.s1_grid_par_threads >= 32 && cfg.s1_grid_par_threads <= 1024)
        threads = cfg.s1_grid_par_threads;

    CUDA_CHECK(cudaMalloc((void **)&d_pass, (size_t)nseeds * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_npass, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_npass, 0, sizeof(int)));

    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev_phase1));
    CUDA_CHECK(cudaEventCreate(&ev_kern0));
    CUDA_CHECK(cudaEventRecord(ev0, 0));

    blocks = (nseeds + 127) / 128;
    es_temp_phase_compact_kernel<<<blocks, 128>>>(
        start_seed, nseeds, cfg.phase_thr, cfg.celly, d_pass, d_npass);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_phase1, 0));
    CUDA_CHECK(cudaMemcpy(&h_npass, d_npass, sizeof(int), cudaMemcpyDeviceToHost));

    fprintf(stderr, "s1-grid-par: phase_pass=%d / %d, block_threads=%d\n",
            h_npass, nseeds, threads);

    CUDA_CHECK(cudaEventRecord(ev_kern0, 0));
    if (h_npass > 0) {
        const size_t sn_bytes =
            (sizeof(EsSurfaceNoise) + (size_t)15) & ~(size_t)15;
        size_t hot_bytes = 0;
        if (cfg.s1_hier256 && cfg.s1_hier256_step > 0
            && (cfg.stage1_range % cfg.s1_hier256_step) == 0) {
            const int max_hot = 2048;
            const int n_axis_h =
                (2 * (cfg.stage1_range / cfg.s1_hier256_step)) + 1;
            const int n_words = (n_axis_h * n_axis_h + 31) >> 5;
            hot_bytes =
                ((size_t)n_words * sizeof(unsigned) + (size_t)15) & ~(size_t)15;
            hot_bytes += (size_t)max_hot * 2 * sizeof(int) + sizeof(int);
            hot_bytes = (hot_bytes + (size_t)15) & ~(size_t)15;
        }
        shmem = sn_bytes + hot_bytes
            + (size_t)threads * (2 * sizeof(double) + 4 * sizeof(int));
        if (shmem > 48 * 1024) {
            fprintf(stderr, "s1-grid-par: shmem %zu too large\n", shmem);
            cudaFree(d_pass);
            cudaFree(d_npass);
            return 1;
        }
        es_temp_grid_par_seed_kernel<<<h_npass, threads, shmem>>>(
            d_pass, h_npass, cfg, d_hits, d_count, do_profile ? d_prof : NULL);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev1, 0));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(wall_ms_out, ev0, ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_phase, ev0, ev_phase1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kern, ev_kern0, ev1));
    fprintf(stderr,
            "s1-grid-par wall: phase_compact=%.3f ms  seed_blocks=%.3f ms  "
            "total=%.3f ms\n",
            (double)ms_phase, (double)ms_kern, (double)*wall_ms_out);

    if (do_profile && d_prof) {
        StageProf h_prof;
        CUDA_CHECK(cudaMemcpy(&h_prof, d_prof, sizeof(StageProf),
                              cudaMemcpyDeviceToHost));
        /* Host-side phase accounting (compact kernel has no cycle sinks). */
        h_prof.n_seed = (unsigned long long)nseeds;
        h_prof.n_phase_pass = (unsigned long long)h_npass;
        h_prof.n_phase_fail = (unsigned long long)(nseeds - h_npass);
        es_print_stage_prof(&h_prof, *wall_ms_out);
        fprintf(stderr,
                "note: grid-par profile — phase has no device cycles (host wall "
                "above); init/stage1 are per-block wall via tid0 clock64; "
                "stage2..period are tid0 only.\n\n");
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev_phase1));
    CUDA_CHECK(cudaEventDestroy(ev_kern0));
    CUDA_CHECK(cudaFree(d_pass));
    CUDA_CHECK(cudaFree(d_npass));
    return 0;
}

/* ======================== end s1-grid-par ======================== */

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --start-seed A --end-seed B --out hits.csv\n"
            "  [--stage1-step 128] [--stage1-range 24320]\n"
            "  [--s1-coarse-thr 55] [--s1-neigh|--no-s1-neigh]\n"
            "  [--s1-neigh-thr 60] [--s1-neigh-radius 64]\n"
            "  [--s1-hier256|--no-s1-hier256] [--s1-hier256-thr 42]\n"
            "  [--s1-hier256-ring 1|2] [--s1-hier256-centers]\n"
            "  [--s1-gradvec|--no-s1-gradvec] [--s1-gradvec-topk 4..8]\n"
            "  [--s1-gradvec-max-fail 0|1|2]\n"
            "  [--period-range %d] [--no-phase] [--phase-thr F]\n"
            "  [--celly 18] [--max-hits N] [--profile] [--profile-s1]\n"
            "  [--s1-grid-par|--no-s1-grid-par] [--s1-grid-par-threads 256]\n"
            "End SurfaceNoise period scan @ Y=%d (celly=%d).\n"
            "Prefilter: true y_offset ef 32-bin LUT OR4 (default thr=known 27/27).\n"
            "Default stage1: n55 (±64 9-pt) + grid-par + gradvec allow1@topk4.\n"
            "  Disable neigh: --no-s1-neigh. Optional hier: --s1-hier256.\n"
            "  Gradvec: default on (allow1 topk=4). Disable: --no-s1-gradvec.\n"
            "    [--s1-gradvec-topk 4..8] [--s1-gradvec-max-fail 0|1|2].\n"
            "  Disable grid-par: --no-s1-grid-par.\n"
            "--profile: host wall (phase vs seed_blocks) + stage cycle shares.\n"
            "TEMP --profile-s1: serial path only; subsampled oct15/14/13 clocks.\n"
            "Fidelity: cubiomes DIM_END SurfaceNoise + C++learning period gates.\n",
            argv0, ES_PERIOD_RANGE, ES_WORLD_Y, ES_CELLY);
}

static int parse_i64(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v;
    if (!s || !*s) return 0;
    v = strtoll(s, &end, 10);
    if (!end || *end) return 0;
    *out = (int64_t)v;
    return 1;
}

static int parse_i32(const char *s, int *out)
{
    int64_t v;
    if (!parse_i64(s, &v)) return 0;
    if (v < -2147483647LL - 1 || v > 2147483647LL) return 0;
    *out = (int)v;
    return 1;
}

static int parse_f64(const char *s, double *out)
{
    char *end = NULL;
    double v;
    if (!s || !*s) return 0;
    v = strtod(s, &end);
    if (!end || *end) return 0;
    *out = v;
    return 1;
}

int main(int argc, char **argv)
{
    int64_t start_seed = 0, end_seed = -1;
    const char *out_path = NULL;
    ScanConfig cfg;
    int argi;
    uint64_t nseeds;
    HitRow *h_hits = NULL;
    HitRow *d_hits = NULL;
    int *d_count = NULL;
    int h_count = 0;
    int threads = 128;
    int blocks;
    FILE *out;
    int i;
    int do_profile = 0;
    StageProf *d_prof = NULL;
    StageProf h_prof;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float wall_ms = 0.0f;

    memset(&cfg, 0, sizeof(cfg));
    cfg.celly = ES_CELLY;
    cfg.stage1_range = ES_STAGE1_RANGE;
    cfg.stage1_step = ES_STAGE1_STEP;
    cfg.stage2_range = ES_STAGE2_RANGE;
    cfg.stage2_step = ES_STAGE2_STEP;
    cfg.stage3_range = ES_STAGE3_RANGE;
    cfg.stage3_step = ES_STAGE3_STEP;
    cfg.period = ES_PERIOD;
    cfg.period_range = ES_PERIOD_RANGE;
    cfg.phase_thr = end_phase_lut::kThreshold;
    cfg.max_hits = 2000000;
    cfg.profile = 0;
    cfg.s1_coarse_thr = ES_STEP1_MAX0;
    cfg.s1_neigh_radius = ES_S1_NEIGH_RADIUS; /* default n55 */
    cfg.s1_neigh_thr = ES_S1_NEIGH_THR;
    cfg.s1_grid_par = 1; /* default: block-per-seed grid stage1 */
    cfg.s1_grid_par_threads = 256;
    cfg.s1_hier256 = 0; /* off: GPU mark-driven still slower than full128 at thr42 */
    cfg.s1_hier256_step = ES_S1_HIER256_STEP;
    cfg.s1_hier256_thr = ES_S1_HIER256_THR;
    cfg.s1_hier256_ring = 2;
    cfg.s1_hier256_centers = 0;
    cfg.s1_gradvec = 1; /* TEMP default: allow1 topk=4 (speed×140+ yield); --no-s1-gradvec to disable */
    cfg.s1_gradvec_topk = 4;
    cfg.s1_gradvec_max_fail = ES_S1_GRADVEC_MAX_FAIL; /* allow1 */

    for (argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "--start-seed") == 0 && argi + 1 < argc) {
            if (!parse_i64(argv[++argi], &start_seed)) {
                fprintf(stderr, "bad --start-seed\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--end-seed") == 0 && argi + 1 < argc) {
            if (!parse_i64(argv[++argi], &end_seed)) {
                fprintf(stderr, "bad --end-seed\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--out") == 0 && argi + 1 < argc) {
            out_path = argv[++argi];
        } else if (strcmp(argv[argi], "--stage1-step") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.stage1_step) || cfg.stage1_step < 1) {
                fprintf(stderr, "bad --stage1-step\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--stage1-range") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.stage1_range) || cfg.stage1_range < 0) {
                fprintf(stderr, "bad --stage1-range\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-coarse-thr") == 0 && argi + 1 < argc) {
            if (!parse_f64(argv[++argi], &cfg.s1_coarse_thr)) {
                fprintf(stderr, "bad --s1-coarse-thr\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-neigh") == 0) {
            cfg.s1_neigh_radius = ES_S1_NEIGH_RADIUS;
        } else if (strcmp(argv[argi], "--no-s1-neigh") == 0) {
            cfg.s1_neigh_radius = 0;
        } else if (strcmp(argv[argi], "--s1-neigh-thr") == 0 && argi + 1 < argc) {
            if (!parse_f64(argv[++argi], &cfg.s1_neigh_thr)) {
                fprintf(stderr, "bad --s1-neigh-thr\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-neigh-radius") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_neigh_radius)
                || cfg.s1_neigh_radius < 0) {
                fprintf(stderr, "bad --s1-neigh-radius\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--period-range") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.period_range) || cfg.period_range < 0) {
                fprintf(stderr, "bad --period-range\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--celly") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.celly)) {
                fprintf(stderr, "bad --celly\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--max-hits") == 0 && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.max_hits) || cfg.max_hits < 1) {
                fprintf(stderr, "bad --max-hits\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--phase-thr") == 0 && argi + 1 < argc) {
            char *end = NULL;
            cfg.phase_thr = strtof(argv[++argi], &end);
            if (!end || *end) {
                fprintf(stderr, "bad --phase-thr\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--no-phase") == 0
                   || strcmp(argv[argi], "--no-yoffset") == 0) {
            cfg.phase_thr = -1e9f;
        } else if (strcmp(argv[argi], "--profile") == 0) {
            do_profile = 1;
            if (cfg.profile < 1) cfg.profile = 1;
        } else if (strcmp(argv[argi], "--profile-s1") == 0) {
            do_profile = 1;
            cfg.profile = 2; /* subsampled clock64 per oct15/14/13 */
        } else if (strcmp(argv[argi], "--s1-grid-par") == 0) {
            cfg.s1_grid_par = 1;
        } else if (strcmp(argv[argi], "--no-s1-grid-par") == 0) {
            cfg.s1_grid_par = 0;
        } else if (strcmp(argv[argi], "--s1-hier256") == 0) {
            cfg.s1_hier256 = 1;
        } else if (strcmp(argv[argi], "--no-s1-hier256") == 0) {
            cfg.s1_hier256 = 0;
        } else if (strcmp(argv[argi], "--s1-hier256-thr") == 0
                   && argi + 1 < argc) {
            if (!parse_f64(argv[++argi], &cfg.s1_hier256_thr)) {
                fprintf(stderr, "bad --s1-hier256-thr\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-hier256-step") == 0
                   && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_hier256_step)
                || cfg.s1_hier256_step < 1) {
                fprintf(stderr, "bad --s1-hier256-step\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-hier256-ring") == 0
                   && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_hier256_ring)
                || cfg.s1_hier256_ring < 1 || cfg.s1_hier256_ring > 2) {
                fprintf(stderr, "bad --s1-hier256-ring (want 1 or 2)\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-hier256-centers") == 0) {
            cfg.s1_hier256_centers = 1;
        } else if (strcmp(argv[argi], "--s1-gradvec") == 0) {
            cfg.s1_gradvec = 1;
        } else if (strcmp(argv[argi], "--no-s1-gradvec") == 0) {
            cfg.s1_gradvec = 0;
        } else if (strcmp(argv[argi], "--s1-gradvec-topk") == 0
                   && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_gradvec_topk)
                || cfg.s1_gradvec_topk < 4 || cfg.s1_gradvec_topk > 8) {
                fprintf(stderr, "bad --s1-gradvec-topk (want 4..8)\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-gradvec-max-fail") == 0
                   && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_gradvec_max_fail)
                || cfg.s1_gradvec_max_fail < 0
                || cfg.s1_gradvec_max_fail > 2) {
                fprintf(stderr, "bad --s1-gradvec-max-fail (want 0..2)\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--s1-grid-par-threads") == 0
                   && argi + 1 < argc) {
            if (!parse_i32(argv[++argi], &cfg.s1_grid_par_threads)
                || cfg.s1_grid_par_threads < 32
                || cfg.s1_grid_par_threads > 1024) {
                fprintf(stderr, "bad --s1-grid-par-threads\n");
                return 1;
            }
        } else if (strcmp(argv[argi], "--help") == 0 || strcmp(argv[argi], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[argi]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!out_path || end_seed < start_seed) {
        usage(argv[0]);
        return 1;
    }

    nseeds = (uint64_t)(end_seed - start_seed) + 1ULL;
    if (nseeds > (uint64_t)INT32_MAX) {
        fprintf(stderr, "seed range too large for one launch\n");
        return 1;
    }

    fprintf(stderr,
            "end_surface_period_scan seeds=%" PRIu64 " celly=%d stage1_step=%d "
            "s1_coarse=%.1f s1_neigh_r=%d s1_neigh_thr=%.1f s1_grid_par=%d "
            "s1_hier256=%d hier_thr=%.1f hier_ring=%d hier_centers=%d "
            "s1_gradvec=%d grad_topk=%d grad_max_fail=%d "
            "period_range=%d phase_thr=%.6f max_hits=%d\n",
            nseeds, cfg.celly, cfg.stage1_step, cfg.s1_coarse_thr,
            cfg.s1_neigh_radius, cfg.s1_neigh_thr, cfg.s1_grid_par,
            cfg.s1_hier256, cfg.s1_hier256_thr, cfg.s1_hier256_ring,
            cfg.s1_hier256_centers, cfg.s1_gradvec, cfg.s1_gradvec_topk,
            cfg.s1_gradvec_max_fail, cfg.period_range, (double)cfg.phase_thr,
            cfg.max_hits);

    h_hits = (HitRow *)malloc((size_t)cfg.max_hits * sizeof(HitRow));
    if (!h_hits) {
        fprintf(stderr, "oom hits\n");
        return 1;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_hits, (size_t)cfg.max_hits * sizeof(HitRow)));
    CUDA_CHECK(cudaMalloc((void **)&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    if (do_profile) {
        CUDA_CHECK(cudaMalloc((void **)&d_prof, sizeof(StageProf)));
        CUDA_CHECK(cudaMemset(d_prof, 0, sizeof(StageProf)));
    }

    if (cfg.s1_grid_par) {
        if (es_temp_run_grid_par((uint64_t)start_seed, (int)nseeds, cfg, d_hits,
                                 d_count, &wall_ms, do_profile, d_prof) != 0)
            return 1;
        if (d_prof) CUDA_CHECK(cudaFree(d_prof));
    } else {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));

        blocks = ((int)nseeds + threads - 1) / threads;
        es_scan_seeds_kernel<<<blocks, threads>>>(
            (uint64_t)start_seed, (int)nseeds, cfg, d_hits, d_count, d_prof);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&wall_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        if (do_profile) {
            CUDA_CHECK(cudaMemcpy(&h_prof, d_prof, sizeof(StageProf),
                                  cudaMemcpyDeviceToHost));
            es_print_stage_prof(&h_prof, wall_ms);
            if (cfg.profile >= 2 && h_prof.n_s1_timed_pts) {
                unsigned long long s1t = h_prof.s1_cycles[0] + h_prof.s1_cycles[1]
                    + h_prof.s1_cycles[2];
                static const char *s1n[3] = {"oct15", "oct14", "oct13"};
                int s;
                fprintf(stderr,
                        "stage1 step clocks (subsampled pts=%llu):\n",
                        (unsigned long long)h_prof.n_s1_timed_pts);
                for (s = 0; s < 3; s++) {
                    double pct =
                        s1t ? (100.0 * (double)h_prof.s1_cycles[s] / (double)s1t)
                            : 0.0;
                    fprintf(stderr, "  %-7s cycles=%llu  share_of_timed=%.2f%%\n",
                            s1n[s], (unsigned long long)h_prof.s1_cycles[s], pct);
                }
                fprintf(stderr, "\n");
            }
            CUDA_CHECK(cudaFree(d_prof));
        }
    }

    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count > cfg.max_hits) h_count = cfg.max_hits;
    if (h_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_hits, d_hits, (size_t)h_count * sizeof(HitRow),
                              cudaMemcpyDeviceToHost));
    }

    out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "failed to open %s\n", out_path);
        return 1;
    }
    fprintf(out, "seed,branch,stage1_x,stage1_z,envelope,peak_x,peak_z,tx,tz,vmain\n");
    for (i = 0; i < h_count; i++) {
        const HitRow *h = &h_hits[i];
        fprintf(out,
                "%" PRId64 ",%s,%d,%d,%.6f,%d,%d,%d,%d,%.6f\n",
                h->seed, h->branch == ES_BRANCH_MIN ? "min" : "max", h->stage1_x,
                h->stage1_z, h->envelope, h->peak_x, h->peak_z, h->tx, h->tz,
                h->vmain);
    }
    fclose(out);

    fprintf(stderr, "wrote %d hits -> %s  wall=%.3f ms\n", h_count, out_path,
            (double)wall_ms);

    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_count));
    free(h_hits);
    return 0;
}
