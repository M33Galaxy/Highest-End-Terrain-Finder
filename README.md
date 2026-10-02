# 末地 SurfaceNoise 周期高点扫描器（CUDA）

用末地 `SurfaceNoise` 的 **main 噪声周期** `P ≈ 49026.656` 格，在给定种子区间里找出
**末地最高地形**候选点：先在 Y=72（`celly=18`）平面上筛出落在 `SurfaceNoise` 包络高位的
点（min/max 两个 branch），再把该点按周期晶格平移 `k·P`（`|k·P| ≤ period_range`），
用 `vmain` 逐点复核，输出所有仍然成立的 `(seed, tx, tz)`。

程序是单文件 CUDA 可执行文件，**只依赖** 同目录两个头文件，无第三方库、无 cubiomes。

```text
end_surface_period_scan.cu   主程序（kernel + CLI + CSV 输出）
end_surface_noise.cuh        末地噪声原语：Java LCG、Perlin、oct13/14/15、vmain、y_offset 预筛
end_phase_lut.cuh            y_offset 相位 32-bin LUT 权重（自动生成，勿手改）
```

## 构建

需要 CUDA Toolkit（`nvcc`），无需其它依赖：

```bash
# Linux / Colab / AutoDL
nvcc -O3 -std=c++17 -arch=sm_89 -o end_surface_period_scan end_surface_period_scan.cu

# Windows
nvcc -O3 -std=c++17 -arch=native -o end_surface_period_scan.exe end_surface_period_scan.cu
```

换架构改 `-arch` 即可（例如 `sm_75` / `sm_86` / `sm_89`）。程序在 host 侧检查
动态 shared memory 是否超过 48 KB，超了会直接报错退出。

## 用法

```bash
./end_surface_period_scan --start-seed 0 --end-seed 999999 --out hits.csv
```

`--start-seed` / `--end-seed` / `--out` 必填，`--end-seed ≥ --start-seed`；
单次 launch 的种子数不能超过 `INT32_MAX`。

```text
--start-seed A        起始种子（默认 0）
--end-seed B          结束种子（闭区间，必填）
--out FILE            输出 CSV（必填）
--help, -h            打印内置 usage
```

### 参数与默认值

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--stage1-step` | `128` | stage1 网格步长（格） |
| `--stage1-range` | `24320` | stage1 网格半径（±格） |
| `--s1-coarse-thr` | `55` | oct15 中心 `max(min,max)` 粗筛阈值 |
| `--s1-neigh` / `--no-s1-neigh` | 开 | ±64 的 9 点邻域 oct15 确认 |
| `--s1-neigh-radius` | `64` | 邻域半径 |
| `--s1-neigh-thr` | `60` | 邻域确认阈值 |
| `--s1-hier256` / `--no-s1-hier256` | 关 | step-256 oct15 屏幕 → 命中附近细化 |
| `--s1-hier256-step` | `256` | 屏幕步长 |
| `--s1-hier256-thr` | `42` | 屏幕阈值（host 侧实测 ~9% FN） |
| `--s1-hier256-ring` | `2` | 细化半宽 = ring × hier_step（1→5×5，2→9×9） |
| `--s1-hier256-centers` | 关 | 只在 256 晶格热点中心做 `three_step` |
| `--s1-gradvec` / `--no-s1-gradvec` | 开 | oct15 梯度位（min OR max）预筛 |
| `--s1-gradvec-topk` | `4` | 4..8，选用 `ES_GRAD_ALLOW1_K*` 掩码 |
| `--s1-gradvec-max-fail` | `1` | `0`=allow0，`1`=allow1，`2`=soft |
| `--s1-grid-par` / `--no-s1-grid-par` | 开 | 见「两种 stage1 执行方式」 |
| `--s1-grid-par-threads` | `256` | 32..1024，每种子 block 的线程数 |
| `--period-range` | `30000000` | 周期平移搜索半径（±格） |
| `--celly` | `18` | 采样 cell 的 Y（Y=72 → 18） |
| `--max-hits` | `2000000` | 命中上限（同时决定 host/device 缓冲大小） |
| `--no-phase` / `--no-yoffset` | 关 | 关闭 y_offset 相位预筛（等价 `--phase-thr -1e9`） |
| `--phase-thr` | `0.97045886` | 相位 LUT 门控阈值 |
| `--profile` | 关 | 打印 wall（phase / seed_blocks）与各阶段 cycle 占比 |
| `--profile-s1` | 关 | 仅串行路径：再打印 oct15/14/13 的抽样子步耗时 |

### 示例

```bash
# 默认全速扫描
./end_surface_period_scan --start-seed 0 --end-seed 4999999 --out hits.csv

# 关掉预筛与加速（最保守，用于对照 / A-B 回滚）
./end_surface_period_scan --start-seed 0 --end-seed 99999 --out ref.csv \
    --no-phase --no-s1-gradvec --no-s1-grid-par --no-s1-neigh

# 打开 256 分层屏幕（噪声大、结构简单时可省样本）
./end_surface_period_scan --start-seed 0 --end-seed 9999999 --out h.csv --s1-hier256

# 性能画像
./end_surface_period_scan --start-seed 0 --end-seed 999999 --out hits.csv --profile
```

## 扫描流程

每个种子依次经过（任一阶段失败即丢弃）：

1. **y_offset 相位预筛**（可用 `--no-phase` 关闭）
   用 Java LCG 推出 oct14/15 的 min/max `y_offset`，换算真实相位
   `ef = frac(celly·684.412·persist + b)`，落进 32 个 bin，查 LUT 取
   `max(score_min, score_max)`；低于阈值即淘汰。默认阈值对应已知周期的
   27/27 全召回（保留约 35%）。
2. **stage1（`±stage1_range`，step-128 网格）**
   - `es_oct15_gradvec_or`：梯度位掩码预筛（默认 allow1 / topk=4）；
   - oct15 `max(min,max) > 55`；
   - ±64 的 9 点邻域 oct15 确认 `> 60`，并把命中点**迁移**到邻域最优点；
   - 累加 oct14，`sum > 100`；
   - 累加 oct13，`sum > 110`，得到 min / max 两个 branch（`branchMin` / `branchMax`）；
   - 同网格内对每个 branch 保留 `sum3` 最大的点。
3. **stage2**：以 stage1 点为心，`±48` / step-16 找 `branch_value` 最大点，要求 `> 130`。
4. **stage3**：`±16` / step-2，要求 `> 137`。
5. **peak 精修**：`±4` / step-1 提纯峰值坐标与包络值。
6. **周期平移 + vmain 复核**：以 peak 为基准，对
   `kx, kz ∈ [-kMax, kMax]`（`kMax = floor(period_range / P)`）取整数偏移
   `round(k·P)`，依次通过
   `mod 5` 相位类预筛 → 抛物线式 parity 跳过（`(x²+z²) >> 37` 的奇偶）→
   `vmain` 复核（min branch 要求 `vmain < 0`，max branch 要求 `vmain > 1`），
   全部通过才写一行结果。

`branch_value`：min branch 取包络 `vmin`，max branch 取 `vmax`。

### 两种 stage1 执行方式

- `--s1-grid-par`（默认）：先用 `es_temp_phase_compact_kernel` 把所有通过相位预筛的种子
  压紧成数组，再 **一个 block 一个种子**：block 内共享一份 `EsSurfaceNoise`，
  在 step-128 网格上 grid-stride 并行，block 归约出最优 min/max 点，最后仅
  `tid==0` 跑 stage2..period。源码注释里的实测约 **1.94×** 加速
  （50 万种子 / `period_range=122566` / n55：42.3 s → 21.8 s）。
- `--no-s1-grid-par`：一个线程一个种子，从头串到尾（`es_scan_seeds_kernel`），
  作为正确性对照与回滚路径。

## 输出

`--out` 写 CSV，首行为表头：

```text
seed,branch,stage1_x,stage1_z,envelope,peak_x,peak_z,tx,tz,vmain
```

| 列 | 含义 |
| --- | --- |
| `seed` | 世界种子 |
| `branch` | `min` 或 `max`（对应 `ES_BRANCH_MIN` / `ES_BRANCH_MAX`） |
| `stage1_x`, `stage1_z` | stage1 命中点（邻域确认后可能已迁移） |
| `envelope` | peak 精修后的 branch 包络值 |
| `peak_x`, `peak_z` | 精修后的峰值坐标 |
| `tx`, `tz` | 按周期平移并复核通过的候选坐标 |
| `vmain` | 该坐标的 `vmain` 采样值 |

运行时 stderr 会打印生效参数、命中数、wall time；`--profile` 另打印种子漏斗
（seed / phase_fail / s1_fail / s2_pass / s3_pass …）、各阶段 cycle 占比，以及
stage1 网格点失败原因分布（fail15 / fail_neigh / fail14 / fail13 / fail_br）。

## 关键常量（`end_surface_noise.cuh`）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `ES_WORLD_Y` | `72` | 目标 Y |
| `ES_CELLY` | `18` | `ES_WORLD_Y / 4` |
| `ES_PERIOD` | `49026.65646` | main 噪声周期（格） |
| `ES_PERIOD_RANGE` | `30000000` | 平移半径 |
| `ES_MAIN_PHASE_MOD` | `5` | 平移相位类数 |
| 门控链 | `55 / 60 / 100 / 110 / 130 / 137` | oct15 / 邻域 / oct14 / oct13 / stage2 / stage3 |
| `ES_STAGE2_RANGE/STEP` | `48 / 16` | stage2 |
| `ES_STAGE3_RANGE/STEP` | `16 / 2` | stage3 |
| `ES_BASE_FREQ` | `684.412` | 末地基础频率 |
| `ES_XZ_FACTOR` / `ES_Y_FACTOR` | `80 / 160` | 坐标缩放 |

## 保真度说明（重要）

- 噪声与判据对齐的是 **cubiomes `initSurfaceNoise(DIM_END)` + `C++learning`
  的 `end_period_common` 判据**，**不是** 官方 JAR。
- `ES_STEP1_MAX0` 用 55 而非 57：源码注释记录 57 在 10× 种子样本的 `env>140`
  点上有约 10% 漏检。
- 源码中标注 `TEMP` 的项：stage1 gradvec 默认 `allow1 / topk=4`，以及 `--profile`
  系列计数。它们影响速度与召回，属于可调实验默认值，用 `--no-s1-gradvec` 可关闭。
- 当前版本**不含**岛屿/高度（islands/height）判定，只做包络 + 周期晶格 + `vmain`。

## 来源

本仓库是从一个更大工作区的 `projects/end-surface-period-standalone/cuda/` 中
抽出的**最小可编译闭包**：`end_surface_period_scan.cu` → `end_surface_noise.cuh`
→ `end_phase_lut.cuh`，再无其它 include，三个文件内容与上游逐字节一致，
便于日后 diff 同步。

`.gitattributes` 按上游行尾固定：`.cu` 为 CRLF，`.cuh` 为 LF，
存储层统一 LF，因此 clone 后的文件与上游字节相同。
