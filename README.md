# 末地 SurfaceNoise 周期高点扫描器 + 真实高度校验（CUDA / host）

两个工具：

1. **`end_surface_period_scan`**（CUDA）— 用末地 `SurfaceNoise` 的 **main 噪声周期**
   `P ≈ 49026.656` 格，在给定种子区间里找出**末地最高地形**候选点：先在 Y=72
   （`celly=18`）平面上筛出落在 `SurfaceNoise` 包络高位的点（min/max 两个 branch），
   再把该点按周期晶格平移 `k·P`（`|k·P| ≤ period_range`），用 `vmain` 逐点复核，
   输出所有仍然成立的 `(seed, tx, tz)`。
2. **`period_height_check`**（host）— 读上面的 CSV，对 `envelope` 超阈值的种子，
   按**真实地表噪声周期** `T = 245133.2823`（`= 5P`）枚举周期循环点，
   在每个点 ±16 格内用 cubiomes 口径的**真实末地地表高度**判定是否 ≥ 73，
   出现 74/75 立刻打屏。详见下文「高度检查」一节。

扫描程序是单文件 CUDA 可执行文件；高度检查是单文件 host 程序。两者都只依赖同目录头文件，
无第三方库、**不链接 cubiomes**（高度函数是按 cubiomes 逐字移植并对拍验证的）。

```text
end_surface_period_scan.cu   扫描主程序（kernel + CLI + CSV 输出，nvcc）
end_surface_noise.cuh        末地噪声原语：Java LCG、Perlin、oct13/14/15、vmain、y_offset 预筛
end_phase_lut.cuh            y_offset 相位 32-bin LUT 权重（自动生成，勿手改）
period_height_check.cu       高度检查工具（host，g++/nvcc 均可编）
end_island_noise.cuh         真实末地地表高度：外岛 simplex + getEndHeightNoise + 列/插值
```

## 构建

扫描程序需要 CUDA Toolkit（`nvcc`）；高度检查只需要一个 C++17 编译器（无需 CUDA）：

```bash
# 扫描器：Linux / Colab / AutoDL
nvcc -O3 -std=c++17 -arch=sm_89 -o end_surface_period_scan end_surface_period_scan.cu

# 扫描器：Windows
nvcc -O3 -std=c++17 -arch=native -o end_surface_period_scan.exe end_surface_period_scan.cu

# 高度检查（host，注意 .cu 要显式 -x c++，与本仓库其它 host 版 .cu 一致）
g++ -O3 -std=c++17 -pthread -x c++ -o period_height_check period_height_check.cu -lm
```

换架构改 `-arch` 即可（例如 `sm_75` / `sm_86` / `sm_89`）。扫描程序在 host 侧检查
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

## 高度检查：`period_height_check`

扫描程序只输出 **SurfaceNoise 包络**高的候选点，它**不是**高度。要回答「这些周期循环点
那里真的有 Y≥73 的地表吗」，用 `period_height_check`：

```bash
./period_height_check --hits hits.csv                      # 默认: env>148, ±3000万, 2d, ±16
./period_height_check --hits hits.csv --envelope-thr 146   # 放宽到 75 个种子
./period_height_check --hits hits.csv --range 1225660 --out hits73.csv
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--hits FILE` | 必填 | 扫描程序输出的 CSV |
| `--envelope-thr F` | `148` | 只查 `envelope > F` 的种子（合法区间 `137..152`，超出直接报错） |
| `--range N` | `30000000` | 循环点搜索半径 ±N 格（合法区间 `122566..30000000`） |
| `--window W` | `16` | 每个循环点检查 ±W 格 |
| `--window-step S` | `1` | 窗口步长（`1` = 穷举 33×33 = 1089 列） |
| `--period F` | `245133.2823` | 真实地表噪声周期 `T = 5P`（= `ES_MAIN_PHASE_MOD × ES_PERIOD`） |
| `--loop MODE` | `2d` | `2d`（全部 kx,kz 组合）/ `cross`（两轴各一条线）/ `1d`（单轴） |
| `--threads N` | 硬件并发 | 循环点级动态调度 |
| `--max-seeds N` | 全部 | 只取 envelope 最高的 N 个种子 |
| `--max-loop-points N` | 不限 | 调试用：每种子最多枚举 N 个循环点 |
| `--out FILE` | 无 | 把所有 ≥73 的列写 CSV |
| `--quiet` | 关 | 不打进度 |
| `--selftest` | — | 内部一致性自检（见下） |

`--range` 用 `T=245133.2823` 换算：`K = floor(range / T)`，每轴 `2K+1` 个点，
`2d` 模式每种子 `(2K+1)²` 个循环点。注意 **`--range 122566` 时 `K=0`**，
每轴只有 1 个点，即只检查峰值本身（这是规格下限的必然结果）。

### 为什么只查 celly 18

末地地表的列密度是 `lerp(upper_drop[y], -3000, noise + depth)` 再 `lerp(lower_drop[y], -30, …)`，
其中 `upper_drop[y] = clamp((78-y)/64)`。成实心需要
`u·(noise+depth+3000) > 3000`，而 `depth ≤ +72`、`noise ≲ +152`，所以
`u > 15/16` 是硬条件 —— 这恰好是 **cy=18**（`(78-18)/64 = 15/16`，cubiomes
`biomenoise.c:602-609` 的注释给出同一结论）。于是：

- 高度 ≥ 73 ⟺ cy=18 内存在实心块（`y=1/2/3` 任一 `noise > 0`）
- 高度 74 ⟺ `y=3` 不成立且 `y=2` 成立；高度 75 ⟺ `y=3` 成立

因此每列只需 **4 个 cell × celly{18,19} = 8 个噪声值**，而不是 4×33 = 132 个。
工具还按窗口缓存 cell（每个 cell 的 depth 只算一次），否则每列会重复算 4 次
25×25 岛深，慢约 30 倍。

### 高度口径（重要）

`end_island_noise.cuh` 逐字移植自 **cubiomes-end**：

- `setEndSeed`（`setSeed` → `consumeCount(17292)` → perlinInit）＋ `sampleSimplex2D`
  得到 25×25 外环的**外岛**项；
- `getEndHeightNoise` = `min(64(x²+z²), 25×25 邻域内的 rsq'·v²)`；
- `sampleSurfaceNoiseBetween(sn, cx, cy, cz, -128, +128)`（16 个 min/max octave + 8 个 main）；
- `sampleNoiseColumnEnd` 的 `upper_drop`/`lower_drop` 与 `getSurfaceHeight` 的
  `lerp3(dy,dx,dz,…)`、`blockspercell=4`、自上而下第一个 `noise>0`；
- **1.14+ overflow void**：`(int)(cx²+cz²) < 0` 时整列 void（无地形）。这是真实行为，
  在 ±10⁷ 量级会否掉相当一部分坐标，不能省。

仓库里其它高度实现（`front_dragon_hsum.cu` 的 `end_depth_simple`、
`stage2_hotpath.cuh` 的 `end_island_depth`）**只在原点附近成立**：它们只有内岛项
`100 - sqrt(64(cx²+cz²)) - 8`，`|cell| > ~25` 后被 clamp 到 −108，此时 cy=18 需要
`noise > 308`（不可能），高度上限掉到 cy=14（Y≈59）。用它们跑 ±3000 万会**恒报"无地形"**。

### 默认档实测结果（end-hits-2B.csv：1e8–2e9 种子，51,128 个命中种子）

```bash
./period_height_check --hits end-hits-2B.csv        # 8 线程, 88 秒
```

`envelope > 148` → 13 个种子；每种子 `2d` / `±3000万` → `K=122` → 60,025 个循环点；
共 780,325 个循环点 × 1089 列：

| 项 | 值 |
| --- | --- |
| 含 ≥73 的循环点 | **9,547**（1.2%） |
| ≥73 的列 | **366,031** |
| 最大高度 | **73** |
| 出现 74 / 75 | **0 / 0** |
| 有 ≥73 的种子 | **11 / 13**（`525351041`、`378335108` 没有） |

几个值得注意的现象：

- **最高就是 73，一个 74/75 都没有。** 机制上说得通：cy=18 内 `upper_drop` 对
  y=1/2/3 是同一个 u，差别只来自插值的 `dy`；而更高一层的 `col[cy=19]` 被拉向 −3000
  更多，所以 `dy` 越小的 y 越容易为正 —— 也就是 **y=1（高度 73）最先成立**，
  74/75 需要明显更强的密度。整个 workspace 找的 Y73/74 里，74 属于罕见事件。
- **envelope 排名 ≠ 高度。** 最高 envelope 的种子 `1400753836`（152.399）在峰值 ±16
  内的最大高度只有 69，全部 60,025 个循环点里只有 429 个含 ≥73；而 `694195937`
  （150.329）在峰值 (8216,−14336) 本身就是 73，1,922 个循环点含 ≥73。
  envelope 只是 SurfaceNoise 侧的代理量，能不能到 73 还取决于当地的岛深/外岛项。

### 成本（本机 8 逻辑核，`2d` / `±16` / step 1，每种子 60,025 循环点 ≈ 6.8 s）

成本 ≈ `种子数 × (2K+1)²`，两点缩放规律：

| 缩放 | 倍率 |
| --- | --- |
| `--range` 从 3000 万降到 123 万（K=122→4） | ×1/741 |
| `--window-step` 从 1 改 2（1089→289 列） | ×1/3.8 |

按阈值外推（默认 range）：

| `--envelope-thr` | 种子数 | 预计 |
| --- | --- | --- |
| 148 | 13 | 88 s（实测） |
| 146 | 75 | ~9 min |
| 145 | 203 | ~23 min |
| 144 | 451 | ~51 min |
| 137 | 51,128 | ~4 天（需 GPU 化，或把 range 缩到 ±123 万 → ~8 min） |

### 输出

- **立刻打屏**（满足「出现 74 直接打印」）：`[74] seed=… x=… z=… (peak=… k=(kx,kz) base=…)`
- 每个种子的汇总：循环点数、含 ≥73 的循环点数、≥73 的列数、最大高度、最佳坐标
- 末尾合计 + 结论（这些种子里到底有没有 ≥73）
- `--out` 另把所有 ≥73 的列写 CSV

### 已做的验证

- `--selftest`：缓存版（`es_end_height73_cached`）== 直算版（`es_end_height73`）
  == 完整 132 列高度版（`es_end_height_exact`）在 ≥73 上的一致性，50 例全过。
- **与 cubiomes 对拍**：用 `tools/cubiomes_height_probe.c`（直接调 cubiomes-end 的
  `getEndSurfaceHeight`，编译方式见文件头注释）在 5 个种子 × 已知锚点
  （`h(-29,28)=69`、`h(0,0)=62`）以及 13 个 `envelope>148` 种子 × 25 个远距离循环点
  （含 `k=±122`，即 ±2990 万格，覆盖 float 截断与 void 分支）共 325 个坐标上，
  **逐点完全一致（0 处不一致）**。
- **对工具实际报出的命中点对拍**：`--out` 产出 651 条 `h=73` 记录，随机抽 8 条
  用 cubiomes 复核，**8/8 一致**。

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
- 扫描程序**不含**岛屿/高度（islands/height）判定，只做包络 + 周期晶格 + `vmain`；
  「真实高度」由 `period_height_check` 负责，两者口径都在 cubiomes 侧对拍过。

## 来源

扫描器的三个文件是从一个更大工作区的
`projects/end-surface-period-standalone/cuda/` 中抽出的**最小可编译闭包**：
`end_surface_period_scan.cu` → `end_surface_noise.cuh` → `end_phase_lut.cuh`，
再无其它 include，三个文件内容与上游逐字节一致，便于日后 diff 同步。

`.gitattributes` 按上游行尾固定：`.cu` 为 CRLF，`.cuh` 为 LF，
存储层统一 LF，因此 clone 后的文件与上游字节相同。

高度侧的两个文件是本仓库新增的（上游没有等价物）：

- `end_island_noise.cuh` — 逐字移植自
  `projects/codex-minecraft-seed-methodology/third_party/cubiomes-end/`
  的 `biomenoise.c`（`setEndSeed` / `getEndHeightNoise` / `sampleSurfaceNoiseBetween` /
  `sampleNoiseColumnEnd` / `getSurfaceHeight`）、`noise.c`（`sampleSimplex2D` /
  `simplexGrad`）、`rng.h`（`skipNextN`），并保持 `EsPerlin` 与本仓库
  `end_surface_noise.cuh` 的表示一致 —— 所以**不引入 cubiomes 依赖**。
- `period_height_check.cu` — 新写的工具。

移植的验证方式见上文「已做的验证」：用 cubiomes-end 源码现编
`getEndSurfaceHeight` 探针对拍。
