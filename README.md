# 末地最高地形搜索（周期扫描 + 真实高度校验）

| 文件 | 说明 |
| --- | --- |
| `end_surface_period_scan.cu` | **主程序**（CUDA）：按末地 SurfaceNoise 周期扫种子区间，输出高包络候选点 CSV |
| `period_height_check.cu` | **高度校验**（host）：读上面的 CSV，查周期循环点附近有没有真实 Y≥73 地表 |
| `end_surface_noise.cuh` | 末地噪声原语（两程序共用） |
| `end_phase_lut.cuh` | y_offset 相位 LUT（主程序用，自动生成勿手改） |
| `end_island_noise.cuh` | 真实末地地表高度（高度校验用） |
| `tools/cubiomes_height_probe.c` | 可选：对拍用 cubiomes 探针，不参与构建 |

## 构建

```bash
# 主程序（需要 CUDA Toolkit）
nvcc -O3 -std=c++17 -arch=native -o end_surface_period_scan end_surface_period_scan.cu

# 高度校验（只要 C++17，不需要 CUDA；.cu 要显式 -x c++）
g++ -O3 -std=c++17 -pthread -x c++ -o period_height_check period_height_check.cu -lm
```

## 主程序：`end_surface_period_scan`

```bash
./end_surface_period_scan --start-seed 0 --end-seed 9999999 --out hits.csv
```

`--start-seed` / `--end-seed` / `--out` 必填，`--end-seed ≥ --start-seed`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--period-range` | `122566` | 周期平移半径 ±格。默认 = **一个总周期** `±T/2`（`T = 5×49026.65646 = 245133.2823`）⟹ `kMax=2`，5×5 = 25 个晶格点 |
| `--stage1-range` / `--stage1-step` | `24320` / `128` | stage1 网格半径 / 步长 |
| `--s1-coarse-thr` | `55` | oct15 粗筛阈值（门控链 55/60/100/110/130/137） |
| `--s1-neigh` / `--no-s1-neigh` | 开 | ±64 的 9 点邻域确认 |
| `--s1-grid-par-threads` | `256` | 每种子 block 的线程数 |
| `--max-hits` | `2000000` | 命中上限 |
| `--no-phase` | 关 | 关闭 y_offset 相位预筛（最慢、最保守，用于对照） |
| `--profile` | 关 | 打印各阶段耗时占比 |

其余（`--s1-hier256*`、`--s1-gradvec*`、`--s1-neigh-thr`、`--celly`、`--profile-s1` 等）见 `--help`。

输出 `--out` CSV，表头：

```
seed,branch,stage1_x,stage1_z,envelope,peak_x,peak_z,tx,tz,vmain
```

`branch` = `min`/`max`；`envelope` = 峰值处包络；`peak_x/z` = 精修峰值；`tx/tz` = 周期平移后复核通过的候选坐标。

## 高度校验：`period_height_check`

```bash
./period_height_check --hits hits.csv
./period_height_check --hits hits.csv --envelope-thr 146 --out hits73.csv
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--hits FILE` | 必填 | 主程序输出的 CSV |
| `--envelope-thr F` | `148` | 只查 `envelope > F` 的种子（合法区间 `137..152`） |
| `--range N` | `30000000` | 循环点搜索半径 ±格（合法区间 `122566..30000000`） |
| `--window W` / `--window-step S` | `16` / `1` | 每个循环点检查 ±W 格、步长 S |
| `--period F` | `245133.2823` | 循环点步长（整周期 `T = 5P`） |
| `--loop MODE` | `2d` | `2d`（全部 kx,kz）/ `cross`（两轴各一条线）/ `1d` |
| `--depth-thr F` | `62` | 跳过 `D_eff < F` 的列（加速用；`--no-depth-filter` 关闭） |
| `--threads N` | 硬件并发 | 线程数 |
| `--out FILE` | 无 | 把所有 ≥73 的列写 CSV |
| `--selftest` | — | 内部一致性自检 |

输出：**出现 74 立即打屏**（`[74] seed=… x=… z=…`）；每个种子打印循环点数、含 ≥73 的循环点数、≥73 的列数、最大高度与最佳坐标；末尾给合计与结论。
高度口径 = cubiomes `getEndSurfaceHeight`（含内岛 + 25×25 外环外岛 + 1.14+ overflow void），不是 Official JAR。

## 结果

目前找到的最高地形是 **Y=73**；**Y=74 可能需要单台 4090D 连续跑 15 天**才能找到。
