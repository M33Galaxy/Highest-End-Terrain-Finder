/* cubiomes_height_probe.c
 *
 * Ground truth 探针：直接调用 cubiomes-end 的 getEndSurfaceHeight，用来对拍
 * 本仓库 end_island_noise.cuh / period_height_check 报出的高度。
 * 不参与本仓库的正常构建（没有 cubiomes 依赖），只在需要复核时手动编。
 *
 * 用法：
 *   cubiomes_height_probe <seed> <x> <z> [<x> <z> ...]     → 每行 "x z h"
 *
 * 编译（把 CUBS 指到 cubiomes-end 目录）：
 *   CUBS=/path/to/cubiomes-end
 *   mkdir -p /tmp/cubobjs
 *   for s in noise biomenoise layers util biomes quadbase generator xradv; do \
 *       gcc -O2 -std=gnu99 -I"$CUBS" -c "$CUBS/$s.c" -o "/tmp/cubobjs/$s.o"; done
 *   gcc -O2 -std=gnu99 -I"$CUBS" -o cubiomes_height_probe cubiomes_height_probe.c \
 *       /tmp/cubobjs/*.o -lm
 *
 * Windows / MSYS2 同样可用（gcc + msys 路径）。
 * 本机实测：与 period_height_check 在已知锚点、±2990 万格的循环点、以及工具
 * 报出的命中点上逐点一致（见 README「已做的验证」）。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "biomenoise.h"

int main(int argc, char **argv)
{
    uint64_t seed;
    int i;

    if (argc < 4 || (argc % 2) != 0) {
        fprintf(stderr, "usage: %s <seed> <x> <z> [<x> <z> ...]\n", argv[0]);
        return 2;
    }
    seed = strtoull(argv[1], NULL, 10);
    for (i = 2; i + 1 < argc; i += 2) {
        const int x = atoi(argv[i]);
        const int z = atoi(argv[i + 1]);
        printf("%d %d %d\n", x, z, getEndSurfaceHeight(MC_1_21, seed, x, z));
    }
    return 0;
}
