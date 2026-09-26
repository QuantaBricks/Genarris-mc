# Genarris-mc 手册：安装与多构象 CSP

本仓库 = Genarris (`gnrs`，随机分子晶体生成) + 多构象驱动脚本 `confgnrs.py`。
本手册只讲**怎么装、怎么跑、结果在哪、出错怎么办**。Genarris 本身各任务参数见 `docs/config/*.md`。

> 给 AI 代理：先读「0. 速查」和「7. 约定与不变量」。所有命令都在仓库根目录 `$REPO` 下执行，
> 除非特别说明。不要改 `[workflow]`（被脚本强制覆盖），想改流程请改 `confgnrs.py` 里的 `TASKS`。

---

## 0. 速查

| 要做的事 | 命令 |
|---|---|
| 激活环境 | `source $REPO/.venv/bin/activate` |
| 小测试（构象排名，建 `cf1..cfN/`） | `python $REPO/confgnrs.py -c inp.conf --stage mini` |
| 全生成（跑所有 `cfN/` 并合并） | `python $REPO/confgnrs.py -c inp.conf --stage full` |
| 一次跑完两步 | `python $REPO/confgnrs.py -c inp.conf --stage all` |
| 单构象直接跑 Genarris | `mpirun -n 16 gnrs -c ui.conf` |
| 最终结果 | `<run_dir>/merged_structures.json` |

`inp.conf` 所在目录就是 `run_dir`，所有输出都写在那里。

---

## 1. 依赖

| 组件 | 版本（已验证） | 用途 |
|---|---|---|
| Linux + OpenMPI | Open MPI 4.1.2（需要 `mpicc`、`mpirun`） | 编译 C 扩展、并行生成 |
| Python | 3.10–3.11（本机 3.11.12，`<3.12` 硬性要求） | |
| [uv](https://docs.astral.sh/uv/) | 0.11 | 建虚拟环境（也可用 `python -m venv`） |
| xTB | 6.7.1，必须在 `PATH` 里 | 构象单点（GFN2）与晶体单点（GFN1 周期性） |
| RDKit | 2026.03 | 构象二面角网格 |

只跑 `generation → symm_rigid_press → dedup` **不需要** PyTorch / GPU。
MLIP 能量计算器（UMA、MACE 等）才需要，见 `README.md`。

---

## 2. 安装

```bash
git clone git@github.com:QuantaBricks/Genarris-mc.git
cd Genarris-mc                     # 以下记为 $REPO；cgenarris 已内置，无需 submodule

# 2.1 虚拟环境
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2.2 构建依赖（必须先装，下一步用 --no-build-isolation）
uv pip install "setuptools>=61.0" "setuptools-scm>=8" wheel "swig>=4.1,<4.3" Cython "numpy>=2.0,<2.3"
MPICC=$(which mpicc) uv pip install mpi4py --no-binary mpi4py

# 2.3 gnrs 本体（可编辑安装，会用 mpicc 编译 C 扩展）
uv pip install -e . --no-build-isolation

# 2.4 构象驱动额外依赖
uv pip install rdkit

# 2.5 xTB：下载 xtb-6.7.1 预编译包，加入 PATH（写进 ~/.bashrc）
export PATH=/path/to/xtb-6.7.1/xtb-dist/bin:$PATH
```

`confgnrs.py` 会按顺序找 gnrs 可执行文件：`$REPO/.venv/bin/gnrs` → `$REPO/gnrs_env/bin/gnrs` → `PATH` 里的 `gnrs`。
装在别处时在 `[conformer]` 里写 `gnrs_cmd = /abs/path/to/gnrs`。

### 2.6 验证安装

```bash
python -c "import gnrs, rdkit, mpi4py, ase; print('python ok')"
xtb --version | grep version        # 应显示 6.7.1
which gnrs mpirun
```

然后跑第 5 节的冒烟测试（约 1 分钟）。

---

## 3. 流程

```
mol.xyz
  │  --stage mini
  ├─ 1. 二面角网格：每个重原子可旋转单键取 n_angles 个角度，共 n_angles^N_rot 个构象
  │     每个做 xTB GFN2 单点 → 能量窗口内均匀取 2×max_conformers 个
  ├─ 2. 每个构象一个 micro-CSP（conformers/micro-cfX/）：
  │     gnrs: generation → symm_rigid_press → dedup
  │     只生成 micro_csp_spgs 这几个空间群，每个 num_structures_per_spg（[conformer] 里的）个
  ├─ 3. 对所有去重后的晶体做 xTB GFN1 周期性单点，每个构象取最低 E/分子
  │     排名 → 前 max_conformers 个写成 cf1/ … cfN/（cf1 最好）
  │
  │  --stage full
  ├─ 4. 依次在 cf1/ … cfN/ 里跑 gnrs（用你自己的 [generation] 参数；已完成的跳过）
  └─ 5. 按每个 cfN 最低 press_energy 排名，合并全部结构 → merged_structures.json
```

两个阶段的唯一区别在写进子目录 `inp.conf` 的 `[generation]`：

| | 小测试 `micro-cfX/` | 全生成 `cfN/` |
|---|---|---|
| `num_structures_per_spg` | `[conformer]` 的值（默认 20） | `[generation]` 的值 |
| `spg_distribution_type` | `[micro_csp_spgs]`（默认 `[14,2,19]`） | `[generation]` 的值 |
| 其余所有段 | 原样透传 | 原样透传 |

---

## 4. 配置文件 `inp.conf`

`[conformer]` 只给 `confgnrs.py` 用；其余段（除 `[workflow]`）原样复制进每个子目录的 `inp.conf` 交给 gnrs。

```ini
[conformer]
mol                     = mol.xyz     # 相对 inp.conf 所在目录
n_angles                = 5           # 每根可旋转键的角度数；网格大小 = n_angles^N_rot
energy_window           = 20.0        # kcal/mol，相对最低构象
max_conformers          = 20          # 进入全生成的构象数（小测试用 2 倍）
charge                  = 0
num_structures_per_spg  = 30          # 仅小测试
micro_csp_spgs          = 14,2,19     # 仅小测试
mpi_np                  = 16          # 每次 gnrs 的 MPI 进程数，也是 xTB 并行进程数
# gnrs_cmd              = /abs/path/to/gnrs

[master]
name                    = lbwl1       # 会被子目录名覆盖
molecule_path           = ["mol.xyz"] # 会被覆盖
Z                       = 4
log_level               = info

[generation]                          # 全生成用
num_structures_per_spg  = 200
spg_distribution_type   = pg_graded   # 或 standard / [14,2,19] / pg_graded:0.1
sr                      = 0.95
max_attempts_per_spg    = 100000000
tol                     = 0.01
ucv_mean                = predict
ucv_mult                = 1.5
max_attempts_per_volume = 10000000
generation_type         = crystal
natural_cutoff_mult     = 1.2

[symm_rigid_press]
sr                      = 0.85
method                  = BFGS
tol                     = 0.01
natural_cutoff_mult     = 1.2
debug_flag              = False
maxiter                 = 5000

[dedup]
stol                    = 0.5
ltol                    = 0.5
angle_tol               = 10
```

`[workflow]` 写不写都一样，脚本固定为 `['generation', 'symm_rigid_press', 'dedup']`。

### `spg_distribution_type` 取值

| 值 | 含义 |
|---|---|
| `standard` | 所有与 Z 兼容的空间群，每个 `num_structures_per_spg` 个 |
| `[14,2,19]` | 只跑列出的空间群 |
| `pg_graded` | CSD 频率前 20 的空间群，目标数从 500（P2₁/c）二次递减到 1；**忽略** `num_structures_per_spg`。每构象约 1400–1550 个原始结构 |
| `pg_graded:0.1` | 同上按比例缩小 10 倍，用于快速测试（每构象约 150 个） |

### 规模估算（lbwl-1，Z=4，3 根可旋转键）

| 步骤 | 规模 | 耗时（16 核） |
|---|---|---|
| 构象网格 `n_angles=5` | 125 个 xTB 单点 | 秒级 |
| micro-CSP，每构象 3 spg × 30 | 每构象十几秒 | 40 个构象约 10 分钟 |
| 全生成 `pg_graded`，16 MPI | 每构象约 1500 结构 | 约 70–90 s/构象 |

网格随可旋转键数指数增长：5 根键 × `n_angles=5` = 3125 个构象。键多时降低 `n_angles`。

---

## 5. 冒烟测试

```bash
mkdir -p /tmp/gnrs_smoke && cd /tmp/gnrs_smoke
cp $REPO/test/lbwl1/mol.xyz .
cat > inp.conf <<'EOF'
[conformer]
mol = mol.xyz
n_angles = 3
max_conformers = 2
num_structures_per_spg = 2
mpi_np = 8

[master]
name = smoke
molecule_path = ["mol.xyz"]
Z = 4

[generation]
num_structures_per_spg = 2
spg_distribution_type = [14]

[symm_rigid_press]
method = BFGS
EOF
python $REPO/confgnrs.py -c inp.conf --stage all
```

成功标志（约 1 分钟）：

- 打印 `Created cf1/ .. cf2/`；
- 打印 `Merged N structures from 2 conformers`；
- 当前目录下有 `merged_structures.json`。

---

## 6. 输出

```
run_dir/
├── inp.conf, mol.xyz
├── _conformer_work/sp/conf_XXXXX/     # 网格构象 xTB 单点（可删）
├── conformers/
│   ├── conf_NNNN.xyz                  # 能量窗口内选出的构象
│   └── micro-cfX/                     # 小测试
│       ├── inp.conf, mol.xyz, run.log
│       ├── structures/dedup/structures.json
│       └── _xtb_crystal/struct_NNNN/  # 晶体 xTB 单点（可删）
├── cf1/ … cfN/                        # 全生成，cf1 = 小测试排名第一
│   ├── inp.conf, mol.xyz, run.log
│   └── structures/{generation,symm_rigid_press,dedup}/structures.json
└── merged_structures.json             # 所有 cfN 去重后结构
```

`merged_structures.json`：键为 `cfN__<hash>`，值为 ASE Atoms 的 JSON 字典。
`info` 里有 `press_energy`、`spg`、`conformer_id`（如 `cf3`）、`conformer_mol_path`。读取方法：

```python
import json
from ase import Atoms
from ase.io.jsonio import decode

data = json.load(open("merged_structures.json"))
structs = {k: Atoms(**{f: decode(json.dumps(v[f])) for f in ("numbers", "positions", "cell", "pbc")})
           for k, v in data.items()}
best = min(data, key=lambda k: data[k]["info"].get("press_energy", float("inf")))
```

---

## 7. 约定与不变量（改代码前必读）

- **断点续跑**：全生成和小测试都以 `structures/dedup/structures.json` 是否存在判断「已完成」，存在就跳过。
  要重跑某个构象，删掉它的 `structures/` 目录。
- **外层脚本不能用 mpirun 启动**。`confgnrs.py` 是单进程，每个构象各起一个独立的 `mpirun` 子进程。
  父进程里一旦 import 了会初始化 MPI 的 gnrs 模块（`gnrs.output`、`gnrs.parallel`），子进程的 `mpirun` 会静默失败。
- **mpirun 的 stdout/stderr 必须写文件**（`run.log`），不能是 PIPE，否则 OpenMPI 的输出转发会死锁。
- **线程**：所有子进程设置 `OMP/MKL/OPENBLAS_NUM_THREADS=1`，并发数由 `mpi_np` 控制。
  共享机器上保持 `mpi_np ≤ 16`。
- **xTB 方法**：分子用 GFN2；晶体只能用 GFN1（GFN2 不支持 PBC，GFN-FF 更慢）。能量取 stdout 最后一行 `TOTAL ENERGY`，单位 Eh。
- **排名依据**：小测试按 xTB GFN1 的 E/分子（`E/Z`），全生成按 `press_energy`（rigid press 的几何评分，不是物理能量）。
- **Python API**：`confgnrs.sample_and_filter(mol_path, n_angles, energy_window, max_conformers, charge, work_dir, n_workers=1)`
  返回 `(paths, energies_Eh)`，按能量升序排列。其他函数都视为内部实现。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `micro-cfX FAILED (rc=1)`，`run.log` 为空 | 外层进程已初始化 MPI（见第 7 节），或 `gnrs_cmd` 路径不对。直接进子目录跑 `mpirun -n 4 <gnrs_cmd> -c inp.conf` 排查 |
| `All xTB single-points failed` | `xtb` 不在 `PATH`，或 `charge` 设错 |
| `WARNING: all xTB calculations failed, cannot rank` | 所有 micro-CSP 都没产出结构；看各 `micro-cfX/run.log` |
| `No cfN/ folders ... run --stage mini first` | 在错误的目录运行，或还没跑小测试 |
| 生成卡住很久 | `max_attempts_per_spg` 太大且体积不合适；检查 `Z`、`ucv_mult` |
| 网格构象太多 | 降低 `n_angles`（网格规模是 `n_angles^N_rot`） |
| `ImportError: rdkit` | 没在 `.venv` 里运行，或漏装第 2.4 步 |
| 编译 C 扩展失败 | 确认 `mpicc` 可用、`swig<4.3`；BLAS/LAPACK 库名不同时改 `setup.py`（见 README） |
