# FYP BO-SVM

实验项目，研究 Bayesian optimisation 在 EEG 预处理和 RBF-SVM 参数选择中的表现。仓库包含实验代码、Bonn 数据文件、保存的结果、分析脚本、实验笔记本和 PNG 图件。

## 在另一台设备上开始

先安装 Git 和 Python 3.10，使用自己的 GitHub 账号访问这个私有仓库。在终端中执行下面的命令。

```bash
git clone https://github.com/fenglinwang10-Camille/FYP-BO-SVM.git
cd FYP-BO-SVM
python -m venv .venv
```

Windows PowerShell 激活环境并安装依赖。

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS 或 Linux 激活环境后安装依赖。

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 依赖版本来自本项目使用的 Python 3.10.20 环境。安装依赖后可以继续运行实验和绘制图件。

## 跨设备同步修改

每次开始工作前先拉取 GitHub 上的更新。

```bash
git pull --ff-only
```

完成修改后，检查文件列表并提交上传。

```bash
git status
git add .
git commit -m "Update experiments and results"
git push
```

换设备前先提交并推送，另一台设备开始时再拉取。如果两台设备同时修改同一文件，Git 可能要求合并冲突。终端首次提交时，可根据 Git 提示设置自己的提交姓名和邮箱。

## 主要文件

| 路径 | 内容 |
| --- | --- |
| `grouped_bo_optimizer.py` | 当前片段固定五折验证的 BO 实验 |
| `grouped_method_comparison.py` | 与随机搜索及固定参考的比较 |
| `parameter_priors.py` | 四种预处理参数映射 |
| `Dataset/` | 项目实际使用的 500 个 Bonn EEG 片段 |
| `results/09_fixed_fold_segment_accuracy/` | 当前 55 维主实验及明确命名的补充实验 |
| `figures/chapter5/` | 实际实验的搜索曲线、参数可视化、混淆矩阵和 GP 诊断图 |
| `chapter4_draft/` | 保留的实验协议审计、搜索历史统计和训练集频谱分析，目录名沿用原项目 |
| `tmp/analyze_search_behaviour.py` | 已保存搜索日志的阶段与映射分析 |
| `tmp/build_chapter5_seed44_figures.py` | 第五章图件的绘图代码，初步输出到 `tmp/` |
| `plot_grouped_parallel_coordinates_v2.py` | 随机初始化与 BO 引导阶段的平行坐标图 |
| `README_RUN_STEPS.md` | 运行命令及各实验分支说明 |

数据出处见 [Bonn 官方下载页](https://neurophysicsbonn.de/downloads)。使用数据时引用 Andrzejak 等人，Physical Review E 64, 061907 (2001)，DOI 10.1103/PhysRevE.64.061907。

## 当前实验状态

主实验使用 55 维特征、片段固定五折验证、片段级 accuracy 目标，每次搜索评价 50 个配置。Uniform BO 和 RS 已保存 seeds 42–46 的配对结果，四种映射各有五组 BO 结果。固定参考在 seeds 42 和 44 上已保存结果。其他版本的结果目录保留了历史实验，具体协议见 `results/README_RESULTS.md`。

`figures/chapter5/` 中的图 5.1 展示五组 uniform BO，图 5.2 和 5.3 展示 seed 44，图 5.4 展示 seed 44 的四种映射。其他参数可视化的 seed 和实验口径见图片及对应脚本。

`.gitignore` 保持仓库聚焦实验，排除本机设置、缓存、临时预览、论文稿件、演示文稿、PDF 和打包导出文件。实验脚本生成的 PNG、CSV 和 JSON 可随代码一起提交。
