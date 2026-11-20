# FYP BO-SVM

实验项目，研究 Bayesian optimisation 在 EEG 预处理和 RBF-SVM 参数选择中的表现。仓库包含实验代码、Bonn 数据文件、保存的结果、分析脚本、PNG 图件。

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
| `nested_bo_rs.py` | 外层五折、内层五折的 BO/RS 配对实验，共享 16 个初始候选 |
| `run_nested_experiment.ps1` | 后台启动或断点续跑新实验 |
| `README_NESTED_EXPERIMENT.md` | 新实验的固定协议、输出和运行说明 |
| `nested_mapping_comparison.py` | 沿用新嵌套划分，补跑 normal、exponential 和 Gamma BO |
| `run_nested_mappings.ps1` | 后台启动或续跑三种映射的 15 次 BO 搜索 |
| `README_NESTED_MAPPINGS.md` | 四种映射同协议比较的实现与输出说明 |
| `parameter_priors.py` | 四种预处理参数映射 |
| `Dataset/` | 项目实际使用的 500 个 Bonn EEG 片段 |
| `results/09_fixed_fold_segment_accuracy/` | 保留的 55 维 BO/RS、四种映射及相关分析结果 |
| `figures/chapter5/` | 实际实验的搜索曲线、参数可视化、混淆矩阵和 GP 诊断图 |
| `chapter4_draft/` | 保留的实验协议审计、搜索历史统计和训练集频谱分析，目录名沿用原项目 |
| `tmp/analyze_search_behaviour.py` | 已保存搜索日志的阶段与映射分析 |
| `tmp/build_chapter5_seed44_figures.py` | 第五章图件的绘图代码，初步输出到 `tmp/` |
| `plot_grouped_parallel_coordinates_v2.py` | 随机初始化与 BO 引导阶段的平行坐标图 |
| `README_RUN_STEPS.md` | 运行命令及各实验分支说明 |

数据出处见 [Bonn 官方下载页](https://neurophysicsbonn.de/downloads)。使用数据时引用 Andrzejak 等人，Physical Review E 64, 061907 (2001)，DOI 10.1103/PhysRevE.64.061907。

## 当前实验状态

三个完整实验均已完成，协议和结果分别保存，不能混用不同投票规则的结果。

| 实验 | 规则与范围 | 五折外层 accuracy，均值 ± SD |
| --- | --- | --- |
| `results/10_nested_cv_shared_initialization/` | 原多数投票，平票取最小编号；uniform BO/RS 与固定参考 | BO 89.4 ± 3.36%，RS 86.6 ± 5.27%，固定参考 86.8 ± 8.11% |
| `results/11_nested_mapping_comparison/` | 原投票规则；四种映射使用相同嵌套划分 | uniform 89.4 ± 3.36%，normal 85.8 ± 6.22%，exponential 86.8 ± 3.27%，gamma 87.0 ± 5.34% |
| `results/13_nested_score_tiebreak/` | 平票时比较平均 SVM 决策分数；uniform BO/RS 与固定参考 | BO 88.0 ± 2.83%，RS 87.4 ± 5.22%，固定参考 86.6 ± 7.50% |

实验 13 的规则根据训练集内层试跑确定，是补充性再分析，并非全新独立验证。不能根据哪套外层测试分数更高来选择最终协议。实现、后台启动和断点恢复分别见 `nested_score_tiebreak.py`、`run_nested_score_tiebreak.ps1` 和 `README_NESTED_SCORE_TIEBREAK.md`。原实验完整说明见 `README_NESTED_EXPERIMENT.md`；映射比较见 `README_NESTED_MAPPINGS.md`。

`results/12_training_feature_pilot/` 保存 55 维、catch24、23 维和投票规则的训练集内层筛选记录。三种特征的最高内层 CV 分别为 85.25%、84.50%、80.50%。这些是固定预处理下的十二组参数试跑，不是外层测试结果，也不是完整五参数 BO 搜索。细节见 `README_GITHUB_FEATURE_PILOT.md`。

换设备后若需重做投票试跑，先运行 `python -B rebuild_training_feature_cache.py`，重建未上传的训练特征缓存。只有 catch24 相关试跑需要额外安装 `pycatch22==1.0.2`，主实验不需要。

## 当前绘图代码

`plot_nested_chapter5.py` 读取原实验 10 的结果。图 5.1 为五折 BO 搜索历史和候选阶段分布；图 5.2 为三种方法的外层准确率均值、样本标准差、散点及 BO/RS 配对比较；图 5.3 为合并五个外层折的混淆矩阵。图 5.2 也可单独用 `plot_nested_outer_performance.py` 生成。该脚本还可生成逐类 recall、specificity 图及 CSV 表，但不要求把所有生成图放进论文。

`plot_nested_candidate_scores.py --fold 5` 展示单折候选分数。`plot_example_signals.py` 生成第四章五类 EEG 示例信号，使用各行独立纵轴。旧版 seed 44 图保留为探索性记录，不应与新的嵌套结果混用。旧的 `fig5_4_exploratory_mapping_test_accuracy.png` 来自实验 09，不代表实验 11 的新映射结果。

原重复划分实验保留在 `results/09_fixed_fold_segment_accuracy/`，划分间有重叠且开发阶段曾查看测试结果。它们仅作为探索性记录。完整目录说明见 `results/README_RESULTS.md`。

`.gitignore` 排除本机设置、临时依赖、特征缓存、日志、论文稿件、演示文稿和 PDF。保留实验代码、必要 PNG、CSV/JSON 结果和搜索断点。
