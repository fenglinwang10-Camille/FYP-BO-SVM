# FYP 项目交接：研究、代码、实验、论文与图件

更新时间：2026-11-15（Asia/Shanghai）。读者：完全没有此前对话上下文的新对话/协作者。

本文件由当前对话历史与本次只读检查的本地代码、README、协议和结果表共同整理。**“文件中存在”“本对话讨论过”“已确定用于论文”是三种不同状态。** 本次没有运行新实验，没有改变生产模型，没有替用户决定采用新的特征版本。

## 0. 新对话先读：最重要的约束

1. 用户在写数学本科 FYP，主题是用 Bayesian optimisation（BO）自动选择 EEG 特征提取/预处理及 RBF-SVM 参数，不是提出新的临床诊断系统。
2. 当前生产主实验是 **Bonn 五分类、55 维特征、片段固定五折验证、片段级 accuracy 目标**。结果根目录为 `results/09_fixed_fold_segment_accuracy/`，协议标识 `recording_fixed_v2`。
3. 本地也存在 53 维去重、58 维四级小波、仅调 SVM 等补充实验。不能混进 55 维主实验，不能据其存在就将论文改成 53 维。
4. 最近明确的章节决定：**第三章 Mathematical Background；第四章 Implementation and Experimental Setup；第五章 Results and Discussion。** 原先放在第四章的结果比较、映射分析、错误分析、讨论已经决定移到第五章。
5. 主研究问题是“同样预算下 BO 相对随机搜索是否有帮助”；四种映射是扩展分析，不要与主目标并列夸大。
6. 主报告指标为 segment-level accuracy、seizure sensitivity；错误分析为每类 recall 与混淆矩阵。Seizure sensitivity 就是 S 类 recall。不要又擅自加入 macro 指标或每类 F1。
7. 用户强调保留并适当补充“为什么这样做”的理由，但不能虚构数值范围来自文献、PSD 或预实验。固定配置不是“文献标准参数”，50 次预算不是经证明最优。
8. 用户通常只要聊天中的英文 LaTeX，不要自动生成 Word/PDF 文档；本次明确要求创建 `handover.md`，是对这一偏好的特定例外。
9. 论文图只用于 A4 论文，不需要 PPT。最新 GP/EI 图要保留所有关键标记，用图例说明，**不要解释箭头**。
10. 很多最终修订仅在聊天中交付，**本地旧 `.tex` 不是自动同步的最终稿**。恢复论文编辑时，应先核对用户最新稿。

## 1. 项目目标与环境

论文工作标题：`An Investigation of Bayesian Optimisation for Automated Parameter Selection in EEG Feature Extraction`。标题曾用于用户的绘图要求；最终学校提交题目应以用户最新论文为准。

核心任务：将 BO 与 EEG 流水线连接，对过滤频带、窗口长度及分类器参数联合选参；用验证分数引导搜索，最终在留出测试片段上评价；与固定参考配置、随机搜索比较，并进行参数映射扩展分析。

项目目录：`C:/Users/王枫淋/Desktop/FYP`。默认 shell 为 PowerShell。

本次检查 `git log` 显示当前 `master` **尚无提交**，大量文件未跟踪。因此不能依赖 Git 提交史推断谁最初写了特征、还原历史版本或自动回滚。不要贸然执行清理或覆盖旧结果。

可用/已使用过的 Python：

- `C:/Users/王枫淋/AppData/Local/Programs/Python/Python310/python.exe`：本对话生成 Matplotlib GP/EI 图已成功。
- `C:/Users/王枫淋/.codex/skill-runtimes/figgenie/Scripts/python.exe`：用于 PDF 文本提取，有 pypdf；曾尝试画图失败，缺少 numpy。
- `D:/Anaconda3/envs/eegbo/python.exe`：补充实验 README 指定环境；新对话运行前检查存在性与依赖。
- `python` 曾不在 PATH，不能假设裸命令可用。

主要依赖：numpy、scipy、pandas、scikit-learn、scikit-optimize、matplotlib、PyWavelets。不要在未确认必要时重装整套环境。

## 2. 从数据到预测：实际主实验

### 2.1 数据与单位

- Bonn EEG：Z、O、N、F、S，各 100 个原始单通道片段，共 500。
- 每片段 4097 个采样点，采样率 173.61 Hz，约 23.6 秒。
- Z：健康、睁眼；O：健康、闭眼；N：非发作期、对侧海马；F：非发作期、致痫区域；S：发作活动。
- 编码顺序：Z=0、O=1、N=2、F=3、S=4。
- 论文统一叫 **segment / segment-level**。代码历史中 `recording`、`record` 指同一个原始片段，不是新的粒度。
- `window` 是从原始片段中切出的短窗口。原始片段时长固定，搜索的 `window_size`/T 是短窗口长度，两者不可混淆。

### 2.2 数据划分与验证

1. 在原始片段级分层划分 80%/20%，400 训练片段、100 测试片段，每类 80/20。
2. 使用训练片段构造固定的五折分层分组验证，原始 ID 作为组；同一片段所有窗口不得跨训练/验证折。
3. 相同 seed 的候选配置使用同一片段折分配；不同窗口长度不改变哪些片段属于哪一折。
4. `StandardScaler` 仅在每个训练折拟合，再转换对应验证折，随后训练 `SVC(kernel='rbf', class_weight='balanced')`。
5. 窗口分类后，以每片段最高票数类别得到最终片段预测。并不要求过半（严格术语更接近 plurality）；平票取最小编码类别。
6. 五个折的片段准确率做不加权算术平均 J；BO 最小化 -J。
7. 根据最高验证 J 选配置，全部外部训练数据重拟合 scaler/SVM，再评价外部测试片段。

为什么投票：每原始片段有一个标签，但产生多个窗口预测，需合并为一个预测。窗口数量随 T 改变，片段级评价使每片段贡献一次评价。不要声称投票一定优于别的聚合策略，除非引用具体比较。

Held-out 意思：在选参前单独留出的原始片段，不用于 scaler 拟合、分类器训练、验证选参；并非每个 CV 验证折的另一个称呼。

**重要限制：** 这些 outer splits 在开发中已有查看历史，且 seeds 42–46 的数据划分重叠。每个搜索运行的测试分数不返回 BO，并不意味着整个开发流程从未接触这些测试数据。不能称为全新外部确认性验证或 subject-independent 验证。

### 2.3 预处理与搜索范围

| 参数 | 代码名称 | 主实验范围 | 映射 |
|---|---|---|---|
| 下截止频率 | f_low | 0.5–5 Hz | 均匀 |
| 上截止频率 | f_high | 30–60 Hz | 均匀 |
| 窗口时长 | window_size / T | 1–5 秒 | 均匀 |
| SVM 惩罚 | C | 0.1–100 | log-uniform |
| RBF 参数 | gamma | 1e-4–1 | log-uniform |

- 完整片段先 Butterworth 带通滤波，再切窗。
- SciPy `butter` 的 order 参数为 4，SOS 表示，前向/反向滤波。不要简单断言最终双向带通的有效阶数就是四阶。
- 窗长样本数 `round(T*FS)`；不重叠，末尾不足一窗舍弃。
- 主生产流程没有新增 50 Hz notch；独立 experiment_2 有 notch，不能混写。
- 1–5 秒使每原片段约产生 4–23 个完整窗口。
- 这些是项目工作范围。目前没有证据可把精确上下界写成某篇论文规定、PSD 推导或系统预实验最优结果。
- Hsu 的 SVM guide 支持指数/对数尺度搜索，不支持恰好 C=[0.1,100]、gamma=[1e-4,1]。

### 2.4 BO 与映射

- GP / Matérn 5/2 / EI，50 配置评价，其中 16 初始化、34 后续。
- 一次配置评价=完整五折流程，不能把 50 次写成 50 个 SVM 拟合。
- 具体 GP 默认实现参数如目标标准化、噪声、EI xi，写入正式论文前从实际调用与安装库核实，不把绘图脚本的示例参数当实验设置。
- uniform / normal / exponential / gamma 四种预处理映射，每轮只采用一种，不是依次施加四种分布。
- 非均匀采用截断分布 inverse-CDF，改变 GP 坐标几何；不是单纯更换 GP 函数先验。
- 若范围为 [a,b]，normal 均值=(a+b)/2，标准差=(b-a)/4；exponential 位置=a，scale=(b-a)/3（rate=3/(b-a)）；Gamma 位置=a，shape alpha=2，scale beta=(b-a)/4。Gamma beta 明确为尺度，不是速率。
- C、SVM gamma 在所有映射比较中仍为 log-uniform；不要混淆 Gamma 分布与核参数 gamma。
- `parameter_priors.py` 使用 `CDF_EPSILON=1e-6` 做数值裁剪。
- 相同分布/种子不自动保证不同方法的初始候选完全相同；只有专门验证过的配对设计才如此描述。

## 3. 特征：55 从哪里来，怎样实现

主生产 `FEATURE_SETTING='nf_enhanced'`。`features.py` 中 `extract_features_by_setting()` 调用 `extract_psd_statistical_features()`，对每窗口提取并拼接数值，得到 n_windows × 55。

| 组别 | 个数 | 具体内容 |
|---|---:|---|
| 频谱功率 | 10 | 五频带绝对功率 + 五相对功率 |
| 功率比 | 6 | delta/theta、theta/alpha、alpha/beta、beta/gamma、(delta+theta)/(alpha+beta)、gamma/五频带总功率 |
| 统计量 | 13 | mean、std、var、max、min、median、range、RMS、line length、skewness、excess kurtosis、energy、中心化过零次数 |
| 熵 | 3 | 幅值直方图 Shannon、spectral entropy、approximate entropy |
| Hjorth | 3 | activity、mobility、complexity |
| 小波 | 20 | db4 level 3 的 A3/D3/D2/D1，各 energy、relative energy、coefficient entropy、mean、std |

细节：

- 频带 .5–4、4–8、8–13、13–30、30–40 Hz；PSD 用 Welch，`nperseg=min(256,len(window))`。
- 频带积分用半开 mask `[low,high)` 和梯形积分；相对功率分母为五个频带功率之和，不是任意全谱积分。
- Shannon 32-bin 直方图，log2；频谱熵为归一化 PSD 的 Shannon 熵，没有再除以 log2(频点数)。
- ApEn m=2、r=.2×被评价序列SD；超过80点时均匀选80点后计算，采用自然对数。80点简化不是已找到文献验证的默认设置。
- Hjorth 使用相邻差分；activity 与 var 相同。mobility 不应未经换算就标 Hz。
- 小波用 PyWavelets 默认 symmetric 延拓；DWT 不是同学论文的 WPD。
- 零分母附近用 epsilon=1e-12；输出 nan/inf 替换为0。
- 原55列零基索引15（gamma/total）基本重复索引9的相对gamma；索引32 Hjorth activity 完全重复索引18 variance。
- 因此叫“55维向量”，不是“55类特征”或“55个独立信息”。删除这两列得到53维，但生产默认尚未切换。

历史函数开关还包括 original(23维)、entropy_hjorth(29维)、expanded(41维)，这些是代码配置，不是主实验自动变更的证据。

### 特征来源的诚实表述

用户反复问“为什么恰好55、是不是网上项目现成的”。结论：有多类特征/方法文献支持，但不能证明这一具体组合最初来自某篇文献。Git 当前无提交，不能用提交史追溯。不能将事后找到的相关论文倒写为最初设计来源。

确实找到 Temko 2011 使用55特征，但其组成与本项目不同。Costa 2024 使用每通道59特征，与本项目类别接近但定义不同。EEGOpt支持多阶段优化和多类特征，不提供本项目完整55组合。所有特征恒定只能说明比较条件一致，不能证明该组合最优。

## 4. 代码地图与结果字段

| 文件 | 功能/注意 |
|---|---|
| data_loader.py | 读取 Bonn 数据 |
| preprocessing.py | 滤波、基础分窗 |
| features.py | 主55特征，其他开关 |
| grouped_bo_optimizer.py | 主分组固定折BO、候选评价、聚合、最终测试；保留旧目标模式，运行时确认 segment_accuracy |
| parameter_priors.py | 有界分布映射 |
| grouped_method_comparison.py | seed42 固定配置/随机/BO比较 |
| run_grouped_segment_priors.py | 四映射运行入口 |
| summarize_grouped_priors.py | 五种子四映射汇总 |
| plot_grouped_method_comparison.py | 主比较图 |
| plot_grouped_surrogate.py | 真实实验代理模型可视化，不等于第三章合成示意图 |
| chapter4_draft/summarise_search_histories.py | 只读已有搜索记录生成汇总曲线和审查报告 |
| analyze_grouped_features.py 等 | 特征/投票/聚合开发性分析，见各目录协议 |
| compare_feature_deduplication.py | 55 vs 53固定预处理配对候选比较 |
| compare_wavelet_levels.py | 53维三级 vs 58维四级小波 |
| compare_joint_preprocessing_bo.py | 53维联合选参 vs 固定预处理只选SVM |

**字段陷阱：** `best_validation_accuracy` / `validation_accuracy` 常指窗口级；目标要读 `best_validation_segment_accuracy`、`validation_segment_accuracy` 或确认过含义的 `score`。测试必须读取 `segment_level_test_metrics`，不要取 `window_level_test_metrics` 写进片段级结果。

## 5. 已核实的主实验结果（55维）

主目录 `results/09_fixed_fold_segment_accuracy/`；每运行应有 `grouped_cv_folds.json`、`grouped_split_manifest.json`、`grouped_bo_history.csv`、`grouped_bo_final_result.json`。

### 5.1 seed42 主方法比较：已完成

来源：`method_comparison_seed_42/grouped_method_comparison.csv`。

| 方法 | 配置评价数 | 最佳片段CV accuracy | 片段测试accuracy | seizure sensitivity |
|---|---:|---:|---:|---:|
| 固定参考 | 1 | 86.25% | 87% | 95% |
| Random Search | 50 | 88.50% | 88% | 95% |
| BO uniform | 50 | 90.25% | 90% | 95% |

固定配置：f_low=.5、f_high=40、T=2、C=10、gamma=.1。称“fixed reference configuration”，不要称文献标准或手工充分调优。

BO 最佳：f_low=.5、f_high=60、T=4.1291537280264174、C=100、gamma=.021957432418804237。

第25次达到90.25%验证，至50次未再提高；可说该次运行 best-so-far 曲线进入平台，不能说证明全局最优。三个参数触边是补充范围检查的动机，不证明外面一定更好。

该BO测试 confusion matrix 行真实、列预测，顺序 Z/O/N/F/S：

```text
19  0  0  0  1
 1 19  0  0  0
 1  0 18  1  0
 0  0  5 15  0
 0  0  1  0 19
```

recall：Z95%、O95%、N90%、F75%、S95%。主要错误之一是 F→N；说“可能与相似特征有关”需证据，不能直接断言生理因果。

### 5.2 四映射 × seeds42–46：20轮已完成

`repeated/summary_by_prior.csv` 的片段测试 accuracy：

| 映射 | 均值 | 样本SD（百分点） |
|---|---:|---:|
| uniform | 86.8% | 3.03 |
| normal | 84.8% | 2.77 |
| exponential | 84.4% | 3.58 |
| gamma | 85.0% | 2.00 |

这些是重叠数据划分上的描述统计。种子同时改变划分与搜索随机性，不能全部归为“BO自身随机波动”。不能根据这些测试分数挑映射后再将同一测试集宣称为未参与选择的确认性结果。

### 5.3 搜索历史报告：已存在且可用

`chapter4_draft/search_history_report/`：

- `mapping_validation_history_5seeds.pdf/png`：55维四映射五轮。
- `uniform_bo_validation_history_5seeds.pdf/png`：均匀BO五轮。
- `bo_vs_random_seed42.pdf/png`：仅seed42对照。
- `mapping_history_mean_sd.csv`、`mapping_results_per_seed.csv`、配对差值CSV、`protocol.json`、`paper_insert.tex`。
- 每运行先累计最大score，再跨运行算均值和sample SD(ddof=1)，阶梯曲线；阴影是SD不是95%CI。
- 协议已记录20条50次历史、同seed四映射的划分/折一致和片段隔离等审查。
- **尚无匹配的随机搜索seeds43–46；不能画成BO和random均为五轮。**

## 6. 本地已有补充实验：不要遗漏，也不要冒充主结果

这些主要根据本次磁盘README/汇总表确认；并非当前对话完整执行过。正式引用前读对应protocol和per-seed文件。

| 分支目录（主结果根下） | 已保存内容/结果 | 用途限制 |
|---|---|---|
| feature_ablation_fixed_config | 五训练划分，固定1–40Hz、5s、C10、gamma.01；original80.50%、entropy_hjorth82.85%、expanded83.05%、nf_enhanced85.45%平均CV | 固定配置训练筛查，不是分别完整BO，不是外部测试 |
| voting_tie_break_training_cv | 原投票与平票时decision-score策略；平均CV88.90→89.25%，一个seed变差 | 开发探索，无测试验证，不自动替换主投票 |
| feature_deduplication_tuned_cv | 5种子×2表示×50同一批SVM候选，55维88.40%±1.28pp，53维88.20%±1.02pp | 预处理固定；调参后的CV偏乐观；未评价test；非等价性证明 |
| wavelet_levels_tuned_cv | level3=53维88.20%±1.02pp，level4=58维87.15%±1.47pp，500候选评价完成 | 比较的是层级与维度一起变化的表示；非全搜索域最优性证明 |
| joint_vs_svm_only_53 | 5种子两策略50次/16初始：joint平均CV89.20%、test86.40%；svm_only CV88.75%、test86.00% | 53维补充，固定臂1–40Hz/5s，两臂维度不同；有先前test暴露 |

53维联合对照的 test SD：joint4.72pp，svm_only2.65pp。不能把0.4pp平均差解释成稳定、显著或因果优势。

还存在 `feature_selection_fixed_config`、`recording_aggregation_fixed_config`、`recording_mean_bo`、soft-voting等文件；本次仅确认存在，未逐项审计结果。需要时先读各协议，不猜完成状态。

`experiments/experiment_2_notch_extended_handcrafted_features/` 是独立分支：固定50Hz notch、扩展范围、扩展非小波特征。目录已有结果表，但本次未审计为最终论文证据；不可混写为当前55维主预处理。

`results/00_*`、`04_*`、`05_*`旧目标、`06_*`、`07_*`、`08_*`及`80_legacy_outputs`/`90_bo_archives`包含旧验证协议/旧目标/旧预算。根README下半段明确标Archived。不能只读其中旧“main result”就引用。

## 7. 论文当前结构与写作决定

### 第三章 Mathematical Background

已有聊天修订覆盖：

1. Optimisation Problem and Search Coordinates：theta=(f_low,f_high,T,C,gamma)，f=-J，theta=M(u)，历史D_t。
2. Gaussian Process Surrogate Models：先验、观测残差、Matérn5/2、后验均值/方差。
3. Bayesian Optimisation and Expected Improvement：顺序选点、EI定义与闭式、best-so-far B_t。
4. Classification and Validation Objective：RBF-SVM、segment-level accuracy、recall/seizure sensitivity、混淆矩阵、投票、held-out。
5. 最后新增 Mathematical Background of EEG Features：PSD绝对/相对功率、比值、统计量、熵、Hjorth、小波，**只定义与作用，不推导**。

用户修订偏好：

- 参数映射不是重点，稍微提到即可；之后又要求简述每种分布mu/sigma、lambda、Gamma alpha/beta。不要再扩成长篇推导。
- Matérn要说缘由：相对平方指数核更弱平滑假设；不能声称该平滑GP精确表达分类准确率跳变或验证过优于所有核。
- EI要说缘由：考虑改进幅度与概率，契合有限预算寻优，高斯下闭式便于计算；不声称其普遍优于所有采集函数。
- GP残差不是说相同固定配置必然每次得分不同；s_t不是CV折间SD。
- 区分SVM一对一类别投票（产生window label）和跨window多数投票（产生segment label）。
- majority voting要解释为什么：多个窗口预测→一个原始片段标签，而不只是说怎么数票。
- held-out要解释为选参前留出的测试片段，区别于CV验证折。
- 不需要EI推导、SVM完整对偶/KKT、全局收敛证明。

**已知聊天输出错误需修复：** 特征背景草稿曾误含无效 `\\begin{ation}...\\end{ation}`，后来给了正确ApEn equation替换。新对话若整理完整稿必须彻底删除无效占位，不照搬。该草稿尚未验证已集成到用户论文。

### 第四章 Implementation and Experimental Setup

仅放实现和设置，不放主结果讨论：

- 4.1 Implementation Overview：总流程图、候选→处理→验证→BO、初始化、最终重拟合/测试。
- 4.2 Data and Experimental Settings
  - Dataset and Classification Task
  - Data Partitioning and Validation
  - Preprocessing and Feature Representation
  - Search Space
  - Comparison Conditions and Evaluation Budget

用户要求保留/补充缘由：联合优化理由；固定特征保持比较一致；分层保持比例；先split后window避免同源跨集合；固定fold便于公平比较；标准化因特征尺度与RBF距离；完整片段先滤波；不重叠/丢弃尾部；窗口时间尺度；对数搜索跨数量级；uniform作为共同初始参照；随机搜索用于区分“搜索多个配置”与“利用历史反馈”；等预算与曲线用途。

这些是方法作用说明，不能包装成精确数值端点由理论推导。也不要在每一段后重复“不是最优/没有依据”，局限集中第五章。

### 第五章 Results and Discussion

用户明确将原4.3–4.6移到第五章：

- 5.1 Comparison with Baseline Methods
- 5.2 Effect of Parameter Mappings
- 5.3 Classification Error Analysis
- 5.4 Overall Discussion and Limitations

5.1–5.3都按 **目的→图表→主要发现→解释及限制** 写，不要把解释全部拖到5.4。

5.2具体映射设置随实验介绍。第三章/第四章原“映射定义见第三章/第四章”等循环引用需按最终位置修正。旧文件目录叫chapter4_draft并不代表这些结果仍应编成第四章。

本对话未完成基于全部结果的第五章英文终稿，也未确认全论文已统一编译。第一、二章及最终结论章未在本对话系统审阅。

### 指标决定（不要重复摇摆）

当前已确定正文主表accuracy+seizure sensitivity，错误分析per-class recall+confusion matrix。macro-recall在均衡test上等于accuracy，用户不希望重复；macro-F1已决定不作主报告。计算过的指标可以留CSV，无需删除或重跑。先前讨论过每类F1，但用户澄清重点是seizure sensitivity，最后版本未保留F1。

主选参目标仍J=mean CV segment accuracy。若改变为sensitivity约束/多目标/加权，需要真正改协议并重跑匹配比较，不能只改文字。

## 8. 图件与已确认的视觉要求

### 8.1 第三章 GP/EI：最终采用 markers 版

`figures/chapter3/gp_ei_selection_u_markers.pdf`（论文首选）、同名png/svg；生成器 `make_gp_ei_selection_markers.py`。

- 左：黑点观测，蓝线posterior mean mu_t(u)，浅蓝95% pointwise credible interval。
- 保留蓝色空心菱形：预测均值最小点。
- 左橙色空心圆：选中u_{t+1}处的预测均值，尚未评价。
- 右橙色实心点：EI最大点。
- 同坐标虚线联系两图；横轴统一u，不用x。
- 用图例说明，**不要箭头**。clean版错误删掉了菱形/空心圆，用户随后要求恢复；因此markers是最新。
- v2有解释箭头，已被用户否定；原u版/clean版仍留在磁盘，不要误选。
- 独立合成示意，Matérn5/2，四观测。不是实际EEG结果，不应保留旧图“adapted from scikit-optimize [sci]”引用。
- 合成脚本：mean最低u约.347，EI选u约.404；不能把这些值当实验选参结果。
- 新图PDF的LaTeX路径要与上传Overleaf位置一致；本地chapter3路径和聊天简化figures路径不是自动同步的。

另有 `gp_prior_posterior.*`旧独立GP图、`matern_prior_samples.*`新近本地文件；后者本次仅确认存在，未在本对话审阅。第三章曾决定GP/EI图足够时删除重复GP单图。

### 8.2 第四章流程图

实际文件目录 `figures/chapter4/figgenie_overview/`；有 `fig.final.pdf`、`fig.final.svg`、`fig.png`、`make_figure.py`、spec/brief/lint等。**本次实际文件名是fig.final，不要凭早期聊天名称找不存在的fig4_1文件。**

三列：左数据准备、中BO、右最终评价；英文，白背景，深色字，浅蓝BO容器。仅两种箭头：深蓝实线#365F82用于数据/评价，暗橙虚线#B86B32用于参数选择反馈。测试沿外边缘只进入最终评价，不返回BO。图内不放范围、结果、caption、公式或PPT装饰。

主关系：Bonn EEG segments→segment split→training(fixed CV)/held-out；candidate→filter/window与RBF-SVM；filter→features→standardisation/SVM→mean segment validation；majority voting标在分类到分数之间；GP/EI反馈candidate；选配置→refit all training→finaltest。

### 8.3 PSD

`chapter4_draft/training_psd/`为搜索后制作的训练集频谱观察，seed42每类80片段；Hann1024、50%overlap、constant detrend，先线性平均再log显示。

这与特征提取Welch最多256点不同。不能把事后图写为最初决定范围的依据。用户原想用PSD判断bandpass是否必要；当前类别均值PSD不能证明滤波比不滤波有效，已建议正文移除或附录观察。没有在本对话完成no-filter对照。

## 9. 参考文献与已读材料

### 用户提供并读过的两份PDF

1. `C:/Users/王枫淋/Downloads/Lu Zhimin_EGE2209334_FYP (1).pdf`：61页，数学本科FYP，SPCA/PCA与EEG。主要借鉴目的→表→发现→解释、比较条件表、用错误计数解释百分比；使用db4三级WPD而非本项目DWT。
2. `D:/HuaweiMoveData/Users/王枫淋/Desktop/1-s2.0-S0010482525013757-main.pdf`：24页，EEGOpt，Das/Chakraborty，2025，DOI10.1016/j.compbiomed.2025.111023。读过方法/特征/实验/局限重点段；不能称每页视觉审查过。

提取文本：`tmp/reference_reading/lu_reading.txt`、`article_reading.txt`。后者有NUL字符，rg用`-a`。

EEGOpt是更广的处理/特征/模型选择，TPE等多sampler、200trial、MCC目标、1s窗50%overlap；不能直接用来证明我们的50次、范围、GP最佳、55维或非重叠窗。

### 论文/官方实现链接

- Bonn原始Andrzejak2001：https://doi.org/10.1103/PhysRevE.64.061907
- EEGOpt：https://www.sciencedirect.com/science/article/pii/S0010482525013757
- Subasi2007 DWT+mixture of experts：https://doi.org/10.1016/j.eswa.2006.02.005 （并非本项目固定55组合）
- Hsu guide：https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf （支持指数尺度，不是具体端点）
- Costa2024：https://pmc.ncbi.nlm.nih.gov/articles/PMC10920642/ （59单变量特征/通道，频谱比值、统计、Hjorth、db4 D1–D5能量）
- 其代码：https://github.com/GoncaloSantosCosta/Comparison-Between-Epileptic-Seizure-Prediction-and-Forecasting
- Temko2011：https://pmc.ncbi.nlm.nih.gov/articles/PMC3036797/ ，DOI10.1016/j.clinph.2010.06.034 （55特征，但含AR误差、spectral edge、SVD entropy等，与本项目不同；它的特征筛选验证不能转移到本项目）
- 另一个研究团队实现：https://github.com/ktapani/Neonatal_Seizure_Detection ，不是本项目来源证明。
- Hjorth1970：https://pubmed.ncbi.nlm.nih.gov/4195653/ ，DOI10.1016/0013-4694(70)90143-4
- Inouye1991 spectral entropy：https://pubmed.ncbi.nlm.nih.gov/1714811/ ，DOI10.1016/0013-4694(91)90138-T
- Pincus1991 ApEn：https://pmc.ncbi.nlm.nih.gov/articles/PMC51218/ ，DOI10.1073/pnas.88.6.2297
- SciPy Welch：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- SciPy Shannon：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
- PyWavelets DWT：https://pywavelets.readthedocs.io/en/stable/ref/dwt-discrete-wavelet-transform.html
- AntroPy：https://raphaelvallat.com/antropy/generated/antropy.app_entropy.html 、同目录spectral_entropy.html、hjorth_params.html。

AntroPy仅供定义/独立实现核对；项目自己写了entropy/Hjorth，不能说代码调用AntroPy。其m2/r.2SD默认不支持本项目80点简化是标准方案。

Bib文件：根`ref.bib`与`chapter4_draft/revised_setup_20261109/references_revision.bib`。聊天新增keys包括Hjorth1970、Inouye1991、Pincus1991；复制代码后需查是否已写入最终bib，不能假设自动存在。

## 10. 论文草稿、附件与交付状态

本地 `chapter4_draft/revised_setup_20261109/chapter4_sections_4_1_4_2.tex` 是旧修订快照，早于“结果移第五章”等决定。曾做preview编译但MiKTeX权限失败，不能宣称整章成功编译。

用户最新内容通常从附件或聊天粘贴；关键附件：

- 第三章原稿：`C:/Users/王枫淋/.codex/attachments/898d9e31-117f-49dc-a94e-79c737f0d3d6/已粘贴的文本.txt`
- 第四章一次完整稿：`C:/Users/王枫淋/.codex/attachments/16d1abd2-a3aa-4e61-b9f6-b8aabdd69991/已粘贴的文本.txt`
- 较早章首/范围/全文材料目录UUID：e8fda7a9-6051-4804-9500-93697eaaa70d、ce0e56cb-d58d-46d5-b709-87186dba3124、6965465b-6b89-45ab-90e9-a87106d0bc8f。

根目录很多slides/Beamer/overleaf文件是历史演示或打包，不等于当前论文主文件。不要因为看到main.tex就默认更新它。

## 11. 用户沟通与操作偏好

- 中文讨论，英文论文LaTeX；本科数学读者，词汇朴实，术语有解释。
- 改稿常要求Part1 LaTeX、Part2中文直译并说明段落目的、Part3修改记录。科研调查/普通问答不必强行套该格式。
- 正文不用itemize、不用加粗/斜体强调、不堆砌连接词，不为了“去AI化”硬换好句子。
- 经常要求完整修改结果，别给不完整占位让用户手动拼接；不要再出现无效LaTeX环境。
- 明确保留缘由，尤其窗口范围、log尺度、uniform主比较理由。避免只列设置没有解释。
- 不反复无必要确认权限。制作图按要求直接完成并检查，再提供可点击路径；不要只给绘图建议。
- 不虚构已运行、已收敛证明、已验证最优、已读全文、导师已同意等事实。
- 当前技能清单可能变化，需要使用时验证路径。曾用academic-paper、scientific-writing、figgenie、matplotlib、pdf；不要因旧技能存在就启动多代理或全论文自动流程。
- 早期Inkscape安装曾出现ARM64包误装，后来下载correct x64并触发UAC，最终状态未确认。不是当前待办，不主动重装。

## 12. 尚待完成/需要新对话注意的事项

### 论文整理

- 取得用户当前最终第三/四章版本再统一，不能把旧稿当最新。
- 将第4/5章分工与所有交叉引用同步；4章章首不再承诺本章讨论结果。
- 整理第三章新特征背景，删除错误ation占位，补引用，不增加不必要推导。
- 确认图3.1用markers版本，图注不再写scikit-optimize adapted，不将左橙圈写成观测。
- 将第5章基线/映射/错误/总讨论写成真正基于已保存数据的英文正文；当前没有完成终稿。
- 将映射shape/scale参数放最终确定位置，避免三、四、五章相互引用却缺定义。
- 不用宏指标替换用户确定的accuracy/seizure sensitivity。

### 实验证据

- 匹配的随机搜索 seeds 43–46 已补齐，每轮 50 次训练 CV 评价。五轮 BO/RS 配对结果见 `results/09_fixed_fold_segment_accuracy/paired_bo_vs_random_seeds_42_46.csv`。固定参考仍只在 seed 42 评价。五个测试划分共覆盖 328 个不同片段，每对划分共享 16–23 个片段，配对结果限于描述性解读。
- 对触边参数的小规模训练CV范围扩展检查曾建议，**本对话未确认完成**；独立notch/扩展实验不自动等同。
- 53维去重已有单独实验，不要再说“完全没有做过去重对照”；是否用于论文由用户选择，若替主流水线需全套一致重算。
- 四级小波比较已有结果，不要重复启动相同500评价。
- 训练CV开发实验不能当独立test泛化证据。
- no-bandpass对照未在本对话确认，PSD不能替代它。

### 如需导师会议

重点问题：主线范围是否合适；55维固定表示是否需消融/重构；去重结果是否纳入；参数边界检查是否必要；重复运行数量与随机性定义；数学深度是否适当。已经有的补充实验应带上，不再把它们当纯未来建议。

## 13. 新对话最小接手流程

1. 阅读本文件的0、5、7、8节，确认用户这次要求的具体章节/任务。
2. 查看`results/README_RESULTS.md`及`chapter4_draft/search_history_report/README.md`；读根README时只采用current部分。
3. 如需数字，读取对应CSV/JSON的segment字段和protocol；确认55/53、validation/test、seed数。
4. 如需改文稿，读用户最新文本；保留设置理由和已接受结构，不擅自扩大实验范围。
5. 如需图，优先复用正确版本和真实记录，清楚区分示意图与结果图。
6. 新实验仅在任务需要时执行；不要把“交接/写作”理解为授权所有长时间重跑。

## 14. 本次handover任务完成情况

- 已核对主脚本常量、主比较表、四映射汇总、补充去重/小波/联合对照README与summary、搜索报告、图件目录。
- 已区分本对话决定和本地新增/其他任务产物，保留来源路径与限制。
- 未重跑训练、未改生产特征、未重生成实验图、未验证所有历史实验/整篇论文编译。
- 当前任务的交付就是本文件。后续更新应记录日期和哪些事实被新结果或用户决定替换。
