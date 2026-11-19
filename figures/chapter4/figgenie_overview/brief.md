# 图形简报

本图用于本科数学论文第四章 Implementation Overview，作为 A4 正文通栏总览图。以用户明确强调的“总览图保留主要模块和反馈关系就够了”为内容层级，保留长描述中的训练选择与测试隔离语义，不将参数映射、交叉验证内部步骤及设置说明塞入总览图。用户已给出模块、反馈、配色和用途，按该具体要求直接制作可审阅初稿。

左侧为 Bonn EEG segments (five classes)、Segment-level data split、Training segments (fixed cross-validation)、Held-out test segments。测试集分支沿左外侧从划分框进入，避免穿过训练数据到预处理的箭头；测试数据再沿底部外侧进入最终测试框。

中央 Bayesian optimisation 容器含 Candidate parameters、Filtering and windowing、Feature extraction、Standardisation and RBF-SVM、Mean segment-level validation accuracy、GP update and EI 六个模块。实线蓝色表示数据处理与评价；虚线橙色表示参数选择与反馈。候选参数分别通过 Filter cutoffs and window length、C and γ 两条虚线箭头控制预处理和分类器。处理链经过特征提取、标准化及 RBF-SVM 到验证准确率；分类器到验证准确率的箭头标为 Majority voting。验证结果经 GP update and EI 反馈到 Candidate parameters；验证得分达到预算后，通过标注 After budget exhaustion 的控制箭头进入选定配置。

右侧为 Selected configuration、Refit on all training data、Final test evaluation，三个框纵向间距收紧。验证结果通向配置选择；测试数据沿图底部进入 Final test evaluation，不连向优化器。最终测试使用所选配置及训练数据拟合的标准化器，由正文和图注解释，不再增加模块。

图形要点：BO 只依据训练 segments 上的交叉验证反馈选择配置，保留测试 segments 用于最终评价。采用 systems / architecture / control-loop 的视觉结构作参考，而不套用 AI 会议 Figure 1 的多图块结构。参考 asplos26-080_fig5 的紧凑回路、asplos26-132_fig6 的一次性步骤与循环分离、nsdi24-043_fig1 的阶段分组；检索所得另外两张机制图只用于比较，未借用其复杂内容。没有来自 Bonn EEG 同领域且完全同结构的 exemplar，故按本图语义自定义列布局。

已明确的编辑假设：只交付简洁总览图；参数范围与 CV 细节留正文；所有标签英文；只使用实线蓝色和虚线橙色两种箭头样式，并在图下加简短图例；SVG/PDF/PNG，不制作 PPTX。

图注草稿：Implementation workflow of the Bayesian optimisation-controlled EEG classification pipeline. Solid blue arrows denote data processing and evaluation; dashed orange arrows denote parameter selection and feedback. The held-out test segments are used only for final evaluation.
