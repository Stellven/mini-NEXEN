# Research Plan - Finetuning of SOTA LMs

- Created at: 2026-02-17T12:20:30.030538+00:00
- Round: 2
- Readiness: refined

## Scope
- 基于“五看三定”视角下的SOTA模型微调战略:从通用能力向Agent Orchestration(多智能体编排)能力迁移
- 针对科学知识合成(Scientific Knowledge Synthesis)场景的长文本与逻辑规划微调策略(参考Storm与LiRA)
- 参数高效微调(PEFT/LoRA)在低资源环境下复现Magentic-One类复杂Agent系统的可行性研究
- Process Supervision(过程监督)在长程任务规划(Long-horizon Planning)数据构建中的应用

## Key Questions
- 在“看自己”的资源约束下,如何通过PEFT技术将GPT-4o级别的Orchestrator(编排者)能力蒸馏至7B-14B规模的开源模型？
- 针对Storm提出的“Research-then-Write”范式,应如何设计指令混合(Instruction Mixing)比例,以平衡检索利用能力与原创写作能力？
- 在多智能体系统(如Magentic-One)中,单一Agent的微调目标应侧重于“工具使用熟练度”还是“全局任务理解”,两者是否存在冲突？
- 如何构建高质量的“错误-修正”轨迹数据集,以提升模型在复杂Agent工作流中的自我反思(Self-Reflection)与鲁棒性？

## Keywords
- Finetuning of SOTA LMs, Agent Orchestration, Magentic-One, Storm, Process Supervision, Instruction Mixing, Retrieval Augmented Fine-tuning (RAFT), Parameter Efficient LLM Fine Tuning (PEFT), Scientific Knowledge Synthesis, Automated Literature Review

## Source Types in Selected Docs
- file
- web

## Source Types in Library
- file: 10
- web: 82

## Gaps and Retrieval Needs
- 缺乏专门针对多智能体“编排者(Orchestrator)”角色的高质量SFT数据集,现有数据多集中于单体Agent任务
- 对于Storm和LiRA这类长流程写作任务,缺乏对“中间产物”(如大纲,文献筛选理由)质量的量化评估标准与微调数据
- 在将SOTA闭源模型(如o1, GPT-4o)的推理过程蒸馏到小模型时,缺乏保留“长程规划”能力的有效压缩方法
- 现有的PEFT方法在处理Agentic Workflow中频繁切换的System Prompt和工具定义时,往往存在上下文适应性不足的问题

## Notes
- 利用“五看三定”中的“看行业/趋势”:Agent系统的核心正从单一模型能力转向多模型协作(Multi-Model Collaboration),微调重心需从通用对话转向特定角色的Role-Playing。
- 结合Magentic-One的发现:Orchestrator是系统的瓶颈,微调应优先尝试解决开源小模型在任务分拆与进度监控上的短板。
- 参考Storm的局限性:现有模型在生成长文大纲时容易陷入局部细节,微调数据需强化全局结构感(Global Structure Awareness)。
- “看自己”:在计算资源有限时,优先探索利用RAG生成的合成数据进行LoRA微调,以替代昂贵的全量SFT。

## Analysis Methods
- 五看三定

## Manually Added Interests
- None

## Locally Extracted Interests
Theme: AI Agents for Scientific Discovery
- Cmbagent 等系统展示了多智能体架构在科研全流程中的应用潜力,通过“创意生成者”与“创意反对者”的协作来规划研究并自动撰写论文。 Sources: Cmbagent
- 尽管自动化系统已能执行从方法论生成到报告撰写的任务,Denario 指出当前生成内容的深度更接近早期研究生水平,缺乏资深专家的宏观视野。 Sources: Denario, Cmbagent
- 随着 AI 逐步接管科研工作流,现有证据建议需要重新审视科学的哲学基础,探讨在非人类主导的生成过程中何为有效的知识与解释。 Sources: Denario

Theme: Language Agents Scientific Knowledge Synthesis
- Lira 采用基于引用的生成设计,强制要求内容锚定具体参考文献,从而有效减少幻觉并提高文献综述的事实一致性。 Sources: Lira
- Paperqa 和 Paperqa2 的评估显示,虽然智能体在处理封闭式问题时表现出色,但在没有多项选择辅助的开放式合成任务中仍面临挑战。 Sources: Paperqa, Paperqa2
- 为了应对科学知识的快速更新,相关系统致力于通过检索增强生成来弥补预训练模型的知识截止限制,但错误信息的传播风险依然是核心难题。 Sources: Paperqa, Paperqa2

Theme: Computational Astrophysics and Machine Learning
- Denario 集成了随机森林回归和 SHAP 值等先进机器学习技术,用于映射宇宙学参数与反馈机制之间的复杂依赖关系。 Sources: Denario
- 计算模拟结果显示,虽然大质量星系的演化与观测数据吻合,但在矮星系区域预测出了比观测结果更高的多样性。 Sources: Denario

Theme: Protein and peptide molecular dynamics
- 研究利用扩散图和聚类算法成功将构象空间划分为折叠,未折叠及中间态等离散的亚稳态。 Sources: Denario
- 基于图论的方法被应用于分析氨基酸网络,通过计算拉普拉斯特征值和节点中心性来表征结构属性。 Sources: Denario

Theme: AI Co-scientist for Biomedical Discovery
- Ai-Co-Scientist 采用“生成,辩论,进化”的机制,通过锦标赛式的反馈循环不断优化研究假设的质量。 Sources: Ai-Co-Scientist
- 该系统通过自然语言交互让研究人员设定目标并提供人工审查,确保自动生成的提案与科学约束和实际需求保持一致。 Sources: Ai-Co-Scientist

Theme: Parameter Efficient LLM Fine Tuning
- 参考文献列表显示,该领域的研究高度依赖于 AlphaFold 等蛋白质结构预测模型以及关于基础模型机遇与风险的先验工作。 Sources: Ai-Co-Scientist, Denario
- 引用的文献表明,当前趋势是将大型语言模型与自主化学研究及自动化系统综述相结合,以提升科研效率。 Sources: Denario, Lira


## Research Plan
1. 五看三定之一:看趋势——分析SOTA模型从通用对话向多智能体协同演进的技术路径
   - 调研Magentic-One与STORM等系统架构,分析大模型应用模式如何从单一的Prompt Engineering转向复杂的Agentic Orchestration(智能体编排)。
   通过对比单体模型与多智能体系统在GAIA等基准测试上的表现,论证引入Orchestrator(协调者)和Specialized Agents(WebSurfer, FileSurfer等)进行分工协作的必要性,指出通用SOTA模型在未经特定微调时处理复杂长流程任务的局限性。
   重点考察学术界在“Process Supervision”(过程监督)与“Outcome Supervision”(结果监督)上的研究趋势,结合Monte Carlo Tree Search (MCTS) 增强推理能力的最新进展,评估将推理过程数据用于微调SOTA模型以提升Agent规划能力的潜力。
   - 追踪自动化科学发现(Automated Scientific Discovery)领域的最新进展,深入分析AI Scientist-V2,Ai-Co-Scientist及LiRA (Literature Review Agents) 的工作流设计。
   探讨如何利用大模型自动化生成文献综述,提出科学假设及执行代码验证,识别出当前SOTA模型在长上下文理解,跨文档知识综合以及减少幻觉方面的技术瓶颈,从而确定微调的高价值切入点。
   分析PaperQA2等系统如何通过RAG(检索增强生成)与Agent结合来提升引文准确性,研判“Retrieval Augmented Fine-tuning (RAFT)”作为微调趋势的重要性。
   - 评估微调技术(Fine-tuning)与上下文学习(In-Context Learning)的融合趋势,特别是在处理Out-of-Distribution(分布外)任务时的表现。
   依据“Demystifying Instruction Mixing”和“MAC-Tuning”等文献,分析混合指令微调(Instruction Mixing)对保持模型通用能力与提升特定领域推理能力的平衡作用,预测未来SOTA模型微调将更侧重于通过多任务混合数据来增强模型的组合推理(Compositional Reasoning)能力。
   研究参数高效微调(PEFT)技术在多智能体部署中的应用前景,特别是针对不同Agent角色(如Coder, Reviewer)快速切换LoRA适配器的可行性分析。
   - 深入剖析多模态能力在Agent系统中的集成趋势。
     - 参考Magentic-One集成GPT-4o处理视觉信息的案例,分析多模态输入(截图,图表)对WebSurfer等Web导航Agent的重要性。
     - 调研Audio和VideoSurfer等未来扩展方向,评估当前SOTA模型在多模态微调方面的数据需求与计算成本。
2. 五看三定之二:看市场/客户——识别自动化科研与复杂任务处理中的痛点与需求
   - 细分科学研究与企业级知识管理的用户场景,界定研究人员对自动化文献综述工具(如LiRA, STORM)的核心需求指标。
   通过分析PaperQA2和DeepReview的应用案例,明确用户对于“引用溯源准确性”,“多源信息冲突解决”以及“长篇逻辑连贯性”的极高要求,指出当前通用SOTA模型在零样本(Zero-shot)状态下难以满足这些专业标准的现状。
   识别企业在部署Agent系统时的成本敏感度与延迟容忍度,分析用户对于本地化微调模型(Local LLMs)以替代昂贵闭源API(如GPT-4o)的潜在需求。
   - 调研开发者在使用Agent框架(如AutoGen, LangGraph)时的实际痛点,特别是在多智能体协作稳定性方面的反馈。
   参考Denario系统中的人机协作模式,分析用户对于“Human-in-the-loop”(人在回路)交互机制的依赖,探讨如何通过微调使模型更好地理解人类反馈信号并自适应调整规划路径。
   挖掘用户对特定领域(如生物医药,材料科学)深度推理能力的渴求,验证通过微调注入领域知识(Domain Knowledge)的市场价值。
   - 分析复杂任务自动化(如WebArena, AssistantBench)中的失败案例,归纳用户对Agent执行鲁棒性的具体期望。
   针对Magentic-One在处理真实网页浏览和文件操作时的错误模式,总结出用户最关心的几类“任务终止”原因,如死循环,工具调用错误等,将其转化为微调目标的具体维度。
   确立以“提高任务完成率(Success Rate)”和“减少人工干预频率”为核心的市场导向优化指标。
3. 五看三定之三:看竞争——对标Magentic-One,STORM及其他SOTA Agent系统的核心竞争力
   - 深度解构Magentic-One的“Orchestrator-Worker”架构,分析其GroupChat机制与AutoGen框架的结合优势。
   对比Magentic-One利用GPT-4o作为核心驱动力与使用开源模型(如Llama 3)进行微调后的潜在性能差异,识别专有微调模型在特定Agent角色(如专门优化过的FileSurfer)上超越通用SOTA模型的可能性。
   分析Magentic-One在GAIA基准测试上取得优异成绩的关键因素,剥离出其Prompt工程与底层模型能力的具体贡献。
   - 对比STORM与传统RAG系统在长文本生成任务上的差异,重点考察STORM的“Perspective-taking”(视角选取)与“Question Generation”(提问生成)模块。
   评估STORM通过模拟对话构建大纲的方法论优劣,对比直接微调模型以生成大纲(Direct Gen)与基于检索增强的大纲生成策略(RAG-expand)的效果差异,找出竞争对手在知识综合深度上的护城河。
   分析DeepReview和Ai-Scientist在同行评审与假设生成方面的特长,建立功能对标矩阵。
   - 考察学术界与工业界在“Retrieval Augmented Fine-tuning (RAFT)”技术上的竞争态势。
   分析现有竞品如何处理检索到的干扰信息(Distractors),参考“Enhancing Code Transformation... Through RAFT”等论文,对比不同微调策略在抑制幻觉和提升检索利用率方面的效果。
   识别竞争对手在数据构建流水线(Data Pipeline)上的优势,特别是他们如何利用合成数据(Synthetic Data)来弥补高质量训练数据不足的问题。
   - 分析多智能体编排框架的生态竞争。
     - 对比LangGraph在Denario中的应用与AutoGen在Magentic-One中的应用,评估不同编排逻辑对模型指令遵循能力的要求。
     - 研究Competitors如何利用Process Supervision数据来优化Orchestrator的决策逻辑,从而降低对昂贵推理模型(如o1)的依赖。
4. 五看三定之四:看自己——评估当前微调SOTA模型的技术储备与资源瓶颈
   - 盘点现有算力资源与SOTA模型(如Llama 3, Qwen 2.5等)的微调适配性,进行全参数微调与PEFT(LoRA/QLoRA)的成本效益分析。
   依据“A Comparison of LLM Finetuning Methods... with Travel Chatbot Use Case”中的评估方法,自我诊断在构建高质量指令微调数据集(Instruction Tuning Datasets)方面的能力短板,特别是缺乏针对Agent工具调用(Tool Calling)和多轮对话状态跟踪的标注数据。
   评估团队在实施“Process Supervision”方面的数据生成能力,即是否具备利用MCTS或类似算法自动生成高质量推理步骤数据的技术栈。
   - 审查内部在RAG与微调结合(RAG + Fine-tuning)方面的技术积累,确认是否掌握RAFT等先进微调方法的核心实现细节。
   通过复现PaperQA2或STORM的部分模块,测试当前基座模型在处理长上下文(Long Context)和多文档冲突时的真实表现,量化与GPT-4o等闭源SOTA模型之间的“智能差距”。
   识别在Agent编排逻辑开发上的经验缺口,特别是在设计高效的Orchestrator Prompt和错误恢复机制(Error Recovery)方面的能力不足。
   - 评估数据清洗与合成流水线的效能,特别是针对科学文献(PDF解析,公式提取)的处理能力。
   参考Magentic-One对多模态数据的处理需求,检查自身是否具备构建多模态指令微调数据的工具链,以及在训练过程中混合多模态数据以防止灾难性遗忘(Catastrophic Forgetting)的策略储备。
   明确在建立自动化评估体系(如基于LLM的E2E评估)方面的差距,判断是否过度依赖人工评估。
   - 审视模型部署与推理优化的技术栈。
     - 评估在支持多Agent并发推理时的系统吞吐量限制。
     - 检查是否掌握动态加载LoRA适配器以支持不同Agent角色(Coder vs. Reviewer)快速切换的技术。
5. 五看三定之五:看机会——发掘基于混合指令微调与过程监督的高价值突破口
   - 锁定“Agent-Centric Fine-tuning”作为核心战略机会,通过构建包含工具使用,规划,反思(Reflection)等多维度能力的混合数据集来微调SOTA模型。
   利用“Instruction Mixing”策略,将通用对话数据,代码生成数据与特定Agent任务数据(如Web浏览日志)按最优比例混合,旨在训练出比肩GPT-4o但在特定Agent任务上更高效,成本更低的开源替代模型。
   探索利用过程监督(Process Supervision)数据来强化模型的自我纠错能力,使其在长链条任务中能够自主识别并修正中间步骤的错误。
   - 抓住“Scientific Knowledge Synthesis”垂直领域的微调机会,打造专注于科学文献深度理解与生成的专家模型。
   结合STORM的大纲生成逻辑与LiRA的综述写作流程,开发一套端到端的微调方案,专门优化模型对学术语体,引用规范及跨学科概念连接的掌握能力。
   利用RAFT技术解决领域知识更新滞后的问题,训练模型在面对检索到的新知识时能够准确整合而非产生幻觉,填补通用模型在专业科研辅助场景下的空白。
   - 发掘将“System 2 Reasoning”(慢思考)能力蒸馏到小参数模型中的机会。
   借鉴Magentic-One中使用o1模型进行推理增强的思路,利用高性能闭源模型生成思维链(CoT)数据,通过知识蒸馏(Knowledge Distillation)微调较小的SOTA模型(如7B-14B参数),使其具备更强的逻辑规划能力,从而在边缘侧或低成本环境下实现复杂的Agent功能。
   探索多智能体协作数据的合成与利用,让模型学习如何作为团队一员(Team Player)进行有效沟通与协作。
6. 五看三定之六:定战略控制点——构建专有的Agent能力数据集与RAFT微调流水线
   - 构建壁垒级的高质量Agent指令数据集(Agent-Instruct Dataset),该数据集应涵盖从STORM,PaperQA等系统衍生出的复杂任务规划轨迹。
   重点采集和合成“过程监督”数据,即不仅包含问题和答案,还包含详细的推理步骤,工具调用参数及环境反馈(Environment Feedback),以此作为提升模型规划能力的核心资产。
   建立自动化的数据合成引擎,利用MCTS等算法在仿真环境中探索最优路径,源源不断地生成高质量的训练样本,形成数据飞轮效应。
   - 打造标准化的RAFT(Retrieval Augmented Fine-tuning)训练流水线,确保模型能够深度适配RAG架构。
   开发专有的“干扰项注入”机制,在微调数据中刻意混入不相关文档,训练模型识别并忽略噪声的能力,从而确立在复杂信息检索场景下的性能优势。
   固化“Instruction Mixing”的配比策略,通过大量实验沉淀出针对不同基座模型(Base Model)的最佳数据混合方案,作为团队的核心技术Know-how。
   - 确立以“角色特化微调”(Role-Specific Fine-tuning)为技术制高点,针对Magentic-One架构中的不同Agent(WebSurfer, Coder, Orchestrator)训练专用的LoRA适配器。
   开发一套动态权重加载系统,使得单一大模型底座能够根据当前任务上下文,瞬间切换至最适合的Agent角色模式,实现资源利用率与任务处理能力的最优化平衡。
   建立基于真实科学文献的持续预训练(Continuous Pre-training)机制,保持模型在科学领域的知识鲜活度。
7. 五看三定之七:定目标——设定基于GAIA与AssistantBench的量化性能指标
   - 设定具体的SOTA对标基准:微调后的模型在GAIA(General AI Assistants benchmark)测试集上的成功率应达到或超过GPT-4o基线的80%水平。
   针对科学文献处理任务,要求在PaperQA2定义的引用准确率(Citation Precision)上达到95%以上,且在AssistantBench的复杂推理题上显著优于未微调的基座模型。
   明确“幻觉率降低”的量化目标,要求在生成长篇文献综述时,事实性错误(Factual Errors)的发生率降低至商业可用水平(如每千字少于1处)。
   - 制定工程化性能指标,确保微调后的Agent系统具备实际部署价值。
   设定推理延迟(Latency)与吞吐量(Throughput)的优化目标,例如在并发处理10个Agent交互时,平均响应时间需控制在用户可接受范围内。
   确立“自动化闭环”目标,即在STORM类文章生成任务中,无需人工干预生成的文章结构完整性(Structural Integrity)达到90%以上,大幅减少人工编辑的工作量。
   - 定义微调效率目标,验证PEFT策略的有效性。
   目标是在使用不超过全量参数5%的可训练参数情况下,实现与全参数微调相当的任务性能(性能损失<2%)。
   设定数据利用率目标,要求通过Instruction Mixing和RAFT技术,仅使用10%的领域特定数据即可实现显著的领域适应能力,验证小样本学习(Few-shot Learning)能力的提升。
8. 五看三定之八:定策略——实施多阶段指令混合与过程监督数据构建策略
   - 执行分层级的数据构建策略:首先收集通用SOTA模型的对话数据以保持语言能力,其次通过运行Magentic-One和STORM在沙箱环境中生成大量Agent交互日志。
   利用“Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search”中的方法,设计专门的奖励模型(Reward Model)来评估推理步骤的质量,筛选出高价值的思维链(CoT)路径作为微调核心数据。
   针对代码生成与工具使用任务,构建包含“错误-修正”对(Error-Correction Pairs)的数据集,让模型学习如何根据解释器反馈修复代码。
   - 实施严格的“Instruction Mixing”配比实验,参考“Demystifying Instruction Mixing”的研究结论,设定如50%通用指令,30%领域特定指令(如科学文献),20% Agent交互指令的初始混合比例。
   在数据预处理阶段,应用RAFT策略,为每个训练样本构建包含“正例文档”,“负例干扰文档”和“思维链解释”的复杂Prompt结构,强制模型在微调过程中学习依赖上下文而非仅靠参数记忆回答问题。
   开发自动化数据增强脚本,将单轮QA数据改写为多轮对话和任务执行轨迹,模拟真实世界的多步操作场景。
   - 建立动态的数据迭代机制(Data Flywheel)。
   在模型初步微调后,将其部署到WebArena等仿真环境中进行压力测试,自动收集失败案例(Failure Cases)。
   利用更强的模型(如GPT-4o或o1)对失败案例进行分析和重写(Re-annotation),将修正后的轨迹重新加入训练集,实现模型能力的螺旋式上升。
   - 针对科学写作场景的特殊数据策略。
     - 构建基于PaperQA2标准的引用预测数据集,训练模型精准定位引用片段。
     - 合成跨文档冲突解决样本,训练Orchestrator在面对矛盾信息时的裁决能力。
9. 五看三定之九:定策略——采用RAFT与PEFT结合的混合微调训练方案
   - 采用全量参数微调(Full Fine-tuning)与参数高效微调(PEFT)相结合的混合策略。
   对于通用的Orchestrator模型,在高性能计算集群上进行全参数指令微调,以确立其强大的规划与指令遵循基座能力;对于WebSurfer,FileSurfer等专用Agent,采用LoRA或QLoRA技术进行轻量级适配训练,便于灵活部署和迭代。
   严格遵循RAFT(Retrieval Augmented Fine-tuning)的训练范式,在训练损失函数中增加对引用准确性的惩罚项,强制模型关注检索上下文。
   - 引入“Curriculum Learning”(课程学习)策略,优化训练过程。
   先使用简单的单步任务数据进行热身训练,逐渐过渡到包含多分支,长视距(Long-horizon)的复杂Agent协同任务数据,防止模型在面对复杂指令时出现崩溃。
   在微调过程中集成“Process Supervision”信号,如果条件允许,探索使用Outcome-supervised Reward Models (ORM) 和 Process-supervised Reward Models (PRM) 进行强化学习微调(RLHF/RLAIF),进一步对齐人类偏好。
   - 实施严格的模型评估与监控(Evaluation & Monitoring)策略。
   在训练过程中每隔固定Step在AssistantBench验证集上进行Zero-shot测试,及时发现过拟合或灾难性遗忘现象。
   对比分析不同Base Model(如Llama 3, Mistral, Qwen)在相同微调数据下的表现,选择最适合Agent任务的底座模型进行后续优化。
   - 针对长上下文能力的专项微调。
     - 使用Ring Attention或LongLoRA等技术扩展模型的上下文窗口。
     - 在微调数据中包含超过32k token的长文档摘要与问答任务,确保模型在处理STORM生成的长篇大纲时不会丢失上下文信息。
10. 五看三定之十:定策略——优化多智能体编排与系统集成架构
   - 基于Magentic-One的参考架构,重新设计适合微调模型的Orchestrator Prompt与状态机(State Machine)。
   针对微调后的模型特性,优化GroupChat流程中的发言轮次控制(Turn-taking)机制,减少无效的客套对话,提高信息交换密度。
   集成LangGraph或类似框架,将微调后的模型封装为可复用的Agent节点,实现工作流的可视化编排与调试。
   - 开发特定领域的工具集(Tool Set)并与微调模型进行深度对齐。
   为FileSurfer配备专门优化的PDF解析与数据清洗工具,为WebSurfer配备无头浏览器接口,并在微调阶段强化模型对这些工具API的调用准确率。
   实现基于PaperQA2算法的RAG检索模块,将其作为标准工具提供给Agent,确保所有外部知识获取均经过严格的检索与重排序(Re-ranking)过程。
   - 构建“Human-in-the-loop”的交互接口,参考Denario的设计理念。
   在Orchestrator的关键决策节点(如大纲确认,高风险操作)引入人工确认步骤,并设计专门的反馈收集UI,将用户的修改意见实时转化为自然语言指令反馈给Agent。
   设计异常监控与熔断机制,当检测到Agent陷入死循环或产生大量错误操作时,自动终止任务并生成诊断报告。
   - 部署与推理优化。
     - 利用vLLM等推理框架优化多Agent并发请求的吞吐量。
     - 探索Speculative Decoding(投机采样)技术,利用小模型辅助大模型加速推理,降低Agent系统的端到端延迟。

## Next Actions
- Ingest missing sources for the identified gaps.
- Run another planning round after updating the library.