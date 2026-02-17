---
name: systems-engineering
description: >
  Use this skill for any systems engineering task: requirements analysis,
  architecture design, MBSE modeling, V-model workflows, interface control
  documents, trade studies, verification & validation planning, risk management,
  FMEA/FMECA, system decomposition, and lifecycle documentation. Also covers
  Huawei systems engineering methods including "三一工程", "五看三定" (FSTD),
  "作战地图", DSMM, and SBIIEM investment evaluation. Triggers include mentions of
  "system design", "requirements", "architecture", "V&V", "MBSE", "ICD",
  "trade study", "FMEA", "ConOps", "system integration", "DoDAF/SysML",
  "三一工程", "五看三定", "四化设计", "系统融合", "作战地图", or "产业投资".
---

# Systems Engineering Skill

## Purpose

This skill guides Claude to produce rigorous, professional systems engineering
artifacts aligned with industry standards: **INCOSE SE Handbook**, **ISO/IEC/IEEE 15288**,
**NASA Systems Engineering Handbook (SP-2016-6105)**, **DoD MIL-STD-882E**, and
**MBSE best practices (SysML/UAF)**.

This skill also incorporates **华为系统工程方法论**（Huawei SE Methods），including:
- **三一工程** — System-level integration to achieve first-class products using constrained components
- **五看三定 (FSTD)** — Strategic analysis framework (5 observations + 3 definitions)
- **作战地图** — Battlefield map methodology for goal-driven resource allocation
- **DSMM** — Domain expert system mind-map modeling for knowledge capture
- **SBIIEM** — System-based industrial investment evaluation methodology

When the user context involves product competitiveness under supply constraints, strategic planning, or industrial investment decisions, apply the relevant Huawei method alongside standard SE processes.

---

## Core Principles

1. **Requirement-driven** — Every design decision traces to a requirement.
2. **Lifecycle thinking** — Consider all phases: Concept → Development → Production → Utilization → Support → Retirement.
3. **Interface-aware** — Systems interact. Always identify and document interfaces.
4. **Verification-first** — For every requirement, define how it will be verified (Test, Analysis, Inspection, Demonstration — TAID).
5. **Risk-informed** — Surface risks early; quantify and mitigate.

---

---

## Huawei Systems Engineering Methods (华为系统工程方法)

> Apply these methods when the user's context involves product strategy, competitive engineering under constraints, or technology investment planning.

---

### Method 1: 三一工程 (3-to-1 Engineering)

**适用场景 / When to apply:** Product design under component supply constraints; achieving competitive performance with "second-tier" components.

**核心理念 / Core Concept:**
- **"三做一"(3→1 Integration):** Use widely available, constrained-generation components (≥15% performance gap vs. best-in-class) and through system-level integration achieve first-class product competitiveness.
- **"三变一"(3→1 Transformation):** Upgrade the supply chain itself — through materials, process, architecture, and algorithm improvements, turn second-tier components into first-tier.

**十六字诀 / 16-Character Formula:**
| 字诀 | 含义 | SE Application |
|---|---|---|
| 需求精准 | Precise requirement definition | Apply 80/20 rule; remove non-critical specs to reduce over-design |
| 举一反三 | Plan A/B-1/B-2 concurrently | Architecture must support both premium and constrained component paths |
| 四化设计 | Platform/Module/Unified/Interchangeable | See "四化" design principles below |
| 认知精准 | Dynamic supply-state awareness | Treat supply status as a variable in architecture decisions |

**四化设计原则 / 4-化 Design Principles:**
1. **平台化 (Platformization):** One hardware/software platform supports N-1 → N+1 product generations
2. **模块化 (Modularization):** Standardized internal interfaces; each module independently developable; PIN-to-PIN compatible across suppliers
3. **归一化 (Unification):** Single component/chip selection per function across all products; hardware CBB (Common Building Block) strategy
4. **互置化 (Interchangeability):** Achieve equivalent system output through different technology paths — e.g., software compensates for hardware limitations (软补硬), algorithms substitute for components (算补芯)

**五系方针 / 5-Series Directives:**
- 系统技术规划与技术储备
- 系统定义目标要求（正向）
- 系统设计（正向）/ 跨系统融合设计
- 系统架构优化（逆向）/ 集中式分布式互替
- 系统分解并根技术/基础理论突破

**关键规则 / Key Rules:**
- Architecture must decouple the **functional layer** (stable) from the **implementation/physical layer** (variable)
- Never design at the implementation layer in isolation — always elevate one level
- System optimization target is **E2E optimal**, not single-point optimal
- Avoid repeated design margins for the same parameter across multiple design stages (防过设计)

**系统融合设计方法 / System Fusion Design:**
- **重新分配指标:** Redistribute specs across subsystems to compensate for weak components
- **系统寻优:** Recombine circuits/arrays/functions to find global optimum (三种方式: 重新组合、软补硬、彻底重构)
- **系统联动:** E2E cross-domain collaboration for coordinated optimization

---

### Method 2: 五看三定 (FSTD — Five Steps to Analyze, Three Targets to Define)

**适用场景 / When to apply:** Strategic planning, technology roadmap creation, competitive positioning, business/product portfolio decisions.

**五看 (Five Observations):**

| # | 维度 | 分析内容 |
|---|---|---|
| 1 | 看行业/趋势 | Industry value migration; disruptive technologies; regulatory changes; new business models |
| 2 | 看市场/客户 | Market size & growth; customer pain points; decision-making structure; buyer personas and KPIs |
| 3 | 看竞争 | 18 competitor intelligence factors: profit, share, product roadmap, quality, pricing, partnerships, IP, culture... |
| 4 | 看自己 | Business model canvas (10 elements); SWOT; internal capability gaps |
| 5 | 看机会 | Blue ocean segments evaluated on: 重要性, 持久性, 独特性, 可衡量性, 可识别性 |

**研发体系"五看"变体 / R&D Variant:**
| # | 维度 |
|---|---|
| 1 | 看趋势 — 技术成熟度, TRL分析, 替代方案 |
| 2 | 看市场 — 产业链格局, 市场空间 |
| 3 | 看客户 — 痛点/爽点/痒点, 可持续商业模式 |
| 4 | 看竞争 — 对手战略控制点, 产品演进节奏 |
| 5 | 看自己 — 差距分析, SWOT |

**三定 (Three Definitions):**

| # | 定义 | 内容 |
|---|---|---|
| 1 | 定战略控制点 | Sustainable competitive moat (成本优势→性能领先→品牌→市场份额→标准/专利组合) |
| 2 | 定目标 | Set dual targets: 达标目标 (baseline, top-down) + 挑战目标 (stretch, bottom-up) |
| 3 | 定策略 | Specific tactics: market, technology, quality, cost, delivery — must form a coherent winning system |

**SE Integration Note:** Use 五看三定 to drive the **Concept of Operations (ConOps)** and **System Requirements** phases. The "五看" maps directly to stakeholder analysis and operational environment definition.

---

### Method 3: 作战地图 (Battlefield Map)

**适用场景 / When to apply:** Program planning, resource allocation, prioritization, project execution under complex multi-stakeholder conditions.

**四步骤 / Four Steps:**

```
作战目标 → [看清战场] → [战略聚焦] → [合理用兵] → 作战结果
(Goal)     (Situational   (Strategic    (Resource      (Outcome)
            Awareness)    Focus)        Deployment)
                    ↑__________________________|
                         复盘迭代 (AAR Review)
```

1. **看清战场 (Situational Awareness)**
   - 宏观: macroeconomic trends
   - 中观: industry and supply chain dynamics
   - 微观: customer, competitor, technology insights
   - Tool: Apply 五看 framework

2. **战略聚焦 (Strategic Focus)**
   - Align to company/product/department strategy
   - Score and weight key factors (国家/客户/产业/伙伴)
   - Rule: "有所为有所不为" — boldly abandon low-ROI opportunities ("石头")

3. **合理用兵 (Resource Deployment)**
   - Match resources to opportunity tiers: 肥肉(saturate) / 瘦肉(minimal) / 骨头(elite force) / 石头(skip)
   - Dynamic adjustment triggers: space/revenue/headcount/profit-per-head changes

4. **复盘迭代 / AAR (After Action Review)**
   - Based on US Army AAR structure:
     - What was supposed to happen? (作战目标)
     - What actually happened? (作战结果)
     - Why the difference? (原因分析)
     - How to improve next time? (迭代优化 — concrete Action Items with owners)

**三张地图 / Three Maps (R&D variant):**
- 技术要素地图: root technology + key technology competitiveness map with layout status (已掌握/正在布局/计划布局/暂未涉及)
- 合作资源地图: external collaboration landscape (未交流/已交流/曾合作/正合作)
- 人才梯队地图: internal + external talent pipeline (领军/中坚/高潜)

---

### Method 4: DSMM — 领域专家系统思维建模

**适用场景 / When to apply:** Knowledge management, expert knowledge capture, building reusable decision frameworks within an organization.

**Core Concept:** Externalize expert tacit knowledge into a structured, learnable system model. Core principle: "萃取智慧，固化方法"

**六步骤 / Six Steps:**
1. **确定领域** — Identify the domain worth modeling (high expert-dependency, long-term value)
2. **选定专家** — Select high-performers with proven decision track record
3. **萃取思维结构** — Interview with structured questions:
   - 你最关心的Top3决策问题是什么？
   - 你的底层动力是什么？
   - 解决这些问题需考虑哪些方面？
   - 哪些过去经历影响了你的决策？
   - 未来哪些因素会影响决策？
4. **模型校验** — Cross-validate using "多方收集，相互印证"; assess: 整体性, 关键要素, 要素关系, 系统环境, 上层目的
5. **构建领域模型** — Aggregate across multiple expert models; use Zachman Framework for multi-role domains
6. **持续迭代与优化** — Models have domain and time scope; re-validate as environment changes

**思维建模系统结构 / System Structure:**
```
[环境/Business Context]
        ↓
[目标/Goal] ← 职责, 使命, 愿景
        ↓
[要素/Key Factors] — 技术, 人才, 商业, 供应...
        ↓
[关系/Relationships] — 协同, 优化, 时空依赖
```

---

### Method 5: SBIIEM — 产业投资系统论证方法

**适用场景 / When to apply:** Evaluating entry into new industries, identifying incremental opportunities in existing industries, M&A / strategic investment decisions.

**Core Question:** "有没有饭吃、怎么吃饭、吃哪一碗饭" (Is there an opportunity? How to capture it? Which slice to take?)

**两阶段七步骤 / Two Phases, Seven Steps:**

**Phase 1 — 战略选题 (Strategic Topic Selection):**

| Step | 内容 |
|---|---|
| 1. 产业机会初步研判 | Scan 5 dimensions: 技术/能力, 市场/需求, 产业链/结构, 政策/制度, 财务/资本 |
| 2. 论证目标与范围聚焦 | Apply 3 principles: 战略相关性, 问题可论证性, 价值密度 |
| 3. 差异化论证视角构建 | Identify research gaps; reframe from Huawei capability lens; focus on core contradictions |
| 4. 分析框架搭建 | Decompose industry into interrelated key factors; ensure logical closure + verifiability |

**Phase 2 — 系统论证 (Systematic Validation):**

| Step | 内容 |
|---|---|
| 5. 战略假设提出与专家访谈 | Formulate testable hypotheses; expert interviews + secondary research cross-validation |
| 6. 系统建模与数据分析 | Causal models, comparative models, scenario models; quantitative + qualitative cross-validation |
| 7. 多维度研讨迭代 | Multi-round review with business, customer, external expert perspectives; identify blind spots |

**五维度初步研判框架 / 5-Dimension Initial Assessment:**
```
技术/能力 ──── TRL分析, 技术壁垒, 路线可持续性
市场/需求 ──── 市场规模, 需求驱动机制, 竞争格局
产业链   ──── 分工格局, 利润分布, 卡点分析
政策/制度 ──── 政策依赖度, 监管不确定性
财务/资本 ──── 盈利模型, 资本密集度, 退出路径
```

---

### Huawei SE Case Study Reference: 543-Mate60 Project

**核心理念 / Core System Engineering Principles Applied:**
1. **客户体验价值牵引** — User experience as the single measurable goal; objectify subjective UX; engineer it quantitatively
2. **STCO (System Technology Re-balancing)** — Break system boundaries; software compensates for hardware (软补硬); algorithms compensate for chips (算补芯); cloud assists edge (云助端); non-Moore compensates for Moore
3. **全局最优 / Global Optimum** — System-level modeling; quantitative trade-offs; Top-down decomposition + Bottom-up verification
4. **复利法则 / Compound Progress** — Fast POC validation; short cycles; accumulate small wins; avoid "big bang" approaches

**可借鉴的工程实践 / Reusable Engineering Practices:**
- 打破部门墙: Cross-functional "混编旅" with single unified goal
- 科学建模: Use quantitative models (human perception models, power/thermal models) as the project-wide shared language — "不做语文题，只做数学题"
- 设计与实现不分离: Chief architects also write and optimize code
- 量化目标: All targets must be measurable and specific throughout the project lifecycle

---



| User Request Keywords | Artifact to Produce |
|---|---|
| "requirements", "shall statements", "stakeholder needs" | Requirements Document (SRS/SyRS) |
| "architecture", "block diagram", "system design" | System Architecture Description |
| "interface", "ICD", "data exchange" | Interface Control Document (ICD) |
| "ConOps", "concept of operations", "use case" | Concept of Operations (ConOps) |
| "trade study", "trade-off", "option analysis" | Trade Study Report |
| "FMEA", "failure mode", "hazard" | FMEA / Risk Register |
| "V&V", "test plan", "verification matrix" | Verification & Validation Plan / RVTM |
| "decomposition", "WBS", "system breakdown" | System Breakdown Structure |
| "MBSE", "SysML", "model" | SysML Diagrams (BDD, IBD, req, sequence) |
| "三一工程", "器件受限", "系统集成竞争力" | 三一工程分析 (System Integration Strategy) |
| "五看三定", "战略规划", "竞争分析", "FSTD" | 五看三定战略分析报告 (FSTD Strategic Analysis) |
| "作战地图", "资源配置", "项目规划" | 作战地图 (Battlefield Map Plan) |
| "专家建模", "知识萃取", "DSMM" | DSMM 思维模型 (Expert Mind Map) |
| "产业投资", "赛道分析", "SBIIEM" | SBIIEM 产业投资论证报告 |

---

## Workflow by Task Type

### 1. Requirements Engineering

**Steps:**
1. Elicit stakeholder needs (who are the stakeholders? what do they need?)
2. Transform needs → system requirements using "The system **shall**..." format
3. Apply SMART criteria: Specific, Measurable, Achievable, Relevant, Time-bound
4. Assign unique IDs: `SYS-REQ-XXXX`
5. Tag attributes: Priority (Critical/High/Medium/Low), Source, Verification Method
6. Check for: completeness, consistency, feasibility, unambiguity, traceability

**Requirement Writing Rules:**
- Use **SHALL** for mandatory requirements
- Use **SHOULD** for goals/recommendations
- Use **WILL** for declarations of fact
- Avoid: "and/or", vague terms ("fast", "reliable"), implementation-prescriptive language
- Each requirement = one obligation only (atomic)

**Output Template:**
```
REQ-ID    | [SYS-REQ-0001]
Title     | [Short name]
Statement | The system SHALL [action/capability] [condition] [constraint]
Rationale | [Why this requirement exists]
Source    | [Stakeholder / regulation / standard]
Priority  | [Critical / High / Medium / Low]
Verify By | [Test / Analysis / Inspection / Demonstration]
Status    | [Draft / Approved / Implemented / Verified]
```

---

### 2. System Architecture Design

**Steps:**
1. Define the system boundary (what's in / what's out)
2. Identify external interfaces (systems, humans, environments)
3. Decompose into functional subsystems (Level 1, Level 2...)
4. Allocate requirements to subsystems
5. Define internal interfaces between subsystems
6. Select architecture pattern (centralized, distributed, federated, layered...)
7. Document key design decisions with rationale (Architecture Decision Records)

**Architecture Description must include:**
- **Context Diagram** — System in its operational environment
- **Functional Architecture** — Functions the system performs
- **Physical/Logical Architecture** — How functions are realized
- **Interface Matrix** — All system interfaces (internal & external)

**SysML Diagram Guidance (use ASCII/text notation if no tool available):**
```
Block Definition Diagram (BDD):
  [System] ──────────────────────────────
  | + attribute: Type                    |
  | + operation(): ReturnType            |
  ──────────────────────────────────────
        ◇ (aggregation) child blocks

Internal Block Diagram (IBD):
  [system:System]
    [sub1:SubsystemA] ──(interface)──▶ [sub2:SubsystemB]

Requirements Diagram:
  «requirement» REQ-0001
  ──────────────────
  id = "SYS-REQ-0001"
  text = "The system shall..."
```

---

### 3. Interface Control Document (ICD)

**ICD Structure:**
1. Interface Identification (unique ID, name, type)
2. Interface Parties (Producer / Consumer systems)
3. Physical Interface (connector, protocol, voltage, form factor)
4. Logical/Data Interface (data format, schema, message structure)
5. Timing & Performance (frequency, latency, bandwidth)
6. Error Handling (fault detection, recovery behavior)
7. Security & Access Control
8. Verification Criteria

**Interface Table Format:**
```
Interface ID | Name | From | To | Type | Protocol | Data | Rate | Notes
ICD-0001     | ...  | Sys A| Sys B | Electrical | CAN 2.0B | Telemetry | 100 Hz | ...
```

---

### 4. Trade Study

**Standard Trade Study Process:**
1. Define the objective & evaluation criteria
2. Generate candidate alternatives (minimum 3)
3. Weight criteria (Analytic Hierarchy Process or direct weighting — must sum to 1.0)
4. Score each alternative against each criterion (1–10 scale, document rationale)
5. Calculate weighted scores
6. Sensitivity analysis (vary top weights ±20%, check if winner changes)
7. Recommend preferred alternative with documented rationale

**Trade Matrix Template:**
```
Criterion        | Weight | Alt A | Alt B | Alt C
Performance      | 0.30   |  8    |  6    |  9
Cost             | 0.25   |  7    |  9    |  5
Schedule Risk    | 0.20   |  6    |  8    |  7
Reliability      | 0.15   |  9    |  7    |  6
Maintainability  | 0.10   |  7    |  6    |  8
─────────────────────────────────────────────────
Weighted Score   |        | 7.45  | 7.30  | 7.00
```

---

### 5. FMEA / Risk Analysis

**FMEA Steps:**
1. Define system scope and indenture level
2. List all functions/components
3. For each function: identify failure modes
4. For each failure mode: identify effects (local, next level, end effect)
5. Identify causes of failure
6. Assign severity (S), occurrence (O), detectability (D) on 1–10 scales
7. Calculate Risk Priority Number: **RPN = S × O × D**
8. Define mitigation actions for RPN ≥ threshold
9. Re-assess after mitigation

**Severity Scale:**
- 10: Catastrophic (loss of life / mission)
- 7–9: Critical (major system failure)
- 4–6: Major (degraded performance)
- 1–3: Minor (nuisance)

**Risk Register Format:**
```
Risk-ID | Description | Likelihood (1-5) | Consequence (1-5) | Risk Score | Mitigation | Owner | Status
```

---

### 6. Verification & Validation Plan

**V-Model Mapping:**
```
Stakeholder Needs ─────────────────────────────► Operational V&V
    System Requirements ─────────────────► System Verification
        Subsystem Requirements ─────► Subsystem Verification
            Component Requirements ► Unit/Component Test
```

**RVTM (Requirements Verification Traceability Matrix):**
```
REQ-ID | Requirement Text | Verify Method | Test ID | Procedure | Pass Criteria | Status
```

**Verification Methods (TAID):**
- **T**est — Execute under controlled conditions, measure result
- **A**nalysis — Mathematical/computational analysis
- **I**nspection — Visual or physical check
- **D**emonstration — Observe operational behavior

---

### 7. Concept of Operations (ConOps)

**ConOps Sections (per IEEE 1362):**
1. **Scope** — System name, purpose, stakeholders
2. **Referenced Documents**
3. **Current System/Situation** — As-Is description
4. **Justification for New System** — Problem statement
5. **Concept for Proposed System** — To-Be description
6. **Operational Scenarios** — Step-by-step use cases
7. **Operational Environment** — Physical, technical environment
8. **Impacts** — Organizational, operational
9. **Analysis of Proposed System** — Benefits, limitations, risks

---

## Output Formatting Standards

### Document Header (all artifacts)
```
Document Title:    [Name]
Document Number:   [Project]-[Type]-[Seq]
Revision:          [A / 1.0 / Draft]
Date:              [YYYY-MM-DD]
Author:            [Name / Team]
Approval Status:   [Draft / Under Review / Approved / Controlled]
Classification:    [Unclassified / Proprietary / Controlled]
```

### Numbering Conventions
- Requirements: `SYS-REQ-XXXX`, `SUB-REQ-XXXX`, `SW-REQ-XXXX`, `HW-REQ-XXXX`
- Interfaces: `ICD-XXXX`
- Risks: `RISK-XXXX`
- Tests: `TEST-XXXX`
- Actions: `ACT-XXXX`

### Traceability
Always include a traceability section or matrix that maps:
- Stakeholder needs → System requirements
- System requirements → Subsystem requirements
- Requirements → Design elements
- Requirements → Verification tests

---

## Common Mistakes to Avoid

| Mistake | Correct Practice |
|---|---|
| Vague requirement: "system shall be fast" | Specify: "system SHALL respond within 200ms under nominal load" |
| Requirements that mix multiple obligations | One SHALL per requirement |
| Architecture without interface definition | Every subsystem boundary = documented interface |
| Trade study without sensitivity analysis | Always vary weights ±20% to test robustness |
| FMEA without mitigation actions | Every high-RPN item needs a mitigation owner and due date |
| Verification plan created at end of project | V&V planned at same time as requirements |

---

## Standards Reference

| Standard | Domain | Key Content |
|---|---|---|
| ISO/IEC/IEEE 15288:2023 | SE Lifecycle | System lifecycle processes |
| ISO/IEC/IEEE 29148:2018 | Requirements | Requirements engineering practices |
| IEEE 1362-1998 | ConOps | Concept of operations format |
| MIL-STD-882E | Safety | System safety analysis |
| MIL-STD-1629A | Reliability | FMECA procedure |
| NASA SP-2016-6105 | SE Handbook | End-to-end SE process |
| INCOSE SE Handbook v4 | SE Practice | Industry best practices |
| OMG SysML v1.6/v2.0 | MBSE | Systems Modeling Language |
| 华为《系统工程方法与实践》 | 华为SE | 三一工程、五看三定、作战地图、DSMM、SBIIEM |

---

## Example Prompts This Skill Handles

- "Write requirements for an autonomous drone delivery system"
- "Create an ICD between the flight computer and the ground station"
- "Run a trade study on battery vs fuel cell for the power subsystem"
- "Build an FMEA for the braking system"
- "Create a ConOps for a hospital patient monitoring system"
- "Develop the verification matrix for our radar system requirements"
- "Design the system architecture for a satellite communication system"
- "Decompose the propulsion system into subsystems"
- "用三一工程方法分析我们在芯片受限情况下的5G基站设计策略"
- "用五看三定对我们进入工业机器人市场做战略分析"
- "为我们的AI加速器研发项目制定作战地图"
- "对我们首席架构师的决策经验做DSMM建模"
- "用SBIIEM方法评估进入车规芯片赛道的可行性"

---

## Quick Reference Checklist

Before delivering any systems engineering artifact, verify:

- [ ] All requirements use SHALL/SHOULD/WILL correctly
- [ ] Every requirement has a unique ID
- [ ] Every requirement has a verification method
- [ ] All interfaces are identified and named
- [ ] Traceability exists between levels
- [ ] Risks are quantified (likelihood × consequence)
- [ ] Document header includes title, number, revision, date
- [ ] Standards references are cited where relevant
- [ ] Assumptions are explicitly stated
- [ ] Open items / TBDs are flagged with owners

**华为方法附加检查项 (when applicable):**
- [ ] 三一工程: 功能层/逻辑层/实现层 已解耦？是否避免过设计？
- [ ] 三一工程: 是否同时储备 Plan A / Plan B-1 / Plan B-2 方案？
- [ ] 四化设计: 架构是否满足平台化/模块化/归一化/互置化要求？
- [ ] 五看三定: 五看分析是否覆盖行业/市场/竞争/自身/机会？
- [ ] 五看三定: 是否定义了战略控制点、双目标（达标+挑战）、策略？
- [ ] 作战地图: 资源是否按肥肉/瘦肉/骨头/石头分级配置？
- [ ] 作战地图: 是否有复盘迭代（AAR）机制？
- [ ] SBIIEM: 论证是否覆盖五个维度（技术/市场/产业链/政策/财务）？
