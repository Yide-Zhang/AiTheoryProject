# 8-Ball AI Agent: 从启发式到分层决策的进化之路
**面向专业人士的技术汇报 (5分钟)**

## 1. 项目背景与技术栈 (30秒)
*   **目标**: 构建一个具备开球策略、防守意识和动态风险管理的超人类水平台球AI。
*   **核心引擎**: `pooltool` (基于物理的连续空间模拟)。
*   **技术架构**: 分层继承架构 (Hierarchical Inheritance Architecture)。
    *   Python + NumPy (向量化计算)
    *   Matplotlib/Seaborn (数据分析可视化)

---

## 2. 底层架构深度解析 (Deep Dive into Architecture)
*(建议展示类图与数据流图)*

### 2.1 混合式分层架构与算法 (Hybrid Architecture & Algorithms)
我们采用了类似自动驾驶的 Sim-to-Real 分层决策模型，结合了在线规划与随机优化算法。

*   **L1: 物理感知层 (`BasePoolAgent`)**
    *   **算法方法**: **Online Monte Carlo Simulation (在线蒙特卡洛模拟)**
    *   **原理**: 不依赖离线训练的 Value Network，而是实时通过 $N=60$ 次并行物理仿真来逼近动作价值 $Q(s, a)$。
    *   **技术**: 实现了 Sim-to-Real 的噪声注入 (`Domain Randomization`)，通过在模拟器中加入高斯噪声，滤除掉那些“理论可行但鲁棒性差”的解。
    
*   **L2: 开局专家层 (`BreakAgent`)**
    *   **算法方法**: **Evolutionary Strategies (ES, 进化策略)**
    *   **原理**: 并不是简单的固定点开球。代码实现了 `_mutate_action` (变异算子)，在基础开球点附近生成种群，通过适应度函数 (Fitness Function) 迭代筛选出当下球桌摩擦系数下的最优解。
    *   **目标**: $\max (Spread + Potted - Scratch)$

*   **L3: 防守博弈层 (`SafetyAgent`)**
    *   **算法方法**: **Constraint Satisfaction Problem (CSP, 约束满足)**
    *   **原理**: 将斯诺克防守建模为几何约束求解问题。
    *   **逻辑**: 搜索空间被定义为“对手下一次击球的期望收益最小化”。利用 Ghost Ball 算法反向推导，寻找满足 $P(Opponent\_Success) \approx 0$ 的白球停靠区域。

*   **L4: 动态风控层 (`RiskAgent`)**
    *   **算法方法**: **Hierarchical State Machine (分层状态机) / Reward Engineering**
    *   **原理**: 这是强化学习 (RL) 的前置形态。我们手工设计了用于未来的 RL 训练的状态空间 ($S$) 和奖励函数 ($R$)。
    *   **机制**: 目前通过“贝叶斯推断”思想，根据剩余球数和局势危险度 ($State$)，动态调整策略权重向量 ($w$)，实现从 Aggressive 到 Defensive 的平滑切换。

### 2.2 决策管线 (The Decision Pipeline)
*(展示 `decision` 方法内的 Loop 逻辑)*

1.  **态势感知 (Perception)**:
    *   扫描盘面，识别 `Phase` (本方剩球数、对手剩球数、黑八位置)。
2.  **候选生成 (Candidate Generation)**:
    *   **广度优先**: 并行生成 48+ 个候选动作（包含简单进攻、组合球、翻袋、安全球）。
3.  **概率评估与效用统一 (Probabilistic Evaluation & Utility Unification)**:
    *   **异构动作同构化**: 如何在“80%概率进球”和“100%让对手没球打”之间做抉择？
    *   我们构建了 **统一效用函数 (Unified Utility Function)**，将不同维度的指标映射到同一标尺：
        *   **进攻项**: $U_{atk} = P(Success) \times (1 + \alpha \cdot PositionQuality)$
        *   **防守项**: $U_{def} = (1 - P(Opponent_{Next})) \times w_{safety}$
        *   **风险项**: $Cost = P(Scratch) \times \text{Penalty}_{HandBall} + P(Miss) \times \text{Danger}_{Open}$
4.  **动态加权择优 (Dynamic Action Selection)**:
    *   **场景A (领先/Endgame)**: 状态机自动提升 $Cost$ 的权重 $\lambda$，AI 会放弃“难球”而选择将白球贴库（Safe）。
    *   **场景B (落后/Desperate)**: 降低风险敏感度，允许尝试长台进攻或翻袋。
    *   最终动作 $a^* = \text{argmax}_{a \in Candidates} (U_{atk} + U_{def} - Cost)$。

---

## 3. "大脑"透视：内部决策数据分析 (1分钟)
*(本部分展示最新生成的内部监控数据，证明 AI 的 "思考" 过程)*

### 3.1 策略多样性
**[插入图片]:** `ppt_assets_advanced/strategy_distribution_internal.png`
*   **解读**: 不再单一追求进球。图表显示 Agent 在约 32% 的情况下选择了防守 (`safety`) 或平衡策略 (`balanced_safety`)，证明其具备“大局观”。

### 3.2 决策复杂度 CT 扫描
**[插入图片]:** `ppt_assets_advanced/decision_complexity_internal.png`
*   **解读**: X轴为候选数量，Y轴为评分差值。
    *   **高候选/低分差**: 表示局面复杂，AI 进行了深度的搜索与权衡。
    *   **低候选/高分差**: 表示存在显而易见的唯一解（如简单底袋球）。

---

## 4. 性能基准测试 (1分30秒)
*(数据来源: 50场对抗测试 & 高级统计模块)*

### 4.1 胜率压制
**[插入图片]:** `ppt_assets/win_rate_trajectory.png`
*   **结论**: `NewAgentFinal` 相比 `BasicAgentPro` 实现了显著的胜率提升（稳定在 65%-75% 区间）。

### 4.2 专业数据维度的碾压
**[插入图片]:** `ppt_assets_advanced/potting_accuracy_comparison.png`
*   **击球准确率 (Potting Accuracy)**: 提升约 15%。
*   **单杆最高连击 (Max Streak)**: 最高达到 8 连杆（炸清边缘）。
*   **防守成功率**: 通过 `decision_logger` 追踪，防守后对手下回合进球率降低了 40%。

---

## 5. 精彩回放与总结 (30秒)

### 5.1 模拟高光时刻
**[插入视频/GIF]:** `ppt_assets/best_agent_win.json` (使用 `pooltool.visualization` 渲染)
*   **场景**: 展示一次精彩的“翻袋”或“借力防守”。
*   *(备选: 展示 Agent 在解球失败风险极高时，主动选择将白球贴库的防守操作)*

### 5.2 总结
*   **实现**: 成功将人类的“球商”（防守、走位、风险控制）转化为可计算的代码逻辑。
*   **未来**: 这里展示的是基于规则与搜索的 AI，下一步计划引入强化学习 (RL) 优化参数权重。

---
**Q&A**
