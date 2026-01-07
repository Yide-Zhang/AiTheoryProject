# NewAgentFinal Implementation Notes

## 1. Design Objectives
- 平衡进攻/防守：结合安全球、拉杆、阵地选择，避免提前击打黑8并维持可清台线路。
- 自适应策略：根据剩余目标球、对手压力与球桌风险，在激进、平衡、防守模式间切换。
- 高质量开球：通过 BreakShotExpert 在 break 状态下枚举高能量方案，最大化散球质量并压制犯规。
- 稳定评估：对候选击球进行多次 Monte Carlo 仿真和扰动微调，获得可靠评分。

## 2. 模块化架构一览
| 层级 | 职责 | 关键代码 |
| --- | --- | --- |
| 开球专家 | 根据球堆结构生成多种 break 方案、二次球切球 | [agents/new_agent_final.py#L17-L140](agents/new_agent_final.py#L17-L140) |
| 安全动作生成器 | 在剩余目标>0时保护黑8，提供藏球/回防策略 | [agents/new_agent_final.py#L143-L346](agents/new_agent_final.py#L143-L346) |
| 平衡规划器 | 解析局面、选择策略、合并候选动作 | [agents/new_agent_final.py#L348-L870](agents/new_agent_final.py#L348-L870) |
| 统一 Agent | Break 判定、候选生成、仿真打分、变异优化 | [agents/new_agent_final.py#L871-L1694](agents/new_agent_final.py#L871-L1694) |
| 对战脚本 | 评估/环境均默认使用 NewAgentFinal | [evaluate.py#L27-L39](evaluate.py#L27-L39), [poolenv.py#L25-L510](poolenv.py#L25-L510) |

## 3. 关键组件细节（可直接放入 PPT）
### 3.1 BreakShotExpert
- 识别球堆头球与相邻次球，拼接核心击打、pop、左右 cut 以及 second-ball 变体。
- 回调 `_package_action` 保持统一动作格式，输出供 break 评分使用。

```python
class BreakShotExpert:
    """Generates high-energy break shot candidates."""

    def __init__(self, agent: "NewAgentFinal") -> None:
        self._agent = agent

    def candidates(self, balls: BallDict, table: pt.Table) -> List[Dict[str, object]]:
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        head_id, head_pos = self._locate_head_ball(balls, table)
        if head_id is None or head_pos is None:
            return []

        actions: List[Dict[str, object]] = []
        base_phi = self._bearing(cue_pos, head_pos)
        base_power = 7.4

        actions.append(self._package_action(base_power, base_phi, 0.0, 0.0, -0.18, "core", head_id))
        actions.append(self._package_action(base_power * 0.97, base_phi, 2.5, 0.0, -0.05, "pop", head_id))

        for offset, spin in ((-2.0, -0.2), (2.0, 0.2)):
            actions.append(
                self._package_action(
                    base_power * 0.99,
                    (base_phi + offset) % 360.0,
                    0.0,
                    spin,
                    -0.12,
                    f"cut_{'L' if offset < 0 else 'R'}",
                    head_id,
                )
            )

        actions.extend(self._second_ball_variants(balls, cue_pos, head_pos))
        return actions
```

### 3.2 SafeActionGenerator（黑8保护 + 复合安全球）
- 根据当前目标球和黑8位置筛选安全打点；若己方尚未清台则避免任何击打路径穿过黑8。
- 当周边没有安全候选时，自动构造 parking/hide 等 fallback。

```python
def generate_safe_actions(
    self,
    balls: BallDict,
    my_targets: Optional[List[str]],
    table: pt.Table,
    limit: int = 32,
) -> List[Dict[str, object]]:
    cue_ball = balls.get("cue")
    if cue_ball is None or my_targets is None:
        return []

    cue_pos = np.array(cue_ball.state.rvw[0][:2])
    eight_ball = balls.get("8")
    eight_pos = None
    if eight_ball is not None and eight_ball.state.s != 4:
        eight_pos = np.array(eight_ball.state.rvw[0][:2])

    own_remaining = [
        bid
        for bid in my_targets
        if bid != "8" and bid in balls and balls[bid].state.s != 4
    ]
    avoid_eight = bool(own_remaining)
    ...
    if avoid_eight and eight_pos is not None and self._line_intersects_eight(cue_pos, ghost_pos, eight_pos):
        continue
    ...
```

### 3.3 BalancedActionPlanner（风险自适应）
- `analyze_state` 汇总己方/对手剩余球、容易球数量、母球风险、黑8危险度与窗口期。
- `select_strategy` 依据风险阈值、风险容忍度切换策略；`generate_actions` 将策略映射为动作组合；`merge_candidates` 用 `_score_candidate_priority` 重新排序。

```python
def select_strategy(self, game_state: BalancedGameState, risk_tolerance: float) -> str:
    if game_state.own_remaining == 0:
        return "eight_only"
    if game_state.easy_shots >= 2 and game_state.eight_danger < 0.3:
        return "aggressive_clearance"
    if game_state.opponent_remaining <= 2 and game_state.own_remaining <= 3:
        return "mixed_endgame"

    risk_pressure = max(game_state.cue_risk, game_state.eight_danger)
    defensive_bar = 0.55 - risk_tolerance * 0.2
    if game_state.easy_shots == 0 or risk_pressure > defensive_bar:
        return "defensive"
    if game_state.offensive_window > (0.5 + 0.2 * risk_tolerance):
        return "balanced_aggressive"
    return "balanced"
```

### 3.4 候选生成 → 仿真 → 精细化
1. `_build_base_candidates` 通过鬼球解 + multiple spin variant 获取基准进攻/防守动作。
2. `_generate_candidates` 将安全模式/平衡规划融入，并记录上一轮策略，方便日志。
3. `_evaluate_candidate` **多次仿真** + `_mutate_action` 随机微调，筛选高分动作。

```python
def _generate_candidates(...):
    base_candidates = self._build_base_candidates(...)

    if self._needs_eight_protection(...):
        safe_candidates = self.safe_generator.generate_safe_actions(...)
        filtered_base = [info for info in base_candidates if info.get("target") != "8"]
        combined = safe_candidates + filtered_base
    else:
        combined = base_candidates

    game_state = self.balanced_planner.analyze_state(...)
    strategy = self.balanced_planner.select_strategy(game_state, self.risk_tolerance)
    balanced_candidates = self.balanced_planner.generate_actions(...)
    merged = self.balanced_planner.merge_candidates(...)
    self._last_strategy = strategy
    return merged
```

```python
def _evaluate_candidate(...):
    samples: List[float] = []
    for _ in range(self.evaluation_repeats):
        noisy_action = self._apply_eval_noise(action)
        shot = self._simulate_action(balls, table, noisy_action)
        if shot is None:
            samples.append(-600.0)
            continue

        base_reward = self._score_shot(shot, last_state_snapshot, my_targets)
        positional_value = self._positional_bonus(shot.balls, my_targets, table, info)
        samples.append(base_reward + positional_value)

    return float(np.mean(samples)) if samples else -600.0
```

### 3.5 Break 评分与防守兜底
- `_decide_break_shot` 仅在 `_is_break_state` 为真时触发，先用 BreakShotExpert 生成动作，再额外变异若干次，交给 `_score_break_shot` 按散球、犯规、Cue 控制评分。
- `_safe_mode_decision` 在检测到“黑8尚未安全”时直接使用 SafeActionGenerator 输出的最优得分动作，并提供 `_fallback_defensive_action` 兜底。

```python
def _decide_break_shot(self, balls: BallDict, table: pt.Table) -> Optional[ActionDict]:
    candidates = self.break_expert.candidates(balls, table)
    if not candidates:
        return None

    scored: List[Tuple[float, ActionDict, Dict[str, object]]] = []
    for info in candidates:
        score = self._evaluate_break_candidate(info["action"], balls, table)
        scored.append((score, info["action"], info))

    for base_score, base_action, info in scored[:3]:
        mutated = self._mutate_action(base_action, scale=0.2)
        score = self._evaluate_break_candidate(mutated, balls, table)
        scored.append((score, mutated, info))

    best = max(scored, key=lambda item: item[0])
    print(
        f"[NewAgentFinal] variant={best[2].get('variant')} score={best[0]:.1f} strategy=break"
    )
    return best[1]
```

## 4. 集成与使用
- `evaluate.py` 默认将 `agent_b` 设为 `NewAgentFinal()`，并保留一行注释方便切换到 BasicAgent 对手：[evaluate.py#L27-L39](evaluate.py#L27-L39)。
- 环境侧 `poolenv.py` 也在本地对战流程中实例化 `NewAgentFinal()`，确保离线训练/调试环境一致：[poolenv.py#L25-L510](poolenv.py#L25-L510)。
- 因所有 legacy Agent 已移除（仅保留 `agent.py`, `basic_agent.py`, `basic_agent_pro.py`, `new_agent_final.py`），无需担心旧实现被误用。

## 5. PPT 制作建议
1. **故事线**：
   - Slide 1：问题背景 + 目标（胜率 vs 安全）。
   - Slide 2：整体架构图（表 2 结构即可直接截图）。
   - Slide 3：BreakShotExpert & SafeActionGenerator（对比进攻/防守）。
   - Slide 4：BalancedActionPlanner 决策流程图（结合公式 $risk\_pressure = \max(cue\_risk, eight\_danger)$）。
   - Slide 5：候选仿真和变异流程，附日志 output 格式。
   - Slide 6：集成与测试结果（evaluate.py 输出、胜率表）。
2. **可视化素材**：
   - 使用表格展示策略切换阈值。
   - 用箭头图示 `_generate_candidates → _evaluate_candidate → best_action` pipeline。
3. **代码附录**：直接引用上文代码块或链接 `agents/new_agent_final.py` 对应行，保证 PPT 精简。
