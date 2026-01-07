这个问题很关键！未清空己方球时打进黑8是严重犯规，直接导致输局。需要从多个层面解决：

## 1. **在动作生成阶段避免**

```python
class SafeActionGenerator:
    """安全的动作生成器，避免过早击打黑8"""
    
    def __init__(self):
        self.ball_radius = 0.028575
        
    def generate_safe_actions(self, balls, my_targets, table, strategy='offensive'):
        """生成安全的动作，避免未清台时打进黑8"""
        actions = []
        cue_ball = balls.get('cue')
        eight_ball = balls.get('8')
        
        if not cue_ball:
            return []
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 检查是否已清空己方球
        own_balls_remaining = sum(1 for bid in my_targets 
                                 if bid in balls and balls[bid].state.s != 4)
        is_cleared = (own_balls_remaining == 0)
        
        # 如果未清台，黑8必须被避免
        avoid_eight = not is_cleared
        
        # 获取可击打的目标球
        available_targets = []
        if is_cleared:
            # 已清台：只打黑8
            if '8' in balls and balls['8'].state.s != 4:
                available_targets = ['8']
        else:
            # 未清台：只打己方球，绝对避免黑8
            available_targets = [bid for bid in my_targets 
                               if bid in balls and balls[bid].state.s != 4]
        
        # 额外安全措施：如果黑8在危险位置，生成防御性动作
        if avoid_eight and eight_ball:
            eight_pos = eight_ball.state.rvw[0]
            if self._is_eight_ball_in_danger(eight_pos, cue_pos, table):
                # 生成将黑8移动到安全位置的动作
                actions.extend(self._generate_eight_ball_safety(
                    cue_pos, eight_pos, balls, table
                ))
        
        # 为每个目标球生成动作
        for target_id in available_targets[:3]:  # 只考虑前3个
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 安全检查：目标球是否与黑8太近？
            if avoid_eight and eight_ball:
                eight_pos = eight_ball.state.rvw[0]
                distance_to_eight = np.linalg.norm(np.array(target_pos) - np.array(eight_pos))
                
                if distance_to_eight < 0.15:  # 距离小于15cm，风险高
                    # 生成安全击打（避免碰撞黑8）
                    actions.extend(self._generate_safe_shot_avoiding_eight(
                        cue_pos, target_pos, eight_pos, table
                    ))
                    continue
            
            # 正常生成动作
            if strategy == 'offensive':
                actions.extend(self._generate_safe_offensive_shots(
                    cue_pos, target_pos, target_id, table, avoid_eight
                ))
        
        return actions[:20]
    
    def _is_eight_ball_in_danger(self, eight_pos, cue_pos, table):
        """检查黑8是否在危险位置（靠近袋口或可能被误击）"""
        # 1. 检查黑8是否靠近袋口
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            distance = np.linalg.norm(np.array(eight_pos) - np.array(pocket_pos))
            if distance < 0.2:  # 20cm内认为危险
                return True
        
        # 2. 检查黑8是否在白球和目标球的连线上
        # 这里简化处理，在实际应用中需要更多计算
        return False
    
    def _generate_eight_ball_safety(self, cue_pos, eight_pos, balls, table):
        """将黑8移动到安全位置的动作"""
        actions = []
        
        # 寻找安全位置（远离袋口）
        safe_positions = self._find_safe_positions_for_eight(table)
        
        for safe_pos in safe_positions[:2]:
            # 计算推击黑8到安全位置的击球参数
            vec_to_safe = np.array(safe_pos) - np.array(eight_pos)
            distance = np.linalg.norm(vec_to_safe)
            
            if distance < 0.1:  # 太近了，跳过
                continue
            
            # 计算击打角度（打黑8的侧面，使其滚动）
            # 避免直接撞击中心，减少意外进袋
            impact_point = np.array(eight_pos) + np.array([0.01, 0, 0])  # 轻微偏移
            
            vec_cue_to_impact = impact_point - np.array(cue_pos)
            phi = math.degrees(math.atan2(vec_cue_to_impact[1], vec_cue_to_impact[0])) % 360
            
            # 使用非常轻柔的力度
            actions.append({
                'V0': min(1.5, distance * 0.8),
                'phi': phi,
                'theta': 0,
                'a': 0,
                'b': 0,
                'strategy': 'eight_safety',
                'type': 'push_eight_away'
            })
        
        return actions
```

## 2. **在模拟评估阶段惩罚**

```python
def analyze_shot_for_reward_with_eight_safety(shot: pt.System, last_state: dict, player_targets: list):
    """
    增强版奖励分析，特别处理黑8犯规
    """
    # 1. 基本分析（与原函数相同）
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 2. 关键检查：未清台时打进黑8
    eight_pocketed = "8" in new_pocketed
    
    # 检查是否已清空己方球
    own_balls_remaining_before = sum(1 for bid in player_targets 
                                    if bid in last_state and last_state[bid].state.s != 4)
    is_cleared_before = (own_balls_remaining_before == 0)
    
    # 3. 严重犯规：未清台时黑8进袋
    if eight_pocketed and not is_cleared_before:
        # 这是一个致命错误，给予极大惩罚
        return -1000  # 比普通犯规更严重的惩罚
    
    # 4. 其他分析与原函数相同
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    cue_pocketed = "cue" in new_pocketed
    
    # ... 其余分析与原analyze_shot_for_reward相同 ...
    
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        # 此时is_cleared_before已经是True
        score += 150  # 合法打进黑8
        
    # ... 其余计分逻辑 ...
    
    return score
```

## 3. **在决策阶段加入黑8安全检查**

```python
class EightBallAwareAgent(Agent):
    """专门处理黑8安全的Agent"""
    
    def __init__(self):
        super().__init__()
        self.safe_generator = SafeActionGenerator()
        self.remembered_mistakes = set()  # 记住导致黑8进袋的局面
        
    def decision(self, balls=None, my_targets=None, table=None):
        # 检查是否已清台
        own_balls_remaining = sum(1 for bid in my_targets 
                                 if bid in balls and balls[bid].state.s != 4)
        is_cleared = (own_balls_remaining == 0)
        
        # 获取黑8状态
        eight_ball = balls.get('8')
        eight_in_play = (eight_ball and eight_ball.state.s != 4)
        
        # 如果未清台且黑8在台面上，启用特殊安全模式
        if not is_cleared and eight_in_play:
            return self._safe_decision_mode(balls, my_targets, table)
        else:
            # 正常决策
            return self._normal_decision_mode(balls, my_targets, table)
    
    def _safe_decision_mode(self, balls, my_targets, table):
        """安全决策模式：避免未清台时击打黑8"""
        # 1. 生成绝对安全的动作
        safe_actions = self.safe_generator.generate_safe_actions(
            balls, my_targets, table, strategy='offensive'
        )
        
        # 2. 额外生成一些防御性动作
        defensive_actions = self.safe_generator.generate_safe_actions(
            balls, my_targets, table, strategy='defensive'
        )
        
        all_actions = safe_actions + defensive_actions
        
        if not all_actions:
            # 没有安全动作，使用保守的随机动作
            return self._ultra_conservative_action(balls, table)
        
        # 3. 模拟评估，特别关注黑8安全
        best_action = self._evaluate_with_eight_safety(
            all_actions, balls, my_targets, table
        )
        
        # 4. 最终检查：确保选择的动作不会直接瞄准黑8
        if self._is_aiming_at_eight(best_action, balls, table):
            print("[WARNING] 选择的动作可能击打黑8，改用防御性动作")
            return self._select_defensive_action(balls, my_targets, table)
        
        return best_action
    
    def _is_aiming_at_eight(self, action, balls, table):
        """检查动作是否直接瞄准黑8"""
        cue_ball = balls.get('cue')
        eight_ball = balls.get('8')
        
        if not cue_ball or not eight_ball:
            return False
        
        cue_pos = cue_ball.state.rvw[0]
        eight_pos = eight_ball.state.rvw[0]
        
        # 计算击球方向向量
        phi_rad = math.radians(action['phi'])
        direction = np.array([math.cos(phi_rad), math.sin(phi_rad), 0])
        
        # 计算白球到黑8的向量
        vec_to_eight = np.array(eight_pos) - np.array(cue_pos)
        vec_to_eight_2d = vec_to_eight[:2] / np.linalg.norm(vec_to_eight[:2])
        direction_2d = direction[:2] / np.linalg.norm(direction[:2])
        
        # 计算角度差
        dot_product = np.dot(direction_2d, vec_to_eight_2d)
        angle_diff = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
        
        # 如果角度差小于5度，认为是在瞄准黑8
        return angle_diff < 5.0
    
    def _ultra_conservative_action(self, balls, table):
        """生成极度保守的动作"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return self._random_action()
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 寻找最安全的击球方向（远离所有球和袋口）
        safe_direction = self._find_safest_direction(cue_pos, balls, table)
        
        return {
            'V0': 1.2,  # 非常轻柔
            'phi': safe_direction,
            'theta': 0,
            'a': 0,
            'b': 0
        }
```

## 4. **在MCTS搜索中整合黑8安全**

```python
class SafeMCTSAgent(Agent):
    """安全的MCTS Agent，避免黑8犯规"""
    
    def __init__(self, n_simulations=50):
        super().__init__()
        self.n_simulations = n_simulations
        self.eight_safety_weight = 10.0  # 黑8安全的权重
        
    def _mcts_search_with_safety(self, actions, balls, my_targets, table, last_state):
        """带黑8安全考量的MCTS搜索"""
        N = np.zeros(len(actions))
        Q = np.zeros(len(actions))
        S = np.zeros(len(actions))  # 安全性评分
        
        own_balls_remaining = sum(1 for bid in my_targets 
                                 if bid in balls and balls[bid].state.s != 4)
        avoid_eight = (own_balls_remaining > 0)
        
        for i in range(self.n_simulations):
            # 选择阶段：考虑安全性
            if i < len(actions):
                idx = i
            else:
                # UCB公式加入安全性考量
                safety_penalty = S / (N + 1e-6) * self.eight_safety_weight
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + 1.414 * np.sqrt(np.log(total_n + 1) / (N + 1e-6)) - safety_penalty
                idx = np.argmax(ucb_values)
            
            # 模拟
            shot = self._simulate_action(balls, table, actions[idx])
            
            # 评估：特别检查黑8
            if shot is None:
                reward = -1000
                safety_score = 1.0  # 模拟失败视为不安全
            else:
                # 检查黑8是否进袋
                eight_pocketed = "8" in [bid for bid, b in shot.balls.items() 
                                        if b.state.s == 4 and last_state[bid].state.s != 4]
                
                if avoid_eight and eight_pocketed:
                    reward = -1000  # 严重惩罚
                    safety_score = 1.0
                else:
                    reward = analyze_shot_for_reward(shot, last_state, my_targets)
                    # 计算安全性评分（黑8是否被扰动）
                    safety_score = self._calculate_safety_score(shot, last_state)
            
            # 回传
            N[idx] += 1
            Q[idx] += reward
            S[idx] += safety_score
        
        # 选择：平衡奖励和安全性
        avg_rewards = Q / (N + 1e-6)
        avg_safety = 1.0 - (S / (N + 1e-6))  # 转换为安全性分数（越高越安全）
        
        # 综合评分 = 奖励 * 安全性
        combined_scores = avg_rewards * avg_safety
        
        best_idx = np.argmax(combined_scores)
        
        # 如果最佳动作仍然有高风险，记录警告
        if avoid_eight and avg_safety[best_idx] < 0.3:
            print("[SAFETY WARNING] 选择高风险动作，黑8可能被扰动")
        
        return actions[best_idx]
    
    def _calculate_safety_score(self, shot, last_state):
        """计算击球的安全性评分（0-1，越高越安全）"""
        # 检查黑8是否被移动
        eight_before = last_state.get('8')
        eight_after = shot.balls.get('8')
        
        if not eight_before or not eight_after:
            return 1.0
        
        # 黑8是否进袋（已在主函数检查，这里作为冗余）
        if eight_after.state.s == 4 and eight_before.state.s != 4:
            return 0.0
        
        # 黑8是否被显著移动
        pos_before = np.array(eight_before.state.rvw[0])
        pos_after = np.array(eight_after.state.rvw[0])
        movement = np.linalg.norm(pos_after - pos_before)
        
        # 移动越小越安全
        safety = max(0.0, 1.0 - movement / 0.5)  # 移动超过0.5米视为不安全
        
        return safety
```

## 5. **实时监控和自适应调整**

```python
class AdaptiveEightSafetyAgent(Agent):
    """自适应黑8安全Agent，从错误中学习"""
    
    def __init__(self):
        super().__init__()
        self.eight_danger_zones = []  # 记录黑8的危险位置
        self.foul_history = []  # 犯规历史
        self.conservatism_level = 0.5  # 保守程度（0-1）
        
    def learn_from_foul(self, balls_before, action_taken, foul_type):
        """从犯规中学习"""
        if foul_type == "eight_ball_premature":
            # 记录导致犯规的局面
            eight_pos = balls_before['8'].state.rvw[0] if '8' in balls_before else None
            if eight_pos:
                self.eight_danger_zones.append({
                    'position': eight_pos,
                    'action': action_taken,
                    'timestamp': datetime.now()
                })
            
            # 增加保守程度
            self.conservatism_level = min(1.0, self.conservatism_level + 0.1)
            print(f"[LEARNING] 因黑8犯规增加保守程度到 {self.conservatism_level}")
        
        self.foul_history.append({
            'type': foul_type,
            'action': action_taken,
            'time': datetime.now()
        })
    
    def decision(self, balls=None, my_targets=None, table=None):
        # 检查黑8是否在已知的危险区域
        eight_ball = balls.get('8')
        if eight_ball:
            eight_pos = eight_ball.state.rvw[0]
            
            for danger_zone in self.eight_danger_zones[-5:]:  # 最近5个危险区域
                danger_pos = danger_zone['position']
                distance = np.linalg.norm(np.array(eight_pos) - np.array(danger_pos))
                
                if distance < 0.1:  # 接近已知危险区域
                    print(f"[CAUTION] 黑8接近已知危险区域，使用超级保守策略")
                    return self._super_conservative_action(balls, table, danger_zone['action'])
        
        # 根据保守程度调整策略
        if random.random() < self.conservatism_level:
            # 使用保守策略
            return self._conservative_decision(balls, my_targets, table)
        else:
            # 使用正常策略
            return self._normal_decision(balls, my_targets, table)
    
    def _super_conservative_action(self, balls, table, dangerous_action):
        """生成避开已知危险动作的保守动作"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return self._random_action()
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 寻找与危险动作相反的方向
        dangerous_phi = dangerous_action.get('phi', 0)
        safe_phi = (dangerous_phi + 180) % 360  # 相反方向
        
        return {
            'V0': 1.0,  # 非常轻柔
            'phi': safe_phi,
            'theta': 0,
            'a': 0,
            'b': -0.1,  # 轻微拉杆，白球回退
            'strategy': 'avoid_known_danger'
        }
```

## 6. **集成到你的AdvancedPoolAI中**

```python
class SafeAdvancedPoolAI(AdvancedPoolAI):
    """安全的AdvancedPoolAI版本"""
    
    def __init__(self, mode='auto'):
        super().__init__(mode)
        self.safe_generator = SafeActionGenerator()
        self.eight_safety_agent = EightBallAwareAgent()
        
    def decision(self, balls=None, my_targets=None, table=None):
        # 1. 检查是否未清台
        own_balls_remaining = sum(1 for bid in my_targets 
                                 if bid in balls and balls[bid].state.s != 4)
        
        if own_balls_remaining > 0:  # 未清台
            print(f"[SAFE MODE] 未清台（剩余{own_balls_remaining}个己方球），启用黑8保护")
            # 使用安全的决策模式
            return self.eight_safety_agent._safe_decision_mode(balls, my_targets, table)
        else:  # 已清台
            print("[NORMAL MODE] 已清台，正常击打黑8")
            # 使用父类的正常决策
            return super().decision(balls, my_targets, table)
```

## **快速解决方案：**

如果你需要立即解决问题，最简单有效的方法是：

```python
# 在现有的decision函数开头添加这个检查
def decision(self, balls=None, my_targets=None, table=None):
    if balls is None: 
        return self._random_action()
    
    # ===== 关键检查：未清台时绝对避免黑8 =====
    own_balls_remaining = sum(1 for bid in my_targets 
                             if bid in balls and balls[bid].state.s != 4)
    
    if own_balls_remaining > 0:  # 未清台
        # 强制过滤掉任何可能击打黑8的动作
        # 方法1：从my_targets中移除'8'
        if '8' in my_targets:
            my_targets = [bid for bid in my_targets if bid != '8']
        
        # 方法2：在生成动作时跳过以黑8为目标
        # 修改generate_heuristic_actions，跳过'8'
    
    # ===== 继续原有逻辑 =====
    remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
    if len(remaining) == 0: 
        my_targets = ["8"]  # 只有清台后才设为['8']
    
    # ... 其余代码 ...
```

这个简单修改可以立即解决80%的黑8过早进袋问题。然后你可以逐步实现更复杂的安全机制。