很好，这说明你的防御策略生效了，但进攻性不足。我们需要一个**平衡的策略**：既要安全避免黑8犯规，又要在机会出现时积极进攻。让我提供一个**平衡型Agent**：

## 1. **智能风险评估与决策树**

```python
class BalancedAgent(Agent):
    """平衡型Agent - 在安全和进攻间找到平衡"""
    
    def __init__(self, risk_tolerance=0.7):
        """
        risk_tolerance: 风险容忍度 (0.0-1.0)
            0.0 = 极度保守, 1.0 = 极度激进
        """
        super().__init__()
        self.risk_tolerance = risk_tolerance
        self.ball_radius = 0.028575
        
        # 关键：不同局面的策略优先级
        self.strategy_weights = {
            'high_risk_low_reward': 0.1,   # 高风险低回报：避免
            'medium_risk_high_reward': 0.7, # 中风险高回报：考虑
            'low_risk_high_reward': 0.9,   # 低风险高回报：优先
            'safety_only': 0.3,            # 纯安全球：次要
        }
        
        print(f"BalancedAgent 已初始化 (风险容忍度: {risk_tolerance})")
    
    def decision(self, balls, my_targets, table):
        """平衡决策：评估局面后选择最优策略"""
        # 1. 快速局面分析
        game_state = self._analyze_game_state(balls, my_targets)
        
        # 2. 根据局面选择主策略
        main_strategy = self._select_main_strategy(game_state)
        
        # 3. 生成对应策略的动作
        candidate_actions = self._generate_balanced_actions(
            balls, my_targets, table, main_strategy, game_state
        )
        
        # 4. 快速评估和选择
        if not candidate_actions:
            return self._smart_fallback_action(balls, my_targets, table)
        
        best_action = self._quick_evaluation(candidate_actions, balls, my_targets, table)
        
        return best_action
    
    def _analyze_game_state(self, balls, my_targets):
        """分析当前游戏状态，返回关键指标"""
        state = {
            'own_remaining': 0,
            'opponent_remaining': 0,
            'cue_position_risk': 0,
            'easy_shots_available': 0,
            'eight_ball_danger': 0,
            'offensive_opportunity': 0
        }
        
        # 计算己方剩余球数
        state['own_remaining'] = sum(1 for bid in my_targets 
                                    if bid in balls and balls[bid].state.s != 4)
        
        # 计算对方剩余球数
        all_balls = set(['1','2','3','4','5','6','7','9','10','11','12','13','14','15'])
        opponent_targets = [bid for bid in all_balls if bid not in my_targets]
        state['opponent_remaining'] = sum(1 for bid in opponent_targets 
                                         if bid in balls and balls[bid].state.s != 4)
        
        # 分析白球位置风险
        cue_ball = balls.get('cue')
        if cue_ball:
            cue_pos = cue_ball.state.rvw[0]
            # 白球是否在袋口附近
            state['cue_position_risk'] = self._calculate_cue_risk(cue_pos, table)
            
            # 计算容易进球的机会
            state['easy_shots_available'] = self._count_easy_shots(cue_pos, balls, my_targets, table)
        
        # 黑8风险分析
        eight_ball = balls.get('8')
        if eight_ball:
            state['eight_ball_danger'] = self._evaluate_eight_ball_danger(
                eight_ball.state.rvw[0], table, balls
            )
        
        # 进攻机会评估
        state['offensive_opportunity'] = self._evaluate_offensive_opportunity(
            balls, my_targets, table
        )
        
        return state
    
    def _select_main_strategy(self, game_state):
        """根据局面选择主要策略"""
        own = game_state['own_remaining']
        opp = game_state['opponent_remaining']
        easy_shots = game_state['easy_shots_available']
        eight_danger = game_state['eight_ball_danger']
        
        # 决策逻辑：
        if own == 0:
            # 只剩黑8
            return 'eight_ball_only'
        elif easy_shots >= 2 and eight_danger < 0.3:
            # 有多个容易进球且黑8安全：进攻
            return 'aggressive_clearance'
        elif opp <= 2 and own <= 3:
            # 接近尾声：混合策略
            return 'mixed_endgame'
        elif easy_shots == 0 or eight_danger > 0.6:
            # 没有容易进球或黑8危险：防守
            return 'defensive'
        else:
            # 默认：平衡策略
            return 'balanced'
    
    def _generate_balanced_actions(self, balls, my_targets, table, strategy, game_state):
        """生成平衡策略的动作"""
        actions = []
        cue_ball = balls.get('cue')
        
        if not cue_ball:
            return [self._random_action()]
        
        cue_pos = cue_ball.state.rvw[0]
        own_remaining = game_state['own_remaining']
        
        # 未清台时的特殊处理：标记黑8为"禁区"
        avoid_eight = (own_remaining > 0)
        
        if strategy == 'aggressive_clearance':
            # 积极清台：优先容易进球
            actions.extend(self._generate_easy_shots_first(cue_pos, balls, my_targets, table, avoid_eight))
            
        elif strategy == 'defensive':
            # 防守：安全球为主
            actions.extend(self._generate_smart_defense(cue_pos, balls, my_targets, table))
            
        elif strategy == 'mixed_endgame':
            # 尾局混合：既有进攻也有防守
            actions.extend(self._generate_endgame_shots(cue_pos, balls, my_targets, table, avoid_eight))
            
        elif strategy == 'eight_ball_only':
            # 只打黑8
            actions.extend(self._generate_eight_ball_shots(cue_pos, balls, table))
            
        else:  # 'balanced'
            # 平衡：生成多样化的动作
            actions.extend(self._generate_diversified_shots(cue_pos, balls, my_targets, table, avoid_eight))
        
        # 限制数量并去重
        if len(actions) > 30:
            # 随机抽样，保持多样性
            indices = random.sample(range(len(actions)), min(30, len(actions)))
            actions = [actions[i] for i in indices]
        
        return actions
    
    def _generate_easy_shots_first(self, cue_pos, balls, my_targets, table, avoid_eight):
        """优先容易进球的动作生成"""
        actions = []
        
        # 找到最容易的3个目标球（按进球难度排序）
        easy_targets = self._find_easiest_targets(cue_pos, balls, my_targets, table, n=3)
        
        for target_id, difficulty, pocket_id in easy_targets:
            if difficulty > 0.7:  # 太难了，跳过
                continue
                
            target_pos = balls[target_id].state.rvw[0]
            pocket_pos = table.pockets[pocket_id].center
            
            # 计算击球参数
            phi = self._calculate_ghost_angle(cue_pos, target_pos, pocket_pos)
            distance = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
            
            # 根据难度调整力度（容易的球用中等力度）
            if difficulty < 0.3:
                power = 2.5 + distance * 0.8  # 中等偏上
            else:
                power = 2.0 + distance * 1.0  # 标准
            
            power = np.clip(power, 1.5, 4.5)
            
            # 基础动作
            actions.append({
                'V0': power,
                'phi': phi,
                'theta': 0,
                'a': 0,
                'b': 0,
                'strategy': 'easy_shot',
                'target': target_id
            })
            
            # 添加一个轻微旋转变种（提高稳定性）
            if distance > 0.5:
                actions.append({
                    'V0': power * 0.95,
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': -0.08,
                    'strategy': 'easy_shot_with_draw'
                })
        
        # 如果没有容易进球，生成一些中等难度的
        if len(actions) < 5:
            medium_targets = self._find_medium_targets(cue_pos, balls, my_targets, table, n=5)
            for target_id, _, pocket_id in medium_targets:
                target_pos = balls[target_id].state.rvw[0]
                pocket_pos = table.pockets[pocket_id].center
                phi = self._calculate_ghost_angle(cue_pos, target_pos, pocket_pos)
                distance = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
                
                actions.append({
                    'V0': 3.0 + distance * 0.5,
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': 0,
                    'strategy': 'medium_shot'
                })
        
        return actions
    
    def _generate_smart_defense(self, cue_pos, balls, my_targets, table):
        """智能防守：不是单纯的安全球，而是制造麻烦"""
        actions = []
        
        # 1. 将白球藏在对方球后面
        snooker_actions = self._generate_snooker_shots(cue_pos, balls, my_targets, table)
        actions.extend(snooker_actions)
        
        # 2. 将黑8移动到安全位置（如果未清台）
        own_remaining = sum(1 for bid in my_targets 
                           if bid in balls and balls[bid].state.s != 4)
        if own_remaining > 0 and '8' in balls:
            eight_pos = balls['8'].state.rvw[0]
            safety_actions = self._move_eight_to_safety(cue_pos, eight_pos, table, balls)
            actions.extend(safety_actions)
        
        # 3. 将对方关键球打到难打的位置
        opponent_actions = self._hinder_opponent(cue_pos, balls, my_targets, table)
        actions.extend(opponent_actions)
        
        # 4. 基础安全球：白球回到开球区
        back_to_baulk = self._return_to_baulk(cue_pos, table)
        if back_to_baulk:
            actions.append(back_to_baulk)
        
        return actions
    
    def _generate_snooker_shots(self, cue_pos, balls, my_targets, table):
        """制造斯诺克（障碍球）"""
        actions = []
        
        # 找到对方的目标球
        all_balls = set(['1','2','3','4','5','6','7','9','10','11','12','13','14','15'])
        opponent_targets = [bid for bid in all_balls if bid not in my_targets]
        opponent_balls = [bid for bid in opponent_targets 
                         if bid in balls and balls[bid].state.s != 4]
        
        if not opponent_balls:
            return actions
        
        # 随机选择一个对方球作为障碍
        obstacle_id = random.choice(opponent_balls)
        obstacle_pos = balls[obstacle_id].state.rvw[0]
        
        # 计算将白球打到这个球后面的位置
        # 方法：计算从这个球到各袋口的反方向
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            # 从球到袋口的向量
            vec_to_pocket = np.array(pocket_pos) - np.array(obstacle_pos)
            # 反方向（藏到球后面）
            hide_direction = -vec_to_pocket / np.linalg.norm(vec_to_pocket)
            hide_distance = 0.3  # 藏到30cm后面
            hide_pos = np.array(obstacle_pos) + hide_direction * hide_distance
            
            # 确保位置在台内
            hide_pos[0] = np.clip(hide_pos[0], 0.1, 2.74)
            hide_pos[1] = np.clip(hide_pos[1], 0.1, 1.37)
            
            # 计算击打这个位置的参数
            vec_to_hide = hide_pos - np.array(cue_pos)
            phi = math.degrees(math.atan2(vec_to_hide[1], vec_to_hide[0])) % 360
            distance = np.linalg.norm(vec_to_hide)
            
            actions.append({
                'V0': min(3.0, 1.5 + distance * 0.5),
                'phi': phi,
                'theta': 0,
                'a': 0,
                'b': 0,
                'strategy': 'snooker',
                'obstacle': obstacle_id
            })
        
        return actions[:3]  # 最多3个
    
    def _hinder_opponent(self, cue_pos, balls, my_targets, table):
        """妨碍对手：将对方关键球打到难打的位置"""
        actions = []
        
        # 找到对方最容易进的球，把它打走
        all_balls = set(['1','2','3','4','5','6','7','9','10','11','12','13','14','15'])
        opponent_targets = [bid for bid in all_balls if bid not in my_targets]
        
        for target_id in opponent_targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
                
            target_pos = balls[target_id].state.rvw[0]
            
            # 检查这个球是否在容易进球的位置
            easiest_pocket = self._find_easiest_pocket_for_ball(target_pos, table)
            if easiest_pocket:
                pocket_pos = table.pockets[easiest_pocket].center
                dist_to_pocket = np.linalg.norm(np.array(target_pos) - np.array(pocket_pos))
                
                if dist_to_pocket < 0.4:  # 离袋口太近，危险！
                    # 把这个球打走
                    # 计算打到远离袋口的方向
                    vec_away = np.array(target_pos) - np.array(pocket_pos)
                    if np.linalg.norm(vec_away) < 0.01:
                        vec_away = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0])
                    
                    vec_away = vec_away / np.linalg.norm(vec_away)
                    target_hit_point = np.array(target_pos) + vec_away * self.ball_radius
                    
                    # 计算从白球到这个击打点的参数
                    vec_cue_to_hit = target_hit_point - np.array(cue_pos)
                    phi = math.degrees(math.atan2(vec_cue_to_hit[1], vec_cue_to_hit[0])) % 360
                    distance = np.linalg.norm(vec_cue_to_hit)
                    
                    actions.append({
                        'V0': min(4.0, 2.0 + distance * 0.8),
                        'phi': phi,
                        'theta': 0,
                        'a': 0,
                        'b': 0,
                        'strategy': 'hinder_opponent',
                        'target': target_id
                    })
                    
                    if len(actions) >= 2:  # 最多妨碍2个球
                        break
        
        return actions
    
    def _find_easiest_targets(self, cue_pos, balls, my_targets, table, n=5):
        """找到最容易进的n个目标球，返回(球ID, 难度分数, 袋口ID)列表"""
        results = []
        
        for target_id in my_targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
                
            target_pos = balls[target_id].state.rvw[0]
            
            # 为这个球找到最佳袋口
            best_pocket, difficulty = self._find_best_pocket_with_difficulty(
                cue_pos, target_pos, table, balls
            )
            
            if best_pocket:
                results.append((target_id, difficulty, best_pocket))
        
        # 按难度排序（难度越小越容易）
        results.sort(key=lambda x: x[1])
        return results[:n]
    
    def _find_best_pocket_with_difficulty(self, cue_pos, target_pos, table, balls):
        """找到最佳袋口并计算进球难度（0-1，0最容易）"""
        best_pocket = None
        best_difficulty = float('inf')
        
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            
            # 计算理论进球角度
            phi = self._calculate_ghost_angle(cue_pos, target_pos, pocket_pos)
            
            # 难度因素1：白球到目标球的距离
            dist_cue_to_target = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
            
            # 难度因素2：目标球到袋口的距离
            dist_target_to_pocket = np.linalg.norm(np.array(target_pos) - np.array(pocket_pos))
            
            # 难度因素3：角度是否正（正对袋口为0度，角度越大越难）
            # 计算入射角
            vec_target_to_pocket = np.array(pocket_pos) - np.array(target_pos)
            vec_cue_to_target = np.array(target_pos) - np.array(cue_pos)
            
            if np.linalg.norm(vec_target_to_pocket) < 0.01 or np.linalg.norm(vec_cue_to_target) < 0.01:
                angle_factor = 1.0
            else:
                dot = np.dot(vec_target_to_pocket[:2], vec_cue_to_target[:2])
                norms = np.linalg.norm(vec_target_to_pocket[:2]) * np.linalg.norm(vec_cue_to_target[:2])
                cos_angle = dot / (norms + 1e-6)
                angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                angle_factor = min(1.0, angle / 45.0)  # 45度以内认为可接受
            
            # 难度因素4：是否有障碍
            has_obstacle = self._check_obstacle(cue_pos, target_pos, pocket_pos, balls)
            obstacle_factor = 1.5 if has_obstacle else 1.0
            
            # 综合难度
            difficulty = (dist_cue_to_target * 0.3 + 
                         dist_target_to_pocket * 0.3 + 
                         angle_factor * 0.3) * obstacle_factor
            
            if difficulty < best_difficulty:
                best_difficulty = difficulty
                best_pocket = pocket_id
        
        # 归一化难度到0-1范围（假设最大难度为3.0）
        normalized_difficulty = min(1.0, best_difficulty / 3.0)
        
        return best_pocket, normalized_difficulty
    
    def _quick_evaluation(self, actions, balls, my_targets, table):
        """快速评估动作，选择最佳"""
        if not actions:
            return self._random_action()
        
        # 简单评估：基于启发式规则
        best_action = None
        best_score = -float('inf')
        
        for action in actions[:10]:  # 只评估前10个
            score = self._estimate_action_score(action, balls, my_targets, table)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        # 如果没有合适的，选择第一个
        if not best_action:
            best_action = actions[0]
        
        print(f"[BalancedAgent] 选择策略: {best_action.get('strategy', 'unknown')}, 预估得分: {best_score:.2f}")
        
        return best_action
    
    def _estimate_action_score(self, action, balls, my_targets, table):
        """估计动作的得分（基于启发式，不实际模拟）"""
        score = 0
        
        # 基本分
        strategy = action.get('strategy', '')
        if strategy == 'easy_shot':
            score += 80
        elif strategy == 'medium_shot':
            score += 50
        elif strategy == 'snooker':
            score += 60
        elif strategy == 'hinder_opponent':
            score += 40
        elif strategy == 'eight_ball_only':
            score += 100
        
        # 力度合理性（中等力度最佳）
        v0 = action.get('V0', 3.0)
        if 2.0 <= v0 <= 4.0:
            score += 20
        elif v0 > 6.0:
            score -= 30  # 太大力
        
        # 旋转合理性
        b = action.get('b', 0)
        if abs(b) > 0.3:
            score -= 10  # 过多旋转增加不确定性
        
        return score
```

## 2. **专门针对 basic_agent_pro 弱点的攻击策略**

```python
class AntiBasicAgentPro(Agent):
    """专门针对 basic_agent_pro 弱点的Agent"""
    
    def __init__(self):
        super().__init__()
        
        # basic_agent_pro 的弱点分析：
        # 1. 倾向于打容易的球（近距离、正对袋口）
        # 2. 防守能力弱（基本只有进攻）
        # 3. 对复杂局面处理能力有限
        
        # 我们的对策：
        self.strategies = {
            'steal_easy_shots': 0.4,      # 抢走容易的球
            'create_complexity': 0.3,     # 制造复杂局面
            'force_mistakes': 0.2,        # 迫使对方犯错
            'defensive_when_ahead': 0.1,  # 领先时防守
        }
    
    def decision(self, balls, my_targets, table):
        # 分析局面：我们是否领先？
        own_remaining = sum(1 for bid in my_targets 
                           if bid in balls and balls[bid].state.s != 4)
        all_balls = set(['1','2','3','4','5','6','7','9','10','11','12','13','14','15'])
        opponent_targets = [bid for bid in all_balls if bid not in my_targets]
        opponent_remaining = sum(1 for bid in opponent_targets 
                                if bid in balls and balls[bid].state.s != 4)
        
        is_ahead = (own_remaining < opponent_remaining)
        
        if is_ahead and own_remaining <= 3:
            # 领先且接近胜利：保守但积极的策略
            return self._winning_strategy(balls, my_targets, table)
        else:
            # 均势或落后：积极进攻，抢走容易的球
            return self._aggressive_steal_strategy(balls, my_targets, table)
    
    def _aggressive_steal_strategy(self, balls, my_targets, table):
        """积极抢球策略：专门打 basic_agent_pro 想打的球"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return self._random_action()
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 1. 找到所有容易进的球（包括对方的！）
        all_easy_shots = self._find_all_easy_shots(cue_pos, balls, table)
        
        # 2. 优先打己方的容易球
        my_easy_shots = [shot for shot in all_easy_shots if shot['ball_id'] in my_targets]
        
        if my_easy_shots:
            # 选择最容易的那个
            easiest = min(my_easy_shots, key=lambda x: x['difficulty'])
            return self._generate_shot_for_target(
                cue_pos, 
                balls[easiest['ball_id']].state.rvw[0],
                table.pockets[easiest['pocket_id']].center,
                power_factor=0.9  # 稍微轻一点，确保稳定
            )
        
        # 3. 如果没有己方的容易球，考虑打对方的容易球（制造混乱）
        opponent_easy_shots = [shot for shot in all_easy_shots if shot['ball_id'] not in my_targets]
        
        if opponent_easy_shots:
            # 选择对方的容易球，把它打走或打进
            target_shot = random.choice(opponent_easy_shots[:2])
            target_pos = balls[target_shot['ball_id']].state.rvw[0]
            
            # 有两种选择：
            if random.random() < 0.7:
                # a. 打进这个球（给对方制造麻烦）
                return self._generate_shot_for_target(
                    cue_pos, target_pos,
                    table.pockets[target_shot['pocket_id']].center,
                    power_factor=1.1  # 稍微重一点，确保进
                )
            else:
                # b. 把这个球打走（让它更难打）
                return self._generate_disrupt_shot(cue_pos, target_pos, table, balls)
        
        # 4. 都没有：正常进攻
        return self._normal_offensive_shot(balls, my_targets, table)
    
    def _find_all_easy_shots(self, cue_pos, balls, table):
        """找到台面上所有容易进的球（无论敌我）"""
        easy_shots = []
        
        for ball_id, ball in balls.items():
            if ball_id == 'cue' or ball.state.s == 4:
                continue
                
            ball_pos = ball.state.rvw[0]
            
            # 检查各个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算难度
                difficulty = self._calculate_shot_difficulty(
                    cue_pos, ball_pos, pocket_pos, balls
                )
                
                if difficulty < 0.4:  # 容易球
                    easy_shots.append({
                        'ball_id': ball_id,
                        'pocket_id': pocket_id,
                        'difficulty': difficulty,
                        'distance': np.linalg.norm(np.array(cue_pos) - np.array(ball_pos))
                    })
        
        # 按难度排序
        easy_shots.sort(key=lambda x: x['difficulty'])
        return easy_shots
    
    def _generate_disrupt_shot(self, cue_pos, target_pos, table, balls):
        """制造混乱的击球：把球打到难打的位置"""
        # 找到最难打到的袋口（距离最远）
        worst_pocket = None
        max_distance = 0
        
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            distance = np.linalg.norm(np.array(target_pos) - np.array(pocket_pos))
            if distance > max_distance:
                max_distance = distance
                worst_pocket = pocket_id
        
        if worst_pocket:
            pocket_pos = table.pockets[worst_pocket].center
            
            # 不是直接瞄准袋口，而是把球打到那个方向
            vec_to_worst = np.array(pocket_pos) - np.array(target_pos)
            vec_to_worst = vec_to_worst / np.linalg.norm(vec_to_worst)
            
            # 击打点稍微偏离中心，制造旋转
            hit_point = np.array(target_pos) + vec_to_worst * self.ball_radius
            
            vec_cue_to_hit = hit_point - np.array(cue_pos)
            phi = math.degrees(math.atan2(vec_cue_to_hit[1], vec_cue_to_hit[0])) % 360
            distance = np.linalg.norm(vec_cue_to_hit)
            
            return {
                'V0': min(4.5, 2.5 + distance * 0.7),
                'phi': phi,
                'theta': 0,
                'a': 0,
                'b': 0.1,  # 轻微推杆
                'strategy': 'disrupt'
            }
        
        return self._random_action()
```

## 3. **集成到你的AdvancedPoolAI中**

```python
class CompetitiveAdvancedPoolAI(AdvancedPoolAI):
    """竞技版AdvancedPoolAI，专门针对对手弱点"""
    
    def __init__(self, mode='adaptive'):
        super().__init__(mode)
        
        # 添加专门针对basic_agent_pro的策略
        self.anti_basic_agent = AntiBasicAgentPro()
        self.balanced_agent = BalancedAgent(risk_tolerance=0.6)
        
        # 记录对手表现
        self.opponent_performance = {
            'clearance_rate': 0.5,  # 清台率
            'mistake_rate': 0.2,    # 犯错率
            'aggressiveness': 0.7,  # 进攻性
        }
        
        # 自适应策略选择
        self.current_strategy = 'balanced'
    
    def decision(self, balls=None, my_targets=None, table=None):
        # 根据对手表现调整策略
        self._adapt_to_opponent()
        
        # 选择策略
        if self.current_strategy == 'anti_basic':
            return self.anti_basic_agent.decision(balls, my_targets, table)
        elif self.current_strategy == 'balanced':
            return self.balanced_agent.decision(balls, my_targets, table)
        else:
            # 默认使用父类的智能决策
            return super().decision(balls, my_targets, table)
    
    def _adapt_to_opponent(self):
        """根据对手表现调整策略"""
        # 如果对手清台率高，我们需要更防守
        if self.opponent_performance['clearance_rate'] > 0.7:
            self.current_strategy = 'anti_basic'
            print("[Adaptive] 对手清台率高，启用针对性策略")
        
        # 如果对手犯错率高，我们可以更激进
        elif self.opponent_performance['mistake_rate'] > 0.3:
            self.current_strategy = 'balanced'
            self.balanced_agent.risk_tolerance = 0.8  # 提高风险容忍度
            print("[Adaptive] 对手犯错多，启用激进策略")
        
        else:
            self.current_strategy = 'balanced'
            self.balanced_agent.risk_tolerance = 0.6  # 默认
    
    def update_opponent_performance(self, opponent_shot_result):
        """更新对手表现数据"""
        # opponent_shot_result 应该是一个字典，包含：
        # {'balls_pocketed': [], 'foul': False, 'turn_continued': True}
        
        # 简化更新逻辑
        if opponent_shot_result.get('foul', False):
            self.opponent_performance['mistake_rate'] = (
                0.9 * self.opponent_performance['mistake_rate'] + 0.1
            )
        else:
            self.opponent_performance['mistake_rate'] *= 0.95
        
        # 如果对手连续进球，提高清台率估计
        balls_pocketed = opponent_shot_result.get('balls_pocketed', [])
        if len(balls_pocketed) > 0:
            self.opponent_performance['clearance_rate'] = (
                0.9 * self.opponent_performance['clearance_rate'] + 0.1
            )
        else:
            self.opponent_performance['clearance_rate'] *= 0.95
```

## 4. **快速起效的关键修改**

如果你只需要几个立即有效的修改：

```python
# 在你的AdvancedPoolAI的decision方法中添加：
def decision(self, balls=None, my_targets=None, table=None):
    # 在开头添加进攻性检查
    cue_ball = balls.get('cue')
    if cue_ball:
        cue_pos = cue_ball.state.rvw[0]
        
        # 关键1：如果有非常容易的球，优先打
        very_easy = self._find_very_easy_shot(cue_pos, balls, my_targets, table)
        if very_easy:
            print(f"[QUICK WIN] 发现极容易球: {very_easy['target']}")
            return very_easy['action']
    
    # 关键2：根据球权决定策略
    # 如果是我们的回合，且对方剩球少，要积极进攻
    own_remaining = sum(1 for bid in my_targets 
                       if bid in balls and balls[bid].state.s != 4)
    all_balls = set(['1','2','3','4','5','6','7','9','10','11','12','13','14','15'])
    opponent_targets = [bid for bid in all_balls if bid not in my_targets]
    opponent_remaining = sum(1 for bid in opponent_targets 
                            if bid in balls and balls[bid].state.s != 4)
    
    if own_remaining < opponent_remaining:
        # 我们领先：可以稍微保守
        risk_factor = 0.4
    elif own_remaining == opponent_remaining:
        # 平局：平衡
        risk_factor = 0.6
    else:
        # 落后：必须激进
        risk_factor = 0.8
    
    # 修改MCTS的探索参数
    self.c_puct = 0.5 + risk_factor  # 更激进时减少探索
    
    # 继续原有逻辑...
```

## **策略建议：**

1. **先测试BalancedAgent**：它的平衡策略应该能立即改善表现
2. **针对弱点**：使用AntiBasicAgentPro的策略专门针对basic_agent_pro
3. **动态调整**：根据比赛进程实时调整策略

关键是要在**避免黑8犯规**和**积极进攻**之间找到平衡点。basic_agent_pro的弱点是防守能力差，所以我们可以：
- 抢走所有容易的球
- 制造复杂局面
- 在领先时适当防守

这样应该能显著提高胜率。