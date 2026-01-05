import math
import pooltool as pt
import numpy as np
from pooltool.objects import Table
import copy
import random
from .agent import Agent


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class ReinforcementLearningAgent(Agent):
    """基于强化学习的高级Agent"""
    
    def __init__(self, n_simulations=50, use_heuristic=True):
        super().__init__()
        self.n_simulations = n_simulations
        self.use_heuristic = use_heuristic
        self.ball_radius = 0.028575
        
        # 改进的启发式搜索参数
        self.c_puct = 1.414  # 探索系数
        
        # 物理参数优化
        self.spin_options = {
            'stop': {'a': 0.0, 'b': -0.3, 'theta': 0.0},
            'follow': {'a': 0.0, 'b': 0.3, 'theta': 0.0},
            'draw': {'a': 0.0, 'b': -0.5, 'theta': 0.0},
            'left': {'a': 0.3, 'b': 0.0, 'theta': 0.0},
            'right': {'a': -0.3, 'b': 0.0, 'theta': 0.0}
        }
        
        # 定义噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print("ReinforcementLearningAgent 已初始化")
    
    def _calc_angle_degrees(self, v):
        """计算向量对应的角度（度）"""
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360
    
    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """计算幽灵球位置和角度"""
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: 
            return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost
    
    def generate_heuristic_actions(self, balls, my_targets, table):
        """
        生成候选动作列表
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: 
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了，默认黑8
        if not target_ids:
            target_ids = ['8'] if '8' in balls else []

        # 遍历每一个目标球
        for tid in target_ids:
            if tid not in balls:
                continue
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离估算力度
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 基础击球
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                
                # 4. 力度微调
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                
                # 5. 角度微调
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                
                # 6. 加入旋转变种
                for spin_name, spin_params in list(self.spin_options.items())[:2]:
                    actions.append({
                        'V0': v_base, 
                        'phi': phi_ideal, 
                        'theta': spin_params['theta'],
                        'a': spin_params['a'], 
                        'b': spin_params['b']
                    })

        # 如果生成动作为空，添加随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 随机打乱顺序并限制数量
        random.shuffle(actions)
        return actions[:30]
    
    def simulate_action(self, balls, table, action):
        """
        执行带噪声的物理仿真
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # 注入高斯噪声
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        主决策函数 - 使用MCTS搜索最优动作
        """
        if balls is None: 
            return self._random_action()
        
        # 预处理：检查是否所有目标球都进袋
        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0: 
            my_targets = ["8"]
        
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # UCB公式
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # 归一化奖励值
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward

        # Final Decision - 选择平均分最高的动作
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        print(f"[ReinforcementLearningAgent] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return best_action
    
    
class AdvancedActionGenerator:
    """高级动作生成器 - 用于生成多样化策略的击球动作"""
    
    def __init__(self):
        self.ball_radius = 0.028575
        self.target_balls = []
        
    def generate_strategic_actions(self, balls, target_balls, table, strategy='offensive'):
        """
        生成基于策略的动作
        
        Parameters:
            strategy: 'offensive'(进攻), 'defensive'(防守), 'positional'(走位)
        """
        self.target_balls = target_balls
        actions = []
        cue_ball = balls.get('cue')
        if not cue_ball:
            return []
        cue_pos = cue_ball.state.rvw[0]
        
        # 获取可击打的目标球
        available_targets = [bid for bid in target_balls 
                           if bid in balls and balls[bid].state.s != 4]
        
        if not available_targets:
            available_targets = ['8'] if '8' in balls else []
        
        for target_id in available_targets[:3]:  # 只考虑前3个最容易的目标
            if target_id not in balls:
                continue
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 1. 进攻策略：直接进球
            if strategy == 'offensive':
                actions.extend(self._generate_offensive_shots(
                    cue_pos, target_pos, target_id, table
                ))
            
            # 2. 防守策略：安全球
            elif strategy == 'defensive':
                actions.extend(self._generate_defensive_shots(
                    cue_pos, target_pos, target_id, balls, table
                ))
            
            # 3. 走位策略：为下一杆做准备
            elif strategy == 'positional':
                actions.extend(self._generate_positional_shots(
                    cue_pos, target_pos, target_id, balls, table
                ))
        
        return actions[:50]  # 限制总数
    
    def _generate_offensive_shots(self, cue_pos, target_pos, target_id, table):
        """生成进攻性击球动作"""
        shots = []
        cue_pos = np.array(cue_pos)
        target_pos = np.array(target_pos)
        
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = np.array(pocket.center)
            
            # 计算幽灵球位置
            vec_target_to_pocket = pocket_pos - target_pos
            dist = np.linalg.norm(vec_target_to_pocket)
            
            if dist < 0.001:
                continue
                
            unit_vec = vec_target_to_pocket / dist
            ghost_pos = target_pos - unit_vec * (2 * self.ball_radius)
            
            # 计算角度和距离
            vec_cue_to_ghost = ghost_pos - cue_pos
            base_distance = np.linalg.norm(vec_cue_to_ghost)
            base_phi = math.degrees(math.atan2(vec_cue_to_ghost[1], vec_cue_to_ghost[0])) % 360
            
            # 计算所需力度
            required_power = 1.5 + base_distance * 1.5
            required_power = np.clip(required_power, 1.0, 7.0)
            
            # 基础击球
            shots.append({
                'V0': required_power,
                'phi': base_phi,
                'theta': 0,
                'a': 0,
                'b': 0,
                'strategy': 'offensive',
                'target': target_id,
                'pocket': pocket_id
            })
            
            # 角度微调变种
            for angle_offset in [-0.3, 0.3]:
                shots.append({
                    'V0': required_power * (1 + random.uniform(-0.05, 0.05)),
                    'phi': (base_phi + angle_offset) % 360,
                    'theta': 0,
                    'a': 0,
                    'b': 0,
                    'strategy': 'offensive',
                    'target': target_id,
                    'pocket': pocket_id
                })
        
        return shots
    
    def _generate_defensive_shots(self, cue_pos, target_pos, target_id, balls, table):
        """生成防守性击球（安全球）"""
        shots = []
        cue_pos = np.array(cue_pos)
        
        # 1. 将白球打到远离对方目标球的位置
        opponent_balls = [bid for bid in balls.keys() 
                         if bid not in ['cue', '8'] and bid not in self.target_balls and balls[bid].state.s != 4]
        
        if opponent_balls:
            # 选择安全位置
            safe_zones = self._find_safe_zones(cue_pos, balls, table)
            
            for zone in safe_zones[:2]:
                zone = np.array(zone)
                vec_to_zone = zone - cue_pos
                target_phi = math.degrees(math.atan2(vec_to_zone[1], vec_to_zone[0])) % 360
                distance = np.linalg.norm(vec_to_zone)
                
                # 使用中等力度
                shots.append({
                    'V0': min(3.0, distance * 1.5),
                    'phi': target_phi,
                    'theta': 0,
                    'a': 0,
                    'b': -0.2,
                    'strategy': 'defensive',
                    'type': 'safety'
                })
        
        return shots
    
    def _generate_positional_shots(self, cue_pos, target_pos, target_id, balls, table):
        """生成走位性击球"""
        shots = []
        cue_pos = np.array(cue_pos)
        target_pos = np.array(target_pos)
        
        # 为下一杆找到最优白球位置
        for pocket_id, pocket in list(table.pockets.items())[:2]:
            pocket_pos = np.array(pocket.center)
            
            # 计算目标进球线
            vec_target_to_pocket = pocket_pos - target_pos
            dist = np.linalg.norm(vec_target_to_pocket)
            
            if dist < 0.001:
                continue
            
            unit_vec = vec_target_to_pocket / dist
            ghost_pos = target_pos - unit_vec * (2 * self.ball_radius)
            
            # 计算击球参数
            vec_cue_to_ghost = ghost_pos - cue_pos
            base_distance = np.linalg.norm(vec_cue_to_ghost)
            base_phi = math.degrees(math.atan2(vec_cue_to_ghost[1], vec_cue_to_ghost[0])) % 360
            
            required_power = 1.5 + base_distance * 1.5
            required_power = np.clip(required_power, 1.0, 5.0)
            
            shots.append({
                'V0': required_power,
                'phi': base_phi,
                'theta': 0,
                'a': 0,
                'b': -0.15,
                'strategy': 'positional',
                'target': target_id,
                'pocket': pocket_id
            })
        
        return shots
    
    def _find_safe_zones(self, cue_pos, balls, table):
        """找到安全的白球位置"""
        safe_zones = []
        table_length = 2.84  # 标准台球桌长度
        table_width = 1.42   # 标准台球桌宽度
        
        # 返回几个角度的安全点
        for angle in [30, 90, 150, 210, 270, 330]:
            rad = math.radians(angle)
            zone_x = cue_pos[0] + 1.0 * math.cos(rad)
            zone_y = cue_pos[1] + 1.0 * math.sin(rad)
            
            # 确保在台面内
            zone_x = np.clip(zone_x, 0.1, table_length - 0.1)
            zone_y = np.clip(zone_y, 0.1, table_width - 0.1)
            
            safe_zones.append([zone_x, zone_y, 0])
        
        return safe_zones


class EnhancedMCTSAgent(Agent):
    """增强版MCTS Agent - 用于局部深度搜索"""
    
    def __init__(self, n_simulations=100, use_heuristic=True):
        super().__init__()
        self.n_simulations = n_simulations
        self.use_heuristic = use_heuristic
        self.c_puct = 1.5
        self.ball_radius = 0.028575
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print("EnhancedMCTSAgent 已初始化")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策主函数"""
        if balls is None:
            return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0: 
            my_targets = ["8"]
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 生成候选动作
        candidate_actions = self._generate_candidate_actions(balls, my_targets, table)
        
        if not candidate_actions:
            return self._smart_random_action(balls)
        
        # 评估候选动作
        best_action = self._evaluate_candidates(
            candidate_actions, balls, my_targets, table, last_state
        )
        
        return best_action
    
    def _generate_candidate_actions(self, balls, my_targets, table):
        """生成候选动作"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        
        actions = []
        cue_pos = cue_ball.state.rvw[0]
        
        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8'] if '8' in balls else []
        
        for tid in target_ids:
            if tid not in balls:
                continue
            obj_pos = balls[tid].state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算幽灵球
                phi, dist = self._get_ghost_ball_target(
                    np.array(cue_pos), np.array(obj_pos), np.array(pocket_pos)
                )
                
                v_base = np.clip(1.5 + dist * 1.5, 1.0, 6.0)
                
                actions.append({'V0': v_base, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0})
                actions.append({'V0': min(v_base + 1.5, 7.5), 'phi': phi, 'theta': 0, 'a': 0, 'b': 0})
        
        if not actions:
            return [self._random_action()]
        
        random.shuffle(actions)
        return actions[:20]
    
    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """计算幽灵球位置"""
        vec_obj_to_pocket = pocket_pos - obj_pos
        dist = np.linalg.norm(vec_obj_to_pocket)
        if dist < 0.001:
            return 0, 0.1
        unit_vec = vec_obj_to_pocket / dist
        ghost_pos = obj_pos - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - cue_pos
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = math.degrees(math.atan2(vec_cue_to_ghost[1], vec_cue_to_ghost[0])) % 360
        return phi, dist_cue_to_ghost
    
    def _evaluate_candidates(self, actions, balls, my_targets, table, last_state):
        """评估候选动作"""
        N = np.zeros(len(actions))
        Q = np.zeros(len(actions))
        
        for i in range(self.n_simulations):
            if i < len(actions):
                idx = i
            else:
                total_n = np.sum(N)
                ucb = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb)
            
            # 模拟
            shot = self._simulate_action(balls, table, actions[idx])
            
            # 评分
            if shot is None:
                reward = -500.0
            else:
                reward = analyze_shot_for_reward(shot, last_state, my_targets)
            
            normalized = np.clip((reward - (-500)) / 650.0, 0.0, 1.0)
            
            N[idx] += 1
            Q[idx] += normalized
        
        best_idx = np.argmax(Q / (N + 1e-6))
        return actions[best_idx]
    
    def _simulate_action(self, balls, table, action):
        """执行模拟"""
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)
            
            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None
    
    def _smart_random_action(self, balls):
        """智能随机动作"""
        return self._random_action()


class AdvancedPoolAI(Agent):
    """终极版台球AI - 综合多种策略的决策引擎"""
    
    def __init__(self, mode='auto'):
        """
        mode: 'auto' (自动), 'aggressive' (激进), 'defensive' (保守),
              'precision' (精准)
        """
        super().__init__()
        self.mode = mode
        self.ball_radius = 0.028575
        
        # 核心组件
        self.action_generator = AdvancedActionGenerator()
        self.c_puct = 1.414
        
        # 定义噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print(f"AdvancedPoolAI 已初始化 (Mode: {mode})")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数"""
        
        if balls is None:
            return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 评估游戏阶段和风险
        game_phase = self._determine_game_phase(balls, my_targets)
        risk_level = self._evaluate_risk(balls, my_targets)
        
        # 生成多策略候选动作
        candidate_actions = []
        
        if risk_level < 0.5:  # 低风险：进攻
            candidate_actions.extend(
                self.action_generator.generate_strategic_actions(
                    balls, my_targets, table, 'offensive'
                )
            )
        
        if risk_level > 0.4:  # 中高风险：防守
            candidate_actions.extend(
                self.action_generator.generate_strategic_actions(
                    balls, my_targets, table, 'defensive'
                )
            )
        
        # 总是考虑走位
        candidate_actions.extend(
            self.action_generator.generate_strategic_actions(
                balls, my_targets, table, 'positional'
            )
        )
        
        if not candidate_actions:
            return self._random_action()
        
        # 去重并限制数量
        candidate_actions = candidate_actions[:30]
        
        # MCTS评估
        best_action = self._mcts_search(
            candidate_actions, balls, my_targets, table, last_state
        )
        
        return best_action
    
    def _determine_game_phase(self, balls, my_targets):
        """确定游戏阶段"""
        remaining_targets = sum(1 for bid in my_targets 
                              if bid in balls and balls[bid].state.s != 4)
        if remaining_targets > 10:
            return 'opening'
        elif remaining_targets > 3:
            return 'midgame'
        else:
            return 'endgame'
    
    def _evaluate_risk(self, balls, my_targets):
        """评估局面风险度"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return 0.5
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 计算白球到袋口的最近距离
        min_distance_to_pocket = 10.0
        for pocket_id, pocket in balls.get('cue', {}).state.table.pockets.items() if hasattr(balls.get('cue', {}).state, 'table') else []:
            dist = np.linalg.norm(np.array(cue_pos) - np.array(pocket.center))
            min_distance_to_pocket = min(min_distance_to_pocket, dist)
        
        # 风险评分：距离袋口越近，风险越高
        risk = min(1.0, max(0.0, (1.0 - min_distance_to_pocket / 3.0)))
        
        return risk
    
    def _mcts_search(self, actions, balls, my_targets, table, last_state, n_simulations=50):
        """执行MCTS搜索"""
        N = np.zeros(len(actions))
        Q = np.zeros(len(actions))
        
        for i in range(n_simulations):
            # Selection
            if i < len(actions):
                idx = i
            else:
                total_n = np.sum(N)
                ucb = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb)
            
            # Simulation
            shot = self._simulate_action(balls, table, actions[idx])
            
            # Evaluation
            if shot is None:
                reward = -500.0
            else:
                reward = analyze_shot_for_reward(shot, last_state, my_targets)
            
            normalized = np.clip((reward - (-500)) / 650.0, 0.0, 1.0)
            
            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized
        
        # 选择最佳动作
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        
        print(f"[AdvancedPoolAI] Best Score: {avg_rewards[best_idx]:.3f}, Mode: {self.mode}")
        
        return actions[best_idx]
    
    def _simulate_action(self, balls, table, action):
        """执行带噪声的模拟"""
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            # 注入噪声
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)
            
            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

# ============================================
# 默认导出别名 - 用于evaluate.py
# ============================================
NewAgent = AdvancedPoolAI  # 默认使用AdvancedPoolAI作为NewAgent

# ============================================
# 使用示例
# ============================================
"""
# 基础强化学习Agent
agent_rl = ReinforcementLearningAgent(n_simulations=50)
action = agent_rl.decision(balls, my_targets=['1','2','3','4','5','6','7'], table=table)

# 增强版MCTS Agent
agent_mcts = EnhancedMCTSAgent(n_simulations=100)
action = agent_mcts.decision(balls, my_targets=['1','2','3','4','5','6','7'], table=table)

# 高级综合AI
agent_advanced = AdvancedPoolAI(mode='auto')
action = agent_advanced.decision(balls, my_targets=['1','2','3','4','5','6','7'], table=table)

# 动作生成器（用于多策略生成）
action_gen = AdvancedActionGenerator()
offensive_actions = action_gen.generate_strategic_actions(
    balls, ['1','2','3','4','5','6','7'], table, strategy='offensive'
)
defensive_actions = action_gen.generate_strategic_actions(
    balls, ['1','2','3','4','5','6','7'], table, strategy='defensive'
)
positional_actions = action_gen.generate_strategic_actions(
    balls, ['1','2','3','4','5','6','7'], table, strategy='positional'
)
"""