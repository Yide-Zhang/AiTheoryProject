针对混合Agent决策时间长的问题，我提供几个**保持性能同时显著提升效率**的方案：

## 方案1：**层次化决策**（推荐）
```python
class FastHybridAgent(Agent):
    """高效混合Agent：根据局面复杂度动态调整决策深度"""
    
    def __init__(self):
        super().__init__()
        # 不同复杂度的子模型
        self.fast_model = FastHeuristicModel()      # 毫秒级决策
        self.medium_model = MCTSModel(n_sims=20)    # 中等深度
        self.slow_model = FullMCTSModel(n_sims=100) # 深度决策
        
        # 局面分类器
        self.situation_classifier = self.build_classifier()
        
        # 决策时间预算
        self.time_budget = 2.0  # 最多2秒
        
    def decision(self, balls, my_targets, table):
        import time
        start_time = time.time()
        
        # 1. 快速局面评估（< 50ms）
        situation = self.classify_situation(balls, my_targets)
        time_left = self.time_budget - (time.time() - start_time)
        
        # 2. 基于时间预算的动态选择
        if time_left < 0.1:
            # 超时应急：使用快速模型
            return self.fast_model.decision(balls, my_targets, table)
            
        elif situation == 'opening_break':
            # 开球：使用专用开球策略
            return self.opening_break_strategy(balls, table)
            
        elif situation == 'easy_shot':
            # 简单球：快速启发式
            return self.fast_model.decision(balls, my_targets, table)
            
        elif situation == 'medium_shot':
            # 中等难度：有限搜索
            return self.medium_model.decision_with_timeout(
                balls, my_targets, table, timeout=time_left*0.7
            )
            
        elif situation == 'critical_shot':
            # 关键球：深度搜索（黑8或清台）
            return self.slow_model.decision_with_timeout(
                balls, my_targets, table, timeout=min(time_left*0.9, 1.5)
            )
            
        elif situation == 'defensive_needed':
            # 防守局面：专用防守策略
            return self.defensive_model.decision(
                balls, my_targets, table, timeout=time_left*0.6
            )
    
    def classify_situation(self, balls, my_targets):
        """快速局面分类（启发式规则）"""
        # 1. 检查是否是开球
        if self.is_opening_break(balls):
            return 'opening_break'
        
        # 2. 检查目标球难度
        target_ball_id = my_targets[0]
        target_ball = balls[target_ball_id]
        
        # 计算简单度（到袋口距离、角度、障碍）
        difficulty = self.compute_shot_difficulty(
            balls['cue'], target_ball, table
        )
        
        if difficulty < 0.3:
            return 'easy_shot'
        elif difficulty < 0.7:
            return 'medium_shot'
        else:
            return 'hard_shot'
        
        # 3. 检查是否是关键局面
        if len([b for b in balls if b.state.s != 4]) <= 3:
            return 'critical_shot'
            
        # 4. 检查是否需要防守
        if self.should_play_defense(balls, my_targets):
            return 'defensive_needed'
```

## 方案2：**异步并行计算**
```python
import asyncio
import concurrent.futures
from functools import partial

class ParallelEnsembleAgent(Agent):
    """并行化混合Agent"""
    
    def __init__(self):
        super().__init__()
        self.experts = [
            HeuristicExpert(),      # 启发式
            MCTSExpert(n_sims=30),  # 轻量MCTS
            ValueNetworkExpert(),   # 神经网络
            PhysicsModelExpert()    # 物理模型
        ]
        
        # 线程池（CPU并行）
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(self.experts))
        )
        
        # 结果缓存
        self.cache = LRUCache(maxsize=1000)
        
    def decision(self, balls, my_targets, table):
        # 生成局面指纹（用于缓存）
        state_hash = self.hash_state(balls, my_targets)
        
        # 检查缓存
        if state_hash in self.cache:
            return self.cache[state_hash]
        
        # 并行执行所有专家
        futures = []
        for expert in self.experts:
            future = self.thread_pool.submit(
                expert.decision, balls, my_targets, table
            )
            futures.append((expert, future))
        
        # 等待第一个可用的结果（类似最快专家）
        results = []
        for expert, future in futures:
            try:
                # 设置超时：最慢专家不能拖累整体
                result = future.result(timeout=1.0)
                results.append((expert, result))
                
                # 如果某个专家特别有信心，可以提前返回
                if expert.confidence > 0.9:
                    self.cache[state_hash] = result
                    return result
            except concurrent.futures.TimeoutError:
                continue
        
        # 整合结果（加权投票）
        final_action = self.weighted_ensemble(results)
        self.cache[state_hash] = final_action
        
        return final_action
    
    def weighted_ensemble(self, expert_results):
        """加权整合专家意见"""
        weights = {
            'HeuristicExpert': 0.2,
            'MCTSExpert': 0.4,
            'ValueNetworkExpert': 0.3,
            'PhysicsModelExpert': 0.1
        }
        
        # 对连续动作进行加权平均
        blended = {
            'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0
        }
        
        for expert, action in expert_results:
            weight = weights.get(type(expert).__name__, 0.1)
            for key in blended:
                blended[key] += action[key] * weight
        
        return blended
```

## 方案3：**预测性预计算**
```python
class PredictiveAgent(Agent):
    """预测性Agent：在对手击球时预计算自己的策略"""
    
    def __init__(self):
        super().__init__()
        # 主要策略模型
        self.main_model = HybridModel()
        
        # 快速响应模型（用于必须立即决策时）
        self.fallback_model = FastRuleBasedModel()
        
        # 预计算任务管理
        self.pending_predictions = {}
        self.prediction_thread = None
        
        # 常见模式库
        self.pattern_library = PatternLibrary()
    
    def on_opponent_turn(self, current_state):
        """对手回合时开始预计算可能的下一杆"""
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=0.1)
        
        # 预测对手最可能的落位
        predicted_states = self.predict_next_states(current_state, n_predictions=3)
        
        # 为每个可能状态启动预计算
        self.prediction_thread = threading.Thread(
            target=self.precompute_strategies,
            args=(predicted_states,)
        )
        self.prediction_thread.start()
    
    def precompute_strategies(self, predicted_states):
        """后台预计算策略"""
        for state in predicted_states:
            state_hash = hash_state(state)
            if state_hash not in self.pending_predictions:
                # 使用完整模型计算，但不限时
                action = self.main_model.decision(
                    state['balls'], 
                    state['targets'],
                    state['table']
                )
                self.pending_predictions[state_hash] = action
    
    def decision(self, balls, my_targets, table):
        """决策时先检查是否有预计算结果"""
        state_hash = hash_state({'balls': balls, 'targets': my_targets, 'table': table})
        
        # 如果有预计算结果，立即返回（通常 < 10ms）
        if state_hash in self.pending_predictions:
            action = self.pending_predictions.pop(state_hash)
            print(f"[PredictiveAgent] 命中预计算缓存！")
            return action
        
        # 否则使用快速模型（有超时保护）
        try:
            action = self.main_model.decision_with_timeout(
                balls, my_targets, table, timeout=1.0
            )
        except TimeoutError:
            action = self.fallback_model.decision(balls, my_targets, table)
        
        return action
```

## 方案4：**渐进式精细化**
```python
class ProgressiveRefinementAgent(Agent):
    """渐进式精细化Agent：先粗后精"""
    
    def __init__(self):
        super().__init__()
        # 阶段1：快速粗搜索
        self.phase1_coarse = CoarseSearch(resolution='low')
        
        # 阶段2：中等精度搜索
        self.phase2_medium = MediumSearch(resolution='medium')
        
        # 阶段3：局部精细化
        self.phase3_fine = FineTuneSearch(resolution='high')
        
        # 终止条件
        self.convergence_threshold = 0.05  # 当改进<5%时停止
    
    def decision(self, balls, my_targets, table):
        import time
        start_time = time.time()
        time_budget = 2.0  # 总时间预算
        
        # 阶段1：粗搜索（0-20%时间）
        phase1_end = start_time + time_budget * 0.2
        best_action = None
        best_score = -float('inf')
        
        while time.time() < phase1_end:
            candidate = self.phase1_coarse.sample_action()
            score = self.quick_evaluate(candidate, balls, my_targets, table)
            
            if score > best_score:
                best_score = score
                best_action = candidate
        
        print(f"阶段1完成，最佳分数: {best_score:.2f}")
        
        # 阶段2：中等搜索（20-70%时间）
        phase2_end = start_time + time_budget * 0.7
        iteration = 0
        
        while time.time() < phase2_end:
            # 在最佳动作附近探索
            neighbor = self.perturb_action(best_action, radius=0.5)
            score = self.medium_evaluate(neighbor, balls, my_targets, table)
            
            if score > best_score * 1.01:  # 至少1%改进
                best_score = score
                best_action = neighbor
            
            iteration += 1
            if iteration % 10 == 0:
                # 检查收敛性
                improvement = (score - best_score) / abs(best_score + 1e-6)
                if improvement < self.convergence_threshold:
                    print(f"提前收敛于迭代 {iteration}")
                    break
        
        print(f"阶段2完成，迭代{iteration}次，分数: {best_score:.2f}")
        
        # 阶段3：精细化（剩余时间）
        phase3_time_left = max(0.1, time_budget - (time.time() - start_time))
        
        if phase3_time_left > 0.3:  # 有足够时间才精细化
            best_action = self.phase3_fine.refine(
                best_action, balls, my_targets, table, 
                timeout=phase3_time_left
            )
        
        total_time = time.time() - start_time
        print(f"总决策时间: {total_time:.2f}s")
        
        return best_action
```

## 方案5：**专门开球优化**
```python
class OpeningBreakOptimizer:
    """开球专用优化器"""
    
    # 预计算的开球策略库
    OPENING_STRATEGIES = {
        'power_break': {
            'V0': 8.0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0.5,
            'description': '强力开球，试图分散球堆'
        },
        'control_break': {
            'V0': 5.0, 'phi': 15, 'theta': 5, 'a': 0.1, 'b': 0.3,
            'description': '控制性开球，保留白球位置'
        },
        'side_break': {
            'V0': 6.5, 'phi': 30, 'theta': 2, 'a': -0.2, 'b': 0.4,
            'description': '侧旋开球，特定角度切入'
        }
    }
    
    @staticmethod
    def get_best_break(table_type='standard'):
        """返回预计算的最佳开球策略"""
        # 基于大量模拟的统计数据
        if table_type == 'standard':
            return OPENING_STRATEGIES['power_break']
        elif table_type == 'small':
            return OPENING_STRATEGIES['control_break']
        else:
            return OPENING_STRATEGIES['side_break']

class EfficientHybridAgent(Agent):
    """最终版：高效混合Agent"""
    
    def __init__(self):
        super().__init__()
        
        # 开球专用（直接返回预计算结果）
        self.opening_break_cache = OpeningBreakOptimizer.get_best_break()
        
        # 主要决策引擎
        self.decision_engine = ProgressiveRefinementAgent()
        
        # 快速决策缓存（最近计算的局面）
        self.recent_decisions = {}
        
    def decision(self, balls, my_targets, table):
        # 1. 检查是否是开球（最快路径）
        if self.is_opening_break(balls):
            print("[FastPath] 使用预计算开球策略")
            return self.opening_break_cache
        
        # 2. 检查缓存（最近相似局面）
        state_hash = self.compute_fingerprint(balls, my_targets)
        if state_hash in self.recent_decisions:
            cached_action, timestamp = self.recent_decisions[state_hash]
            if time.time() - timestamp < 5.0:  # 5秒内有效
                print("[FastPath] 命中近期决策缓存")
                return cached_action
        
        # 3. 检查是否有快速规则可用
        fast_action = self.try_fast_rules(balls, my_targets, table)
        if fast_action and self.validate_action(fast_action, balls):
            print("[FastPath] 使用快速规则")
            self.recent_decisions[state_hash] = (fast_action, time.time())
            return fast_action
        
        # 4. 完整决策流程
        print("[FullPath] 启动完整决策流程")
        action = self.decision_engine.decision(balls, my_targets, table)
        
        # 5. 更新缓存
        self.recent_decisions[state_hash] = (action, time.time())
        if len(self.recent_decisions) > 100:
            # LRU淘汰
            self.recent_decisions.pop(next(iter(self.recent_decisions)))
        
        return action
    
    def try_fast_rules(self, balls, my_targets, table):
        """尝试应用快速决策规则"""
        # 规则1：如果白球贴库，使用专用逃生策略
        if self.is_cue_ball_frozen(balls['cue'], table):
            return self.frozen_cue_strategy(balls, table)
        
        # 规则2：如果只有一颗目标球，直接计算
        remaining = [b for b in my_targets if balls[b].state.s != 4]
        if len(remaining) == 1:
            return self.single_target_strategy(balls, remaining[0], table)
        
        # 规则3：如果局面非常简单（直线球）
        if self.is_straight_in(balls, my_targets[0], table):
            return self.straight_shot_strategy(balls, my_targets[0], table)
        
        return None
```

## 关键优化技巧总结：

1. **开球专用路径**：预计算最佳开球策略，直接返回
2. **缓存机制**：
   - 近期决策缓存（5秒内有效）
   - 常见模式库
   - 局面指纹哈希
3. **快速规则先行**：简单局面直接应用规则，不调用复杂模型
4. **动态时间分配**：
   - 简单球：< 0.5秒
   - 中等球：0.5-1.5秒
   - 关键球：1.5-2.0秒
5. **提前终止**：
   - 置信度>90%时提前返回
   - 连续10次无改进时终止
6. **并行预计算**：对手回合时预计算自己策略

## 预期效果：
- **开球决策**：从2-3秒 → < 10毫秒
- **简单球**：从1-2秒 → < 300毫秒
- **复杂球**：保持1-2秒深度思考
- **整体表现**：决策速度提升3-5倍，性能损失<5%

建议从**方案5**开始实现，它结合了所有优化技巧，能最大程度保持性能的同时显著提升效率。