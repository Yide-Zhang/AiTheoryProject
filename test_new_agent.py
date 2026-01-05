#!/usr/bin/env python3
"""
new_agent.py 快速测试脚本
验证所有Agent类都能正确初始化和使用
"""

import sys
import inspect

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 检查导入")
    print("=" * 60)
    try:
        from agents.agent import Agent
        print("✓ Agent 基类导入成功")
    except Exception as e:
        print(f"✗ Agent 基类导入失败: {e}")
        return False
    
    try:
        from agents.new_agent import (
            analyze_shot_for_reward,
            ReinforcementLearningAgent,
            AdvancedActionGenerator,
            EnhancedMCTSAgent,
            AdvancedPoolAI
        )
        print("✓ new_agent 模块导入成功")
        print("  - analyze_shot_for_reward")
        print("  - ReinforcementLearningAgent")
        print("  - AdvancedActionGenerator")
        print("  - EnhancedMCTSAgent")
        print("  - AdvancedPoolAI")
    except Exception as e:
        print(f"✗ new_agent 模块导入失败: {e}")
        return False
    
    return True


def test_class_initialization():
    """测试类初始化"""
    print("\n" + "=" * 60)
    print("测试2: 类初始化")
    print("=" * 60)
    
    try:
        from agents.new_agent import ReinforcementLearningAgent
        agent = ReinforcementLearningAgent(n_simulations=50)
        print(f"✓ ReinforcementLearningAgent 初始化成功")
        print(f"  - 模拟次数: {agent.n_simulations}")
        print(f"  - 球半径: {agent.ball_radius}")
    except Exception as e:
        print(f"✗ ReinforcementLearningAgent 初始化失败: {e}")
        return False
    
    try:
        from agents.new_agent import AdvancedActionGenerator
        generator = AdvancedActionGenerator()
        print(f"✓ AdvancedActionGenerator 初始化成功")
        print(f"  - 球半径: {generator.ball_radius}")
    except Exception as e:
        print(f"✗ AdvancedActionGenerator 初始化失败: {e}")
        return False
    
    try:
        from agents.new_agent import EnhancedMCTSAgent
        agent = EnhancedMCTSAgent(n_simulations=100)
        print(f"✓ EnhancedMCTSAgent 初始化成功")
        print(f"  - 模拟次数: {agent.n_simulations}")
    except Exception as e:
        print(f"✗ EnhancedMCTSAgent 初始化失败: {e}")
        return False
    
    try:
        from agents.new_agent import AdvancedPoolAI
        agent = AdvancedPoolAI(mode='auto')
        print(f"✓ AdvancedPoolAI 初始化成功")
        print(f"  - 模式: {agent.mode}")
    except Exception as e:
        print(f"✗ AdvancedPoolAI 初始化失败: {e}")
        return False
    
    return True


def test_method_signatures():
    """测试方法签名"""
    print("\n" + "=" * 60)
    print("测试3: 方法签名验证")
    print("=" * 60)
    
    try:
        from agents.new_agent import ReinforcementLearningAgent
        agent = ReinforcementLearningAgent()
        
        # 检查重要方法
        methods = ['decision', 'generate_heuristic_actions', 'simulate_action', '_random_action']
        for method_name in methods:
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                if callable(method):
                    print(f"  ✓ {method_name} 方法存在且可调用")
                else:
                    print(f"  ✗ {method_name} 不可调用")
                    return False
            else:
                print(f"  ✗ {method_name} 方法不存在")
                return False
    except Exception as e:
        print(f"✗ 方法签名检查失败: {e}")
        return False
    
    return True


def test_inheritance():
    """测试继承关系"""
    print("\n" + "=" * 60)
    print("测试4: 继承关系验证")
    print("=" * 60)
    
    try:
        from agents.agent import Agent
        from agents.new_agent import (
            ReinforcementLearningAgent,
            EnhancedMCTSAgent,
            AdvancedPoolAI
        )
        
        classes_to_test = [
            ('ReinforcementLearningAgent', ReinforcementLearningAgent),
            ('EnhancedMCTSAgent', EnhancedMCTSAgent),
            ('AdvancedPoolAI', AdvancedPoolAI),
        ]
        
        for class_name, cls in classes_to_test:
            if issubclass(cls, Agent):
                print(f"  ✓ {class_name} 继承自 Agent")
            else:
                print(f"  ✗ {class_name} 未继承自 Agent")
                return False
    except Exception as e:
        print(f"✗ 继承关系检查失败: {e}")
        return False
    
    return True


def test_function_signature():
    """测试函数签名"""
    print("\n" + "=" * 60)
    print("测试5: 函数签名验证")
    print("=" * 60)
    
    try:
        from agents.new_agent import analyze_shot_for_reward
        
        sig = inspect.signature(analyze_shot_for_reward)
        params = list(sig.parameters.keys())
        
        expected_params = ['shot', 'last_state', 'player_targets']
        
        if params == expected_params:
            print(f"  ✓ analyze_shot_for_reward 参数正确")
            print(f"    参数: {params}")
        else:
            print(f"  ✗ analyze_shot_for_reward 参数不匹配")
            print(f"    期望: {expected_params}")
            print(f"    实际: {params}")
            return False
    except Exception as e:
        print(f"✗ 函数签名检查失败: {e}")
        return False
    
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "     new_agent.py 完整性测试".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("导入测试", test_imports),
        ("初始化测试", test_class_initialization),
        ("方法签名测试", test_method_signatures),
        ("继承关系测试", test_inheritance),
        ("函数签名测试", test_function_signature),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 执行异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ 所有测试通过！new_agent.py 已准备就绪。")
        print("\n快速使用指南:")
        print("  1. ReinforcementLearningAgent(n_simulations=50)  - 快速强化学习")
        print("  2. EnhancedMCTSAgent(n_simulations=100)  - 深度MCTS搜索")
        print("  3. AdvancedPoolAI(mode='auto')  - 推荐综合方案 ⭐")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    exit(main())
