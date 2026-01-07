
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from poolenv import PoolEnv
from agents import BasicAgentPro, NewAgentFinal
from utils import set_random_seed

# 设置中文字体，尝试几个常见的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def run_advanced_analytics():
    # 1. 配置
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppt_assets_advanced")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    set_random_seed(enable=False, seed=100) # 使用不同的种子
    n_games = 50 
    
    # 初始化 Agent
    agent_baseline = BasicAgentPro()
    agent_hero = NewAgentFinal()
    
    # 映射名称到 ID
    agent_map = {
        'BasicAgentPro': agent_baseline,
        'NewAgentFinal': agent_hero
    }
    
    # 统计数据结构
    stats = {
        'BasicAgentPro': {
            'games_played': 0, 'games_won': 0,
            'shots_total': 0, 'shots_potted': 0, 'fouls': 0,
            'streaks': [], 'current_streak': 0,
            'shots_in_winning_games': []
        },
        'NewAgentFinal': {
            'games_played': 0, 'games_won': 0,
            'shots_total': 0, 'shots_potted': 0, 'fouls': 0,
            'streaks': [], 'current_streak': 0,
            'shots_in_winning_games': []
        }
    }
    
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    env = PoolEnv()
    
    print(f"正在进行 {n_games} 局的高级数据分析对战...")
    
    for i in range(n_games):
        print(f"Progress: {i+1}/{n_games}", end='\r')
        env.reset(target_ball=target_ball_choice[i % 4])
        
        # 确定本局谁是 Player A, 谁是 Player B
        # 偶数局: A=Basic, B=New
        # 奇数局: A=New, B=Basic
        if i % 2 == 0:
            pA_name = 'BasicAgentPro'
            pB_name = 'NewAgentFinal'
            pA_agent = agent_baseline
            pB_agent = agent_hero
        else:
            pA_name = 'NewAgentFinal'
            pB_name = 'BasicAgentPro'
            pA_agent = agent_hero
            pB_agent = agent_baseline
            
        current_game_shots = {pA_name: 0, pB_name: 0}
        
        # 重置局内连杆统计 (其实应该在每次击球时重置或增加)
        stats[pA_name]['current_streak'] = 0
        stats[pB_name]['current_streak'] = 0
        
        while True:
            player_symbol = env.get_curr_player() # 'A' or 'B'
            current_agent_name = pA_name if player_symbol == 'A' else pB_name
            current_agent = pA_agent if player_symbol == 'A' else pB_agent
            
            obs = env.get_observation(player_symbol)
            action = current_agent.decision(*obs)
            
            step_info = env.take_shot(action)
            current_game_shots[current_agent_name] += 1
            stats[current_agent_name]['shots_total'] += 1
            
            # 分析击球结果
            is_foul = False
            is_pot = False
            
            # 检查犯规
            if step_info.get('WHITE_BALL_INTO_POCKET') or \
               step_info.get('FOUL_FIRST_HIT') or \
               (step_info.get('NO_POCKET_NO_RAIL') and not step_info.get('ME_INTO_POCKET')): 
                # 注意：NO_POCKET_NO_RAIL在进球时通常为False，但在某些规则下如果进球了不算NoRail
                # 这里主要看 PoolEnv 逻辑。如果进白球肯定犯规。
                # 如果没碰到任何球(step_info里通常通过NO_HIT判断，这里PoolEnv好像没直接返NO_HIT但有NO_POCKET_NO_RAIL)
                is_foul = True
                stats[current_agent_name]['fouls'] += 1
            
            # 检查进球 (打进自己的球)
            if step_info.get('ME_INTO_POCKET'):
                is_pot = True
                stats[current_agent_name]['shots_potted'] += 1
                stats[current_agent_name]['current_streak'] += 1
            else:
                # 记录断杆，把之前的连杆数据存入
                if stats[current_agent_name]['current_streak'] > 0:
                    stats[current_agent_name]['streaks'].append(stats[current_agent_name]['current_streak'])
                stats[current_agent_name]['current_streak'] = 0
            
            # 检查胜负
            done, info = env.get_done()
            if done:
                winner_symbol = info['winner'] # 'A', 'B', 'SAME'
                
                # 记录最后一次streak (如果刚好赢了)
                if stats[current_agent_name]['current_streak'] > 0:
                    stats[current_agent_name]['streaks'].append(stats[current_agent_name]['current_streak'])
                
                stats['BasicAgentPro']['games_played'] += 1
                stats['NewAgentFinal']['games_played'] += 1
                
                if winner_symbol != 'SAME':
                    winner_name = pA_name if winner_symbol == 'A' else pB_name
                    stats[winner_name]['games_won'] += 1
                    stats[winner_name]['shots_in_winning_games'].append(current_game_shots[winner_name])
                break
                
    # 数据汇总计算
    summary = []
    for name, data in stats.items():
        summary.append({
            'Agent': name,
            'Win Rate': (data['games_won'] / n_games) * 100,
            'Potting Accuracy': (data['shots_potted'] / data['shots_total']) * 100 if data['shots_total'] else 0,
            'Foul Rate': (data['fouls'] / data['shots_total']) * 100 if data['shots_total'] else 0,
            'Avg Streak': np.mean(data['streaks']) if data['streaks'] else 0,
            'Max Streak': np.max(data['streaks']) if data['streaks'] else 0,
            'Avg Shots per Win': np.mean(data['shots_in_winning_games']) if data['shots_in_winning_games'] else 0
        })
    
    df = pd.DataFrame(summary)
    print("\n\n=== Advanced Analytics Results ===")
    print(df.to_string(index=False))
    
    # 保存 Excel/CSV
    csv_path = os.path.join(output_dir, 'agent_comparison_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed stats saved to {csv_path}")

    # ================= 绘图 =================
    
    # 1. 核心指标对比雷达图 (归一化处理)
    # 指标：Win Rate, Potting Acc, (100-Foul Rate), (Avg Streak/3 * 100 放大), (10/Avg Shots * 100 效率)
    
    # 简化版：直接画柱状图对比关键指标
    metrics = ['Potting Accuracy', 'Foul Rate', 'Avg Streak', 'Avg Shots per Win']
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取数据
    val_basic = [
        df[df['Agent']=='BasicAgentPro']['Potting Accuracy'].values[0],
        df[df['Agent']=='BasicAgentPro']['Foul Rate'].values[0],
        df[df['Agent']=='BasicAgentPro']['Avg Streak'].values[0],
        df[df['Agent']=='BasicAgentPro']['Avg Shots per Win'].values[0]
    ]
    
    val_new = [
        df[df['Agent']=='NewAgentFinal']['Potting Accuracy'].values[0],
        df[df['Agent']=='NewAgentFinal']['Foul Rate'].values[0],
        df[df['Agent']=='NewAgentFinal']['Avg Streak'].values[0],
        df[df['Agent']=='NewAgentFinal']['Avg Shots per Win'].values[0]
    ]
    
    # 绘制
    rects1 = ax.bar(x - width/2, val_basic, width, label='BasicAgentPro', color='#888888')
    rects2 = ax.bar(x + width/2, val_new, width, label='NewAgentFinal', color='#2ca02c') # Green for hero
    
    ax.set_ylabel('Value')
    ax.set_title('Agent Technical Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'technical_comparison.png'))
    plt.close()
    
    # 2. 连续进球能力分布 (Streak Distribution) - 很专业的指标
    # 统计两个agent的所有streaks
    streaks_basic = stats['BasicAgentPro']['streaks']
    streaks_new = stats['NewAgentFinal']['streaks']
    
    plt.figure(figsize=(10, 6))
    # 限制bins范围，通常1-8
    max_s = max(max(streaks_basic) if streaks_basic else 1, max(streaks_new) if streaks_new else 1)
    bins = np.arange(1, max_s + 2) - 0.5
    
    plt.hist([streaks_basic, streaks_new], bins=bins, label=['BasicAgentPro', 'NewAgentFinal'], 
             color=['#888888', '#2ca02c'], density=True)
    
    plt.title('Consecutive Potting Probability (Break Building)')
    plt.xlabel('Consecutive Balls Potted')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max_s + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'streak_distribution.png'))
    plt.close()
    
    print(f"Professional charts saved to {output_dir}")

if __name__ == "__main__":
    run_advanced_analytics()
