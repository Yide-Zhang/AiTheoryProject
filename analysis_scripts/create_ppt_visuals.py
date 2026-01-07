
import os
import sys

# Add project root to sys.path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from poolenv import PoolEnv
from agents import BasicAgentPro, NewAgentFinal
from utils import set_random_seed

def create_ppt_assets():
    # 1. 设置保存目录 (Relative to project root)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppt_assets")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    set_random_seed(enable=False, seed=42)
    
    n_games = 30 # Run 30 games for stats
    env = PoolEnv()
    
    # Setup agents
    # Agent A: BasicAgentPro (Baseline)
    # Agent B: NewAgentFinal (Our Hero)
    agent_a = BasicAgentPro()
    agent_b = NewAgentFinal()
    agents = [agent_a, agent_b]
    
    # Stats tracking
    game_results = [] # 'A' or 'B' or 'SAME'
    game_lengths = [] # number of shots
    cumulative_win_rate_b = []
    
    best_game_record = None
    min_shots_win = 1000
    
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    print(f"Starting evaluation of {n_games} games...")
    
    wins_a = 0
    wins_b = 0
    draws = 0
    
    for i in range(n_games):
        print(f"Game {i+1}/{n_games}...", end='\r')
        env.reset(target_ball=target_ball_choice[i % 4])
        
        while True:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            # Decide action
            if player == 'A':
                # Game logic uses players[i%2] to switch start order, but here let's simplify:
                # To be consistent with evaluate.py's turn logic:
                current_agent = agents[0] if player == 'A' else agents[1]
                # Actually, evaluate.py swaps who is 'A' and 'B' in the storage array.
                # Let's just fix Agent A = BasicPro, Agent B = NewAgentFinal.
                # But env.reset() doesn't determine who starts, usually A starts?
                # In evaluate.py:
                # players = [agent_a, agent_b] -> game 0: A=agent_a, B=agent_b
                # game 1: A=agent_b, B=agent_a (swapped roles?)
                # evaluate.py logic:
                # if player == 'A': action = players[i%2].decision
                # else: action = players[(i+1)%2].decision
                
                # So in even games (0, 2), players[0] (BasicPro) works for A.
                # In odd games (1, 3), players[1] (NewAgent) works for A.
                
                # We need to know who is who to track stats correctly.
                # Let's follow evaluate.py logic exactly to be fair.
                pass
            
            # Using evaluate.py specific logic for agent selection
            if player == 'A':
                action = agents[i % 2].decision(*obs)
            else:
                action = agents[(i + 1) % 2].decision(*obs)
                
            env.take_shot(action)
            done, info = env.get_done()
            
            if done:
                winner = info['winner'] # 'A', 'B', 'SAME'
                
                # Map winner 'A'/'B' back to Agent Identity
                # If i%2 == 0: A=BasicPro, B=NewFinal. Win A -> BasicPro wins. Win B -> NewFinal wins.
                # If i%2 == 1: A=NewFinal, B=BasicPro. Win A -> NewFinal wins. Win B -> BasicPro wins.
                
                real_winner_agent = None
                if winner == 'SAME':
                    draws += 1
                    game_results.append('Draw')
                else:
                    if i % 2 == 0:
                        # Even games: A=BasicPro, B=NewFinal
                        if winner == 'A': 
                            wins_a += 1
                            real_winner_agent = 'BasicAgentPro'
                        else: 
                            wins_b += 1
                            real_winner_agent = 'NewAgentFinal'
                    else:
                        # Odd games: A=NewFinal, B=BasicPro
                        if winner == 'A': 
                            wins_b += 1 # NewFinal is A
                            real_winner_agent = 'NewAgentFinal'
                        else: 
                            wins_a += 1 # BasicPro is B
                            real_winner_agent = 'BasicAgentPro'
                            
                # Calculate Win Rate for NewAgentFinal (wins_b)
                total_valid = wins_a + wins_b + draws # Count all games
                # win rate = wins / total
                wr = wins_b / (i + 1)
                cumulative_win_rate_b.append(wr)
                
                game_lengths.append(env.hit_count)
                
                # Save Highlight: Won by NewAgentFinal and short game
                if real_winner_agent == 'NewAgentFinal':
                    if env.hit_count < min_shots_win:
                        min_shots_win = env.hit_count
                        best_game_record = env.shot_record
                        print(f"\nNew best game found! Length: {env.hit_count}")
                
                break
                
    print(f"\nEvaluation Complete.")
    print(f"Final Score: BasicPro {wins_a} - {wins_b} NewAgentFinal (Draws: {draws})")

    # 2. 生成图表
    
    # Chart 1: Cumulative Win Rate
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_games + 1), cumulative_win_rate_b, marker='o', label='NewAgentFinal Win Rate')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate')
    plt.title('Agent Performance: Cumulative Win Rate over Games')
    plt.xlabel('Number of Games Played')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'win_rate_performance.png'))
    plt.close()
    
    # Chart 2: Game Length Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(game_lengths, bins=range(min(game_lengths), max(game_lengths) + 2), alpha=0.7, color='green', edgecolor='black')
    plt.title('Game Duration Distribution (Shots per Game)')
    plt.xlabel('Shots per Game')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, 'game_length_dist.png'))
    plt.close()
    
    print(f"Charts saved to {output_dir}")
    
    # 3. 保存精彩回放
    if best_game_record:
        replay_path = os.path.join(output_dir, 'best_agent_win.json')
        best_game_record.save(replay_path)
        print(f"Highlight replay saved to {replay_path}")
        print("You can view this replay using pooltool GUI.")
    else:
        print("No wins by NewAgentFinal recorded to save.")

if __name__ == '__main__':
    create_ppt_assets()
