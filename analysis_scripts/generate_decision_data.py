
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pooltool as pt
from agents.new_agent_final import NewAgentFinal
from agents.decision_logger import logger
from poolenv import PoolEnv

def generate_data():
    print("------------------------------------------------")
    print("Initializing Internal Decision Data Generation...")
    print("------------------------------------------------")
    
    agent = NewAgentFinal()
    env = PoolEnv()
    
    # Update logger directory to be absolute path in root
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppt_assets_advanced")
    logger.log_dir = log_dir # Manually update since logger was initialized at import time with default relative path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    n_games = 5
    print(f"Running {n_games} games. Data will be saved to '{log_dir}/internal_decision_log.csv'")
    
    for i in range(n_games):
        logger.new_game()
        print(f"Starting Game {i+1}...")
        
        target_type = 'solid' if i % 2 == 0 else 'stripe'
        env.reset(target_ball=target_type)
        # Ensure done is False (workaround for potential bug in PoolEnv or just safety)
        env.done = False
        
        step_count = 0
        done_flag = False
        while not done_flag:
            step_count += 1
            if step_count > 60:
                print("Game too long, forcing termination.")
                break
                
            player_id = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player_id)
            
            action = agent.decision(balls, my_targets, table)
            
            # Use take_shot now
            info = env.take_shot(action)
            
            # Update done status
            # get_done returns (is_done, info_dict)
            d, _ = env.get_done()
            if d:
                done_flag = True

    logger.save_logs()
    print("Data generation complete.")
    print(f"Logs saved: {len(logger.logs)} decisions recorded.")

if __name__ == "__main__":
    generate_data()
