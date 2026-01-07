
import pooltool as pt
import sys
import os

def sensitive_show(shot):
    """
    Attempt to show the shot using pooltool.
    Some environments might not support 3D rendering.
    """
    try:
        pt.show(shot, title="Best Game Replay")
    except Exception as e:
        print(f"Error displaying replay: {e}")
        print("Ensure you have a display environment available.")

if __name__ == "__main__":
    replay_path = os.path.join("ppt_assets", "best_agent_win.json")
    
    if len(sys.argv) > 1:
        replay_path = sys.argv[1]
        
    if not os.path.exists(replay_path):
        print(f"Replay file not found: {replay_path}")
        sys.exit(1)
        
    print(f"Loading replay from {replay_path}...")
    # Load MultiSystem
    ms = pt.MultiSystem.load(replay_path)
    
    print("Launching PoolTool GUI... (Press ESC to exit, 'n' for next shot, 'p' for previous)")
    sensitive_show(ms)
