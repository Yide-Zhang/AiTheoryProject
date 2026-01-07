
import json
import time
import os
import csv
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Candidate:
    score: float
    strategy: str
    confidence: float
    metadata: str = ""

class DecisionLogger:
    def __init__(self, log_dir="ppt_assets_advanced"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.logs = []
        self.game_id = 0
        self.shot_id = 0
        
    def new_game(self):
        self.game_id += 1
        self.shot_id = 0
        
    def log_decision(self, 
                     step_type: str, 
                     chosen_score: float, 
                     candidates: List[Candidate],
                     best_strategy: str):
        
        self.shot_id += 1
        
        # Calculate stats from candidates
        scores = [c.score for c in candidates]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # Prepare log entry
        entry = {
            "game_id": self.game_id,
            "shot_id": self.shot_id,
            "timestamp": time.time(),
            "step_type": step_type,
            "best_strategy": best_strategy,
            "chosen_score": chosen_score,
            "candidate_count": len(candidates),
            "avg_cand_score": avg_score,
            "max_cand_score": max_score,
            "score_spread": max_score - min_score,
            "top_candidate_strategies": [c.strategy for c in candidates[:5]],
            "candidates_dump": [asdict(c) for c in candidates] # Full dump for detailed analysis
        }
        self.logs.append(entry)

    def save_logs(self):
        # Save Summary CSV
        csv_path = os.path.join(self.log_dir, "internal_decision_log.csv")
        if not self.logs:
            return

        # Use fixed headers to avoid issues if candidates_dump is missing in some way (unlikely but safe)
        headers = ["game_id", "shot_id", "timestamp", "step_type", "best_strategy", "chosen_score", 
                   "candidate_count", "avg_cand_score", "max_cand_score", "score_spread"]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for entry in self.logs:
                writer.writerow(entry)
                
        # Save Full JSON
        json_path = os.path.join(self.log_dir, "internal_decision_log_full.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2)

logger = DecisionLogger()
