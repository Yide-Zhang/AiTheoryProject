import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pooltool as pt

from .new_agent_fast_hybrid_v2 import FastHybridAgentV2
from .new_agent_integrated import ActionDict, BallDict


@dataclass
class BalancedGameState:
    own_remaining: int
    opponent_remaining: int
    easy_shots: int
    cue_risk: float
    eight_danger: float
    offensive_window: float


class BalancedActionPlanner:
    """Generates supplemental actions that balance risk and offense."""

    def __init__(self, agent: FastHybridAgentV2) -> None:
        self._agent = agent
        self.easy_threshold = 0.38
        self.medium_threshold = 0.58

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_state(
        self,
        balls: BallDict,
        my_targets: Optional[Sequence[str]],
        table: pt.Table,
    ) -> BalancedGameState:
        if not balls or my_targets is None:
            return BalancedGameState(0, 0, 0, 0.0, 0.0, 0.0)

        cue_ball = balls.get("cue")
        cue_pos = (
            np.array(cue_ball.state.rvw[0][:2]) if cue_ball is not None else np.zeros(2)
        )

        own_remaining = sum(
            1 for bid in my_targets if bid != "8" and bid in balls and balls[bid].state.s != 4
        )

        opponent_remaining = self._count_opponent_remaining(balls)
        avoid_eight = own_remaining > 0
        easy_shots = self._count_easy_shots(cue_pos, balls, my_targets, table, avoid_eight)
        cue_risk = self._calculate_cue_risk(cue_pos, table)
        eight_danger = self._estimate_eight_danger(cue_pos, balls, table)
        offensive_window = 0.0
        if own_remaining:
            offensive_window = min(1.0, easy_shots / max(1, own_remaining))
        elif easy_shots:
            offensive_window = 0.75

        return BalancedGameState(
            own_remaining=own_remaining,
            opponent_remaining=opponent_remaining,
            easy_shots=easy_shots,
            cue_risk=cue_risk,
            eight_danger=eight_danger,
            offensive_window=offensive_window,
        )

    def select_strategy(
        self,
        game_state: BalancedGameState,
        risk_tolerance: float,
    ) -> str:
        if game_state.own_remaining == 0:
            return "eight_only"
        if game_state.easy_shots >= 2 and game_state.eight_danger < 0.3:
            return "aggressive_clearance"
        if game_state.opponent_remaining <= 2 and game_state.own_remaining <= 3:
            return "mixed_endgame"

        risk_pressure = max(game_state.cue_risk, game_state.eight_danger)
        defensive_bar = 0.55 - risk_tolerance * 0.2
        if game_state.easy_shots == 0 or risk_pressure > defensive_bar:
            return "defensive"
        if game_state.offensive_window > (0.5 + 0.2 * risk_tolerance):
            return "balanced_aggressive"
        return "balanced"

    def generate_actions(
        self,
        strategy: str,
        balls: BallDict,
        my_targets: Optional[Sequence[str]],
        table: pt.Table,
        game_state: BalancedGameState,
    ) -> List[Dict[str, object]]:
        if my_targets is None:
            return []
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        avoid_eight = game_state.own_remaining > 0
        ranked_targets = self._rank_targets(
            cue_pos,
            balls,
            my_targets,
            table,
            avoid_eight,
            limit=8,
        )

        if strategy == "aggressive_clearance":
            return self._generate_easy_shots(ranked_targets, limit=6)
        if strategy == "mixed_endgame":
            return (
                self._generate_easy_shots(ranked_targets, limit=3)
                + self._generate_medium_patterns(ranked_targets, limit=3)
                + self._generate_adaptive_safeties(cue_pos, balls, my_targets, table, count=2)
            )
        if strategy == "defensive":
            return self._generate_adaptive_safeties(cue_pos, balls, my_targets, table, count=5)
        if strategy == "eight_only":
            return self._generate_eight_finishers(cue_pos, balls, table)
        if strategy == "balanced_aggressive":
            combined = self._generate_easy_shots(ranked_targets, limit=4)
            combined += self._generate_medium_patterns(ranked_targets, limit=2)
            return combined
        return self._generate_diversified_mix(
            ranked_targets,
            cue_pos,
            balls,
            my_targets,
            table,
        )

    def merge_candidates(
        self,
        base_candidates: List[Dict[str, object]],
        balanced_candidates: List[Dict[str, object]],
        game_state: BalancedGameState,
        risk_tolerance: float,
        limit: int,
    ) -> List[Dict[str, object]]:
        if not balanced_candidates:
            return base_candidates

        combined = balanced_candidates + base_candidates
        for info in combined:
            info["_balanced_priority"] = self._score_candidate_priority(
                info,
                game_state,
                risk_tolerance,
            )

        combined.sort(key=lambda info: info.get("_balanced_priority", 0.0), reverse=True)
        trimmed = combined[:limit]
        for info in trimmed:
            info.pop("_balanced_priority", None)
        return trimmed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_opponent_remaining(self, balls: BallDict) -> int:
        opponent_targets = self._agent._infer_opponent_targets()
        if not opponent_targets:
            return 0
        return sum(1 for bid in opponent_targets if bid in balls and balls[bid].state.s != 4)

    def _count_easy_shots(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        my_targets: Sequence[str],
        table: pt.Table,
        avoid_eight: bool,
    ) -> int:
        ranked = self._rank_targets(cue_pos, balls, my_targets, table, avoid_eight, limit=6)
        return sum(1 for entry in ranked if entry[-1] <= self.easy_threshold)

    def _calculate_cue_risk(self, cue_pos: np.ndarray, table: pt.Table) -> float:
        distance = self._agent._distance_to_cushion(cue_pos, table)
        cushion_risk = float(np.clip((0.12 - min(distance, 0.12)) / 0.12, 0.0, 1.0))
        pocket_risk = 0.0
        for pocket in table.pockets.values():
            pocket_dist = np.linalg.norm(cue_pos - np.array(pocket.center[:2]))
            pocket_risk = max(pocket_risk, float(np.clip((0.25 - min(pocket_dist, 0.25)) / 0.25, 0.0, 1.0)))
        return float(np.clip(0.6 * cushion_risk + 0.4 * pocket_risk, 0.0, 1.0))

    def _estimate_eight_danger(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        table: pt.Table,
    ) -> float:
        eight_ball = balls.get("8")
        if eight_ball is None or eight_ball.state.s == 4:
            return 0.0
        eight_pos = np.array(eight_ball.state.rvw[0][:2])
        pocket_dist = min(
            np.linalg.norm(eight_pos - np.array(pocket.center[:2]))
            for pocket in table.pockets.values()
        )
        pocket_pressure = float(np.clip((0.32 - min(pocket_dist, 0.32)) / 0.32, 0.0, 1.0))
        direction = eight_pos - cue_pos
        seg_len = np.linalg.norm(direction)
        if seg_len <= 1e-6:
            exposure = 1.0
        else:
            path_clear, clearance = self._agent._check_path_clear(
                cue_pos,
                eight_pos,
                balls,
                ignore_ids={"cue", "8"},
            )
            exposure = clearance if path_clear else clearance * 0.5
        return float(np.clip(0.55 * pocket_pressure + 0.45 * exposure, 0.0, 1.0))

    def _rank_targets(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        my_targets: Sequence[str],
        table: pt.Table,
        avoid_eight: bool,
        limit: int,
    ) -> List[Tuple[str, str, float, float, float, bool, float]]:
        candidates: List[Tuple[str, str, float, float, float, bool, float]] = []
        for target_id in my_targets:
            if avoid_eight and target_id == "8":
                continue
            obj_ball = balls.get(target_id)
            if obj_ball is None or obj_ball.state.s == 4:
                continue
            obj_pos = np.array(obj_ball.state.rvw[0][:2])
            best_entry: Optional[Tuple[str, str, float, float, float, bool, float]] = None
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array(pocket.center[:2])
                phi, distance, ghost_pos = self._agent._ghost_ball_solution(cue_pos, obj_pos, pocket_pos)
                if distance <= 1e-6:
                    continue
                path_clear, clearance = self._agent._check_path_clear(
                    cue_pos,
                    ghost_pos,
                    balls,
                    ignore_ids={"cue", target_id},
                )
                difficulty = self._estimate_difficulty(
                    cue_pos,
                    obj_pos,
                    pocket_pos,
                    clearance,
                    path_clear,
                )
                entry = (target_id, pocket_id, phi, distance, clearance, path_clear, difficulty)
                if best_entry is None or difficulty < best_entry[-1]:
                    best_entry = entry
            if best_entry is not None:
                candidates.append(best_entry)
        candidates.sort(key=lambda entry: entry[-1])
        return candidates[:limit]

    def _estimate_difficulty(
        self,
        cue_pos: np.ndarray,
        obj_pos: np.ndarray,
        pocket_pos: np.ndarray,
        clearance: float,
        path_clear: bool,
    ) -> float:
        vec_ct = obj_pos - cue_pos
        vec_tp = pocket_pos - obj_pos
        dist_ct = np.linalg.norm(vec_ct)
        dist_tp = np.linalg.norm(vec_tp)
        if dist_ct <= 1e-6 or dist_tp <= 1e-6:
            return 0.0
        cos_angle = float(
            np.clip(np.dot(vec_ct, vec_tp) / (dist_ct * dist_tp), -1.0, 1.0)
        )
        angle = math.degrees(math.acos(cos_angle))
        angle_penalty = min(angle / 70.0, 1.0)
        dist_penalty = min(dist_ct / 2.2, 1.0)
        pocket_penalty = min(dist_tp / 1.6, 1.0)
        clearance_penalty = 1.0 - clearance
        blocked_penalty = 0.25 if not path_clear else 0.0
        difficulty = (
            0.35 * dist_penalty
            + 0.2 * pocket_penalty
            + 0.25 * angle_penalty
            + 0.2 * clearance_penalty
            + blocked_penalty
        )
        return float(np.clip(difficulty, 0.0, 1.4))

    def _generate_easy_shots(
        self,
        ranked_targets: List[Tuple[str, str, float, float, float, bool, float]],
        limit: int,
    ) -> List[Dict[str, object]]:
        actions: List[Dict[str, object]] = []
        for entry in ranked_targets:
            if entry[-1] > self.easy_threshold:
                continue
            target_id, pocket_id, phi, distance, clearance, path_clear, difficulty = entry
            base_power = np.clip(1.6 + distance * 1.15 - difficulty * 0.4, 1.1, 5.2)
            action = self._agent._pack_action(base_power, phi, 0.0, 0.0, 0.0)
            actions.append(
                {
                    "action": action,
                    "strategy": "balanced_offense",
                    "target": target_id,
                    "pocket": pocket_id,
                    "clearance": clearance,
                    "path_clear": path_clear,
                }
            )
            if distance > 0.45:
                variant = self._agent._pack_action(
                    base_power * 0.96,
                    (phi + random.uniform(-0.65, 0.65)) % 360.0,
                    0.0,
                    0.0,
                    -0.06,
                )
                actions.append(
                    {
                        "action": variant,
                        "strategy": "balanced_offense",
                        "target": target_id,
                        "pocket": pocket_id,
                        "clearance": clearance * 0.95,
                        "path_clear": path_clear,
                    }
                )
            if len(actions) >= limit:
                break
        return actions

    def _generate_medium_patterns(
        self,
        ranked_targets: List[Tuple[str, str, float, float, float, bool, float]],
        limit: int,
    ) -> List[Dict[str, object]]:
        actions: List[Dict[str, object]] = []
        for entry in ranked_targets:
            difficulty = entry[-1]
            if not self.easy_threshold < difficulty <= self.medium_threshold:
                continue
            target_id, pocket_id, phi, distance, clearance, path_clear, _ = entry
            base_power = np.clip(1.9 + distance * 1.3, 1.3, 5.8)
            top_spin = self._agent._pack_action(base_power, phi, 0.0, 0.12, 0.0)
            back_spin = self._agent._pack_action(base_power * 0.95, phi, 0.0, -0.08, -0.02)
            actions.append(
                {
                    "action": top_spin,
                    "strategy": "balanced_positional",
                    "target": target_id,
                    "pocket": pocket_id,
                    "clearance": clearance,
                    "path_clear": path_clear,
                }
            )
            actions.append(
                {
                    "action": back_spin,
                    "strategy": "balanced_positional",
                    "target": target_id,
                    "pocket": pocket_id,
                    "clearance": clearance * 0.92,
                    "path_clear": path_clear,
                }
            )
            if len(actions) >= limit:
                break
        return actions

    def _generate_adaptive_safeties(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        my_targets: Sequence[str],
        table: pt.Table,
        count: int,
    ) -> List[Dict[str, object]]:
        actions: List[Dict[str, object]] = []
        safe_actions = self._agent.safe_generator.generate_safe_actions(
            balls,
            list(my_targets),
            table,
            limit=max(count, 3),
        )
        for info in safe_actions:
            cloned = dict(info)
            cloned["strategy"] = "balanced_safety"
            actions.append(cloned)
            if len(actions) >= count:
                break

        if len(actions) < count:
            parking = self._parking_safety(cue_pos, table)
            if parking is not None:
                actions.append(parking)
        if len(actions) < count:
            hide_action = self._bury_cue_behind_ball(cue_pos, balls, my_targets)
            if hide_action is not None:
                actions.append(hide_action)
        return actions[:count]

    def _parking_safety(self, cue_pos: np.ndarray, table: pt.Table) -> Optional[Dict[str, object]]:
        xmin, xmax, ymin, ymax = self._agent._pos_from_table(table)
        safe_zone = np.array([
            0.5 * (xmin + xmax),
            ymin + 0.2 * (ymax - ymin),
        ])
        direction = safe_zone - cue_pos
        distance = np.linalg.norm(direction)
        if distance <= 1e-6:
            return None
        phi = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
        action = self._agent._pack_action(1.1 + distance * 0.35, phi, 0.0, 0.0, 0.08)
        return {
            "action": action,
            "strategy": "balanced_safety",
            "target": "cue_reset",
            "pocket": None,
            "clearance": 0.0,
            "path_clear": True,
        }

    def _bury_cue_behind_ball(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        my_targets: Sequence[str],
    ) -> Optional[Dict[str, object]]:
        opponent_targets = self._agent._infer_opponent_targets() or []
        blockers = [
            bid
            for bid in opponent_targets
            if bid in balls and balls[bid].state.s != 4
        ]
        if not blockers:
            blockers = [
                bid
                for bid in my_targets
                if bid in balls and balls[bid].state.s != 4
            ]
        if not blockers:
            return None
        blocker_id = min(
            blockers,
            key=lambda bid: np.linalg.norm(cue_pos - np.array(balls[bid].state.rvw[0][:2])),
        )
        blocker_pos = np.array(balls[blocker_id].state.rvw[0][:2])
        direction = blocker_pos - cue_pos
        distance = np.linalg.norm(direction)
        if distance <= 1e-6:
            return None
        phi = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
        action = self._agent._pack_action(np.clip(0.9 + distance * 0.4, 0.9, 2.2), phi, 0.0, 0.0, -0.05)
        return {
            "action": action,
            "strategy": "balanced_safety",
            "target": blocker_id,
            "pocket": None,
            "clearance": 0.0,
            "path_clear": True,
        }

    def _generate_eight_finishers(
        self,
        cue_pos: np.ndarray,
        balls: BallDict,
        table: pt.Table,
    ) -> List[Dict[str, object]]:
        ranked = self._rank_targets(
            cue_pos,
            balls,
            ["8"],
            table,
            avoid_eight=False,
            limit=3,
        )
        actions: List[Dict[str, object]] = []
        for entry in ranked:
            target_id, pocket_id, phi, distance, clearance, path_clear, difficulty = entry
            power = np.clip(1.7 + distance * 1.1, 1.2, 4.8)
            action = self._agent._pack_action(power, phi, 0.0, 0.0, 0.0)
            actions.append(
                {
                    "action": action,
                    "strategy": "balanced_eight",
                    "target": target_id,
                    "pocket": pocket_id,
                    "clearance": clearance,
                    "path_clear": path_clear,
                }
            )
            if difficulty > self.easy_threshold and distance > 0.8:
                safety = self._agent._pack_action(power * 0.9, phi, 0.0, 0.0, -0.08)
                actions.append(
                    {
                        "action": safety,
                        "strategy": "balanced_eight",
                        "target": target_id,
                        "pocket": pocket_id,
                        "clearance": clearance * 0.9,
                        "path_clear": path_clear,
                    }
                )
        return actions

    def _generate_diversified_mix(
        self,
        ranked_targets: List[Tuple[str, str, float, float, float, bool, float]],
        cue_pos: np.ndarray,
        balls: BallDict,
        my_targets: Sequence[str],
        table: pt.Table,
    ) -> List[Dict[str, object]]:
        easy = self._generate_easy_shots(ranked_targets, limit=3)
        medium = self._generate_medium_patterns(ranked_targets, limit=2)
        safety = self._generate_adaptive_safeties(cue_pos, balls, my_targets, table, count=2)
        return easy + medium + safety

    def _score_candidate_priority(
        self,
        info: Dict[str, object],
        game_state: BalancedGameState,
        risk_tolerance: float,
    ) -> float:
        clearance = float(info.get("clearance", 0.0))
        path_bonus = 0.05 if info.get("path_clear", True) else -0.1
        strategy = str(info.get("strategy", ""))
        if "safety" in strategy:
            safety_weight = self._agent.strategy_weights.get("safety_only", 0.3)
            return safety_weight * (1.1 - game_state.offensive_window) + path_bonus
        difficulty_proxy = 1.0 - clearance
        if difficulty_proxy > 0.55:
            weight = self._agent.strategy_weights.get("high_risk_low_reward", 0.2)
        elif difficulty_proxy > 0.35:
            weight = self._agent.strategy_weights.get("medium_risk_high_reward", 0.6)
        else:
            weight = self._agent.strategy_weights.get("low_risk_high_reward", 0.9)
        aggression_bias = risk_tolerance * 0.4 if "offense" in strategy else 0.15
        reward = weight * (clearance + aggression_bias)
        risk_penalty = game_state.eight_danger * (1.0 - risk_tolerance) * 0.4
        return reward - risk_penalty + path_bonus


class FastHybridAgentV3(FastHybridAgentV2):
    """FastHybridAgentV2 with explicit risk-balanced planning."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
        risk_tolerance: float = 0.65,
    ) -> None:
        super().__init__(max_candidates, evaluation_repeats, refine_top_k, refine_trials)
        self.risk_tolerance = float(np.clip(risk_tolerance, 0.05, 0.95))
        self.strategy_weights: Dict[str, float] = {
            "high_risk_low_reward": 0.15,
            "medium_risk_high_reward": 0.65,
            "low_risk_high_reward": 0.9,
            "safety_only": 0.4,
        }
        self.balanced_planner = BalancedActionPlanner(self)
        self._last_strategy = "balanced"

    def _generate_candidates(  # type: ignore[override]
        self,
        balls: BallDict,
        my_targets: List[str],
        table: pt.Table,
        phase: str,
    ) -> List[Dict[str, object]]:
        base_candidates = super()._generate_candidates(balls, my_targets, table, phase)
        if not balls or not my_targets:
            return base_candidates

        game_state = self.balanced_planner.analyze_state(balls, my_targets, table)
        strategy = self.balanced_planner.select_strategy(game_state, self.risk_tolerance)
        self._last_strategy = strategy
        balanced_candidates = self.balanced_planner.generate_actions(
            strategy,
            balls,
            my_targets,
            table,
            game_state,
        )
        if not balanced_candidates:
            return base_candidates

        merged = self.balanced_planner.merge_candidates(
            base_candidates,
            balanced_candidates,
            game_state,
            self.risk_tolerance,
            limit=self.max_candidates,
        )
        print(
            f"[FastHybridAgentV3] strategy={strategy} easy={game_state.easy_shots} "
            f"own={game_state.own_remaining} opp={game_state.opponent_remaining}"
        )
        return merged
