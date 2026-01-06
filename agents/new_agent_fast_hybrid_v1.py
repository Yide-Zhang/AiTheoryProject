import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pooltool as pt

from .new_agent_integrated import ActionDict, BallDict, IntegratedAgent


class BreakShotExpert:
    """生成针对开球局面的高效候选动作."""

    def __init__(self, agent: IntegratedAgent) -> None:
        self._agent = agent

    def candidates(self, balls: BallDict, table: pt.Table) -> List[Dict[str, object]]:
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        head_id, head_pos = self._locate_head_ball(balls, table)
        if head_id is None or head_pos is None:
            return []

        actions: List[Dict[str, object]] = []
        base_phi = self._bearing(cue_pos, head_pos)
        base_power = 7.4

        actions.append(self._package_action(base_power, base_phi, 0.0, 0.0, -0.18, "core", head_id))
        actions.append(self._package_action(base_power * 0.97, base_phi, 2.5, 0.0, -0.05, "pop", head_id))

        for offset, spin in ((-2.0, -0.2), (2.0, 0.2)):
            actions.append(
                self._package_action(
                    base_power * 0.99,
                    (base_phi + offset) % 360.0,
                    0.0,
                    spin,
                    -0.12,
                    f"cut_{'L' if offset < 0 else 'R'}",
                    head_id,
                )
            )

        actions.extend(self._second_ball_variants(balls, cue_pos, head_pos))
        return actions

    def _second_ball_variants(self, balls: BallDict, cue_pos: np.ndarray, head_pos: np.ndarray) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        neighbors = self._find_adjacent_balls(balls, head_pos)
        for direction, info in neighbors.items():
            if info is None:
                continue
            bid, pos = info
            phi = self._bearing(cue_pos, pos)
            side_spin = -0.22 if direction == "left" else 0.22
            tag = f"second_{direction}"
            results.append(self._package_action(7.1, phi, 0.0, side_spin, -0.08, tag, bid))
        return results

    def _find_adjacent_balls(
        self,
        balls: BallDict,
        head_pos: np.ndarray,
    ) -> Dict[str, Optional[Tuple[str, np.ndarray]]]:
        neighbors: Dict[str, Optional[Tuple[str, np.ndarray]]] = {"left": None, "right": None}
        for bid, ball in balls.items():
            if bid in {"cue", "8"} or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2])
            delta = pos - head_pos
            dist = float(np.linalg.norm(delta))
            if dist < 0.04 or dist > 0.09:
                continue
            key = "left" if delta[0] < 0 else "right"
            current = neighbors[key]
            if current is None or dist < np.linalg.norm(current[1] - head_pos):
                neighbors[key] = (bid, pos)
        return neighbors

    def _locate_head_ball(self, balls: BallDict, table: pt.Table) -> Tuple[Optional[str], Optional[np.ndarray]]:
        obj_ball = balls.get("1")
        if obj_ball is not None and obj_ball.state.s != 4:
            return "1", np.array(obj_ball.state.rvw[0][:2])

        candidates: List[Tuple[str, np.ndarray]] = []
        for bid, ball in balls.items():
            if bid == "cue" or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2])
            candidates.append((bid, pos))

        if not candidates:
            return None, None

        _, _, ymin, ymax = self._agent._pos_from_table(table)
        prefer_positive_axis = abs(ymax) >= abs(ymin)
        head_ball = max(
            candidates,
            key=lambda item: item[1][1] if prefer_positive_axis else -item[1][1],
        )
        return head_ball

    def _bearing(self, start: np.ndarray, end: np.ndarray) -> float:
        vec = end - start
        return math.degrees(math.atan2(vec[1], vec[0])) % 360.0

    def _package_action(
        self,
        V0: float,
        phi: float,
        theta: float,
        a: float,
        b: float,
        tag: str,
        target_id: str,
    ) -> Dict[str, object]:
        action = self._agent._pack_action(V0, phi, theta, a, b)
        return {
            "action": action,
            "strategy": "break",
            "target": target_id,
            "pocket": None,
            "clearance": 1.0,
            "path_clear": True,
            "variant": tag,
        }


class FastHybridAgentV1(IntegratedAgent):
    """集成常规策略与开球专家的高速混合智能体."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
    ) -> None:
        super().__init__(max_candidates, evaluation_repeats, refine_top_k, refine_trials)
        self.break_expert = BreakShotExpert(self)
        self.break_evaluation_repeats = max(4, evaluation_repeats + 1)
        self.break_cluster_threshold = 0.56

    def decision(
        self,
        balls: Optional[BallDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
    ) -> ActionDict:
        if balls is None or table is None:
            return self._random_action()

        if self._is_break_state(balls):
            break_action = self._decide_break_shot(balls, table)
            if break_action is not None:
                return break_action

        return super().decision(balls, my_targets, table)

    def _decide_break_shot(self, balls: BallDict, table: pt.Table) -> Optional[ActionDict]:
        candidates = self.break_expert.candidates(balls, table)
        if not candidates:
            return None

        scored: List[Tuple[float, ActionDict, Dict[str, object]]] = []
        for info in candidates:
            score = self._evaluate_break_candidate(info["action"], balls, table)
            scored.append((score, info["action"], info))

        for base_score, base_action, info in scored[:3]:
            mutated = self._mutate_action(base_action, scale=0.2)
            score = self._evaluate_break_candidate(mutated, balls, table)
            scored.append((score, mutated, info))

        best = max(scored, key=lambda item: item[0])
        print(
            f"[FastHybridAgentV1] variant={best[2].get('variant')} score={best[0]:.1f} strategy=break"
        )
        return best[1]

    def _evaluate_break_candidate(self, action: ActionDict, balls: BallDict, table: pt.Table) -> float:
        results: List[float] = []
        for _ in range(self.break_evaluation_repeats):
            noisy = self._apply_eval_noise(action)
            shot = self._simulate_action(balls, table, noisy)
            if shot is None:
                results.append(-400.0)
                continue
            results.append(self._score_break_shot(shot, table))
        return float(np.mean(results)) if results else -400.0

    def _score_break_shot(self, shot: pt.System, table: pt.Table) -> float:
        pocketed = [bid for bid, ball in shot.balls.items() if bid != "cue" and ball.state.s == 4]
        cue_in = shot.balls.get("cue") and shot.balls["cue"].state.s == 4
        eight_in = "8" in pocketed

        score = len(pocketed) * 90.0
        if cue_in:
            score -= 220.0
        if eight_in:
            score -= 80.0

        spread = self._estimate_spread(shot, table)
        score += spread * 140.0

        cue_ball = shot.balls.get("cue")
        if cue_ball is not None and cue_ball.state.s != 4:
            cue_pos = np.array(cue_ball.state.rvw[0][:2])
            cushion_dist = self._distance_to_cushion(cue_pos, table)
            if cushion_dist < 0.08:
                score -= (0.08 - cushion_dist) * 120.0
            else:
                score += min(cushion_dist - 0.08, 0.12) * 80.0

        if not pocketed and spread < 0.15:
            score -= 160.0
        return score

    def _estimate_spread(self, shot: pt.System, table: pt.Table) -> float:
        positions = [
            np.array(ball.state.rvw[0][:2])
            for bid, ball in shot.balls.items()
            if bid != "cue" and ball.state.s != 4
        ]
        if len(positions) < 2:
            return 0.0
        max_pair = self._max_pair_distance(positions)
        normalized = np.clip((max_pair - self.break_cluster_threshold) * 3.0, 0.0, 1.0)
        return float(normalized)

    def _is_break_state(self, balls: BallDict) -> bool:
        object_positions: List[np.ndarray] = []
        object_count = 0
        for bid, ball in balls.items():
            if bid == "cue":
                continue
            if ball.state.s == 4:
                return False
            object_count += 1
            object_positions.append(np.array(ball.state.rvw[0][:2]))

        if object_count < 10 or len(object_positions) < 2:
            return False
        max_pair = self._max_pair_distance(object_positions)
        return max_pair <= self.break_cluster_threshold

    def _max_pair_distance(self, positions: List[np.ndarray]) -> float:
        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
