import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pooltool as pt

from .new_agent_fast_hybrid_v1 import FastHybridAgentV1
from .new_agent_integrated import ActionDict, BallDict


class SafeActionGenerator:
    """生成以黑8安全为核心的候选动作."""

    def __init__(self, agent: FastHybridAgentV1) -> None:
        self._agent = agent
        self.ball_radius = agent.ball_radius

    def generate_safe_actions(
        self,
        balls: BallDict,
        my_targets: Optional[List[str]],
        table: pt.Table,
        limit: int = 32,
    ) -> List[Dict[str, object]]:
        cue_ball = balls.get("cue")
        if cue_ball is None or my_targets is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        eight_ball = balls.get("8")
        eight_pos = None
        if eight_ball is not None and eight_ball.state.s != 4:
            eight_pos = np.array(eight_ball.state.rvw[0][:2])

        own_remaining = [
            bid
            for bid in my_targets
            if bid != "8" and bid in balls and balls[bid].state.s != 4
        ]
        avoid_eight = bool(own_remaining)

        targets: List[str] = []
        if avoid_eight:
            targets = own_remaining
        elif eight_pos is not None:
            targets = ["8"]

        actions: List[Dict[str, object]] = []

        if avoid_eight and eight_pos is not None:
            actions.extend(self._generate_eight_ball_safety(cue_pos, eight_pos, table))

        for target_id in targets[:4]:
            obj_ball = balls.get(target_id)
            if obj_ball is None or obj_ball.state.s == 4:
                continue
            obj_pos = np.array(obj_ball.state.rvw[0][:2])
            target_near_eight = eight_pos is not None and self._near_eight(obj_pos, eight_pos)

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array(pocket.center[:2])
                phi, distance, ghost_pos = self._agent._ghost_ball_solution(cue_pos, obj_pos, pocket_pos)
                path_clear, clearance_ratio = self._agent._check_path_clear(
                    cue_pos,
                    ghost_pos,
                    balls,
                    ignore_ids={"cue", target_id},
                )

                if avoid_eight and eight_pos is not None and self._line_intersects_eight(cue_pos, ghost_pos, eight_pos):
                    continue

                base_power = float(np.clip(1.25 + distance * 1.3, 0.9, 6.6))
                variant = "safe_offense"

                if target_near_eight:
                    base_power = min(base_power, 2.4)
                    variant = "guarded_offense"

                actions.append(
                    self._build_action(
                        base_power,
                        phi,
                        0.0,
                        0.0,
                        0.0,
                        target_id,
                        pocket_id,
                        variant,
                        path_clear,
                        clearance_ratio,
                    )
                )

                for delta in (-0.4, 0.4):
                    tweaked_phi = (phi + delta) % 360.0
                    actions.append(
                        self._build_action(
                            base_power,
                            tweaked_phi,
                            0.0,
                            0.0,
                            0.0,
                            target_id,
                            pocket_id,
                            f"{variant}_adjust",
                            path_clear,
                            clearance_ratio * 0.9,
                        )
                    )

        return actions[:limit]

    def _generate_eight_ball_safety(self, cue_pos: np.ndarray, eight_pos: np.ndarray, table: pt.Table) -> List[Dict[str, object]]:
        actions: List[Dict[str, object]] = []
        safe_positions = self._find_safe_positions_for_eight(table)

        for idx, safe_pos in enumerate(safe_positions[:2]):
            vec = safe_pos - eight_pos
            distance = float(np.linalg.norm(vec))
            if distance < 0.12:
                continue
            offset = np.array([0.012 if vec[0] >= 0 else -0.012, 0.0])
            impact_point = eight_pos + offset
            direction = impact_point - cue_pos
            if np.linalg.norm(direction) <= 1e-6:
                continue
            phi = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
            V0 = float(np.clip(0.9 + distance * 0.8, 0.8, 2.6))
            actions.append(
                self._build_action(
                    V0,
                    phi,
                    0.0,
                    0.0,
                    -0.05,
                    "8",
                    None,
                    f"eight_safety_{idx}",
                    True,
                    1.0,
                )
            )
        return actions

    def _find_safe_positions_for_eight(self, table: pt.Table) -> List[np.ndarray]:
        xmin, xmax, ymin, ymax = self._agent._pos_from_table(table)
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        span_x = max(0.01, (xmax - xmin) * 0.25)
        span_y = max(0.01, (ymax - ymin) * 0.25)
        return [
            np.array([center_x, center_y + span_y]),
            np.array([center_x, center_y - span_y]),
            np.array([center_x - span_x, center_y]),
            np.array([center_x + span_x, center_y]),
        ]

    def _near_eight(self, target_pos: np.ndarray, eight_pos: np.ndarray) -> bool:
        return float(np.linalg.norm(target_pos - eight_pos)) < 0.16

    def _line_intersects_eight(
        self,
        cue_pos: np.ndarray,
        ghost_pos: np.ndarray,
        eight_pos: np.ndarray,
    ) -> bool:
        segment = ghost_pos - cue_pos
        seg_len = float(np.linalg.norm(segment))
        if seg_len <= 1e-6:
            return False
        unit = segment / seg_len
        rel = eight_pos - cue_pos
        proj = float(np.dot(rel, unit))
        if proj <= 0.0 or proj >= seg_len:
            return False
        closest = cue_pos + unit * proj
        distance = float(np.linalg.norm(eight_pos - closest))
        return distance < self.ball_radius * 2.2

    def _build_action(
        self,
        V0: float,
        phi: float,
        theta: float,
        a: float,
        b: float,
        target: str,
        pocket: Optional[str],
        variant: str,
        path_clear: bool,
        clearance_ratio: float,
    ) -> Dict[str, object]:
        packed = self._agent._pack_action(V0, phi, theta, a, b)
        return {
            "action": packed,
            "strategy": "safe" if target == "8" else "offense",
            "target": target,
            "pocket": pocket,
            "path_clear": path_clear,
            "clearance": float(np.clip(clearance_ratio, 0.0, 1.0)),
            "variant": variant,
        }


class FastHybridAgentV2(FastHybridAgentV1):
    """具备黑8安全保护的 Fast Hybrid Agent 版本."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
    ) -> None:
        super().__init__(max_candidates, evaluation_repeats, refine_top_k, refine_trials)
        self.safe_generator = SafeActionGenerator(self)
        self.premature_eight_penalty = 1000.0
        self.safety_angle_threshold = 5.0

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

        if my_targets is None:
            return super().decision(balls, my_targets, table)

        if self._needs_eight_protection(balls, my_targets):
            safe_action = self._safe_mode_decision(balls, my_targets, table)
            if safe_action is not None:
                return safe_action

        return super().decision(balls, my_targets, table)

    def _generate_candidates(  # type: ignore[override]
        self,
        balls: BallDict,
        my_targets: List[str],
        table: pt.Table,
        phase: str,
    ) -> List[Dict[str, object]]:
        base_candidates = super()._generate_candidates(balls, my_targets, table, phase)
        if not self._needs_eight_protection(balls, my_targets):
            return base_candidates

        safe_candidates = self.safe_generator.generate_safe_actions(balls, my_targets, table, limit=self.max_candidates)
        filtered_base = [info for info in base_candidates if info.get("target") != "8"]
        combined = safe_candidates + filtered_base
        combined.sort(
            key=lambda info: (
                info.get("strategy") != "offense",
                -info.get("clearance", 0.0),
            )
        )
        return combined[: self.max_candidates]

    def _safe_mode_decision(
        self,
        balls: BallDict,
        my_targets: List[str],
        table: pt.Table,
    ) -> Optional[ActionDict]:
        safe_candidates = self.safe_generator.generate_safe_actions(balls, my_targets, table, limit=self.max_candidates)
        if not safe_candidates:
            return None

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        evaluated: List[Tuple[float, ActionDict, Dict[str, object]]] = []
        for info in safe_candidates:
            score = self._evaluate_candidate(
                info["action"],
                info,
                balls,
                table,
                my_targets,
                last_state_snapshot,
            )
            evaluated.append((score, info["action"], info))

        if not evaluated:
            return None

        best_score, best_action, best_info = max(evaluated, key=lambda item: item[0])
        if self._needs_eight_protection(balls, my_targets) and self._is_aiming_directly_at_eight(best_action, balls):
            return self._fallback_defensive_action(balls, table)

        print(
            f"[FastHybridAgentV2] safe-mode strategy={best_info.get('strategy')} target={best_info.get('target')} score={best_score:.1f}"
        )
        return best_action

    def _fallback_defensive_action(self, balls: BallDict, table: pt.Table) -> ActionDict:
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return self._random_action()
        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        xmin, xmax, ymin, ymax = self._pos_from_table(table)
        center = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
        direction = center - cue_pos
        if np.linalg.norm(direction) <= 1e-6:
            direction = np.array([1.0, 0.0])
        phi = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
        return self._pack_action(1.2, phi, 0.0, 0.0, -0.1)

    def _needs_eight_protection(self, balls: BallDict, my_targets: Optional[List[str]]) -> bool:
        if my_targets is None:
            return False
        eight_ball = balls.get("8")
        if eight_ball is None or eight_ball.state.s == 4:
            return False
        return any(
            bid != "8" and bid in balls and balls[bid].state.s != 4
            for bid in my_targets
        )

    def _is_aiming_directly_at_eight(self, action: ActionDict, balls: BallDict) -> bool:
        cue_ball = balls.get("cue")
        eight_ball = balls.get("8")
        if cue_ball is None or eight_ball is None or eight_ball.state.s == 4:
            return False
        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        eight_pos = np.array(eight_ball.state.rvw[0][:2])
        direction = np.array([
            math.cos(math.radians(action["phi"])),
            math.sin(math.radians(action["phi"])),
        ])
        vec_to_eight = eight_pos - cue_pos
        if np.linalg.norm(vec_to_eight) <= 1e-6:
            return True
        direction /= np.linalg.norm(direction)
        vec_to_eight /= np.linalg.norm(vec_to_eight)
        dot = float(np.clip(np.dot(direction, vec_to_eight), -1.0, 1.0))
        angle = math.degrees(math.acos(dot))
        return angle < self.safety_angle_threshold

    def _score_shot(self, shot: pt.System, last_state: BallDict, my_targets: List[str]) -> float:  # type: ignore[override]
        base_score = super()._score_shot(shot, last_state, my_targets)
        if self._premature_eight_pocketed(shot, last_state, my_targets):
            # 父类已经扣除了约 420 分，这里叠加到 -1000 左右
            base_score -= max(0.0, self.premature_eight_penalty - 420.0)
        return base_score

    def _premature_eight_pocketed(
        self,
        shot: pt.System,
        last_state: BallDict,
        my_targets: List[str],
    ) -> bool:
        eight_before = last_state.get("8")
        eight_after = shot.balls.get("8")
        if eight_before is None or eight_after is None:
            return False
        if eight_before.state.s == 4 or eight_after.state.s != 4:
            return False
        remaining = [
            bid
            for bid in my_targets
            if bid != "8" and bid in last_state and last_state[bid].state.s != 4
        ]
        return bool(remaining)
