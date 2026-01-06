import copy
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pooltool as pt

from .agent import Agent


ActionDict = Dict[str, float]
BallDict = Dict[str, Any]


class IntegratedAgent(Agent):
    """融合启发式、防守策略与局面评估的智能体."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
    ) -> None:
        super().__init__()
        self.ball_radius = 0.028575
        self.max_candidates = max_candidates
        self.evaluation_repeats = evaluation_repeats
        self.refine_top_k = refine_top_k
        self.refine_trials = refine_trials
        self.internal_noise = {
            "V0": 0.08,
            "phi": 0.12,
            "theta": 0.05,
            "a": 0.002,
            "b": 0.002,
        }
        self.spin_variants: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 0.0),
            (0.0, -0.25, 0.0),
            (0.0, 0.25, 0.0),
            (0.25, 0.0, 0.0),
            (-0.25, 0.0, 0.0),
        ]
        self._initial_group: Optional[Tuple[str, ...]] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def decision(self, balls: Optional[BallDict] = None, my_targets: Optional[List[str]] = None, table: Optional[pt.Table] = None) -> ActionDict:
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        if self._initial_group is None:
            base_group = [bid for bid in my_targets if bid != "8"]
            if base_group:
                self._initial_group = tuple(sorted(base_group))

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        phase = self._classify_phase(balls, my_targets)
        candidates = self._generate_candidates(balls, my_targets, table, phase)

        if not candidates:
            return self._random_action()

        evaluated: List[Tuple[float, ActionDict, Dict[str, str]]] = []
        for action_info in candidates:
            score = self._evaluate_candidate(
                action_info["action"],
                action_info,
                balls,
                table,
                my_targets,
                last_state_snapshot,
            )
            evaluated.append((score, action_info["action"], action_info))

        evaluated.sort(key=lambda item: item[0], reverse=True)
        refined_results = evaluated[:]

        for base_score, base_action, info in evaluated[: self.refine_top_k]:
            for _ in range(self.refine_trials):
                mutated_action = self._mutate_action(base_action, scale=0.25)
                score = self._evaluate_candidate(
                    mutated_action,
                    info,
                    balls,
                    table,
                    my_targets,
                    last_state_snapshot,
                )
                refined_results.append((score, mutated_action, info))

        if not refined_results:
            return self._random_action()

        best_score, best_action, best_info = max(refined_results, key=lambda item: item[0])
        print(
            f"[IntegratedAgent] strategy={best_info.get('strategy')} target={best_info.get('target')} "
            f"pocket={best_info.get('pocket')} score={best_score:.1f}"
        )
        return best_action

    # ------------------------------------------------------------------
    # 候选动作生成
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        balls: BallDict,
        my_targets: List[str],
        table: pt.Table,
        phase: str,
    ) -> List[Dict[str, object]]:
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        targets = self._active_targets(balls, my_targets)
        if not targets:
            targets = ["8"] if "8" in balls else []

        offensive_actions: List[Dict[str, object]] = []
        defensive_actions: List[Dict[str, object]] = []

        for target_id in targets:
            obj_ball = balls.get(target_id)
            if obj_ball is None or obj_ball.state.s == 4:
                continue
            obj_pos = np.array(obj_ball.state.rvw[0][:2])

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array(pocket.center[:2])
                phi, distance, ghost_pos = self._ghost_ball_solution(cue_pos, obj_pos, pocket_pos)
                if distance <= 1e-6:
                    continue

                path_clear, clearance_ratio = self._check_path_clear(
                    cue_pos,
                    ghost_pos,
                    balls,
                    ignore_ids={"cue", target_id},
                )
                base_power = np.clip(1.4 + distance * 1.6, 1.0, 7.6)

                base_action = self._pack_action(
                    base_power,
                    phi,
                    0.0,
                    0.0,
                    0.0,
                )
                offensive_actions.append(
                    {
                        "action": base_action,
                        "strategy": "offense",
                        "target": target_id,
                        "pocket": pocket_id,
                        "clearance": clearance_ratio,
                        "path_clear": path_clear,
                    }
                )

                for offset in (-0.45, 0.45):
                    tweaked = self._pack_action(
                        np.clip(base_power * (1.0 + random.uniform(-0.05, 0.08)), 0.9, 7.7),
                        (phi + offset) % 360,
                        0.0,
                        0.0,
                        0.0,
                    )
                    offensive_actions.append(
                        {
                            "action": tweaked,
                            "strategy": "offense",
                            "target": target_id,
                            "pocket": pocket_id,
                            "clearance": clearance_ratio,
                            "path_clear": path_clear,
                        }
                    )

                for spin_a, spin_b, spin_theta in self.spin_variants[:2]:
                    spin_action = self._pack_action(
                        base_power,
                        phi,
                        spin_theta,
                        spin_a,
                        spin_b,
                    )
                    offensive_actions.append(
                        {
                            "action": spin_action,
                            "strategy": "positional",
                            "target": target_id,
                            "pocket": pocket_id,
                            "clearance": clearance_ratio,
                            "path_clear": path_clear,
                        }
                    )

            if phase in {"early", "mid"}:
                safety_action = self._construct_safety_action(cue_pos, obj_pos)
                if safety_action is not None:
                    defensive_actions.append(
                        {
                            "action": safety_action,
                            "strategy": "safety",
                            "target": target_id,
                            "pocket": None,
                            "clearance": 0.0,
                            "path_clear": True,
                        }
                    )

        all_actions = offensive_actions + defensive_actions
        if not all_actions:
            return []

        all_actions.sort(
            key=lambda info: (
                info["strategy"] != "offense",
                -info.get("clearance", 0.0),
            )
        )
        return all_actions[: self.max_candidates]

    def _construct_safety_action(self, cue_pos: np.ndarray, obj_pos: np.ndarray) -> Optional[ActionDict]:
        direction = obj_pos - cue_pos
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return None
        unit = direction / norm
        retreat_dir = unit * -1.0
        phi = math.degrees(math.atan2(retreat_dir[1], retreat_dir[0])) % 360
        power = np.clip(0.85 + norm * 0.2, 0.7, 1.4)
        return self._pack_action(power, phi, 0.0, 0.0, -0.2)

    # ------------------------------------------------------------------
    # 评分相关
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        action: ActionDict,
        info: Dict[str, object],
        balls: BallDict,
        table: pt.Table,
        my_targets: List[str],
        last_state_snapshot: BallDict,
    ) -> float:
        samples: List[float] = []
        for _ in range(self.evaluation_repeats):
            noisy_action = self._apply_eval_noise(action)
            shot = self._simulate_action(balls, table, noisy_action)
            if shot is None:
                samples.append(-600.0)
                continue

            base_reward = self._score_shot(shot, last_state_snapshot, my_targets)
            positional_value = self._positional_bonus(shot.balls, my_targets, table, info)
            samples.append(base_reward + positional_value)

        return float(np.mean(samples)) if samples else -600.0

    def _simulate_action(self, balls: BallDict, table: pt.Table, action: ActionDict) -> Optional[pt.System]:
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        system = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        try:
            cue.set_state(
                V0=action["V0"],
                phi=action["phi"],
                theta=action["theta"],
                a=action["a"],
                b=action["b"],
            )
            pt.simulate(system, inplace=True)
            return system
        except Exception:
            return None

    def _score_shot(self, shot: pt.System, last_state: BallDict, my_targets: List[str]) -> float:
        new_pocketed = [
            bid
            for bid, ball in shot.balls.items()
            if last_state[bid].state.s != 4 and ball.state.s == 4
        ]

        own_targets = set(my_targets)
        own_pocketed = [bid for bid in new_pocketed if bid in own_targets]
        enemy_pocketed = [
            bid
            for bid in new_pocketed
            if bid not in own_targets and bid not in {"cue", "8"}
        ]

        cue_in = "cue" in new_pocketed
        eight_in = "8" in new_pocketed

        first_contact = self._extract_first_hit(shot)
        foul_first = False
        if first_contact is None:
            if not self._only_eight_left(last_state, my_targets):
                foul_first = True
        elif first_contact not in own_targets:
            foul_first = True

        foul_no_rail = self._check_no_rail_foul(shot, first_contact, new_pocketed)

        score = 0.0
        if cue_in and eight_in:
            return -450.0
        if cue_in:
            score -= 130.0
        if eight_in:
            legal_black = self._only_eight_left(last_state, my_targets)
            score += 180.0 if legal_black else -420.0

        if foul_first:
            score -= 50.0
        if foul_no_rail:
            score -= 35.0

        score += len(own_pocketed) * 58.0
        score -= len(enemy_pocketed) * 18.0

        if score == 0.0 and not cue_in and not eight_in and not foul_first and not foul_no_rail:
            score = 12.0

        return score

    def _positional_bonus(
        self,
        balls_after: BallDict,
        my_targets: List[str],
        table: pt.Table,
        info: Dict[str, object],
    ) -> float:
        cue_ball = balls_after.get("cue")
        if cue_ball is None:
            return 0.0
        cue_pos = np.array(cue_ball.state.rvw[0][:2])

        own_remaining = [
            bid
            for bid in my_targets
            if bid in balls_after and balls_after[bid].state.s != 4 and bid != "8"
        ]
        bonus = 0.0

        if own_remaining:
            distances = [
                np.linalg.norm(cue_pos - np.array(balls_after[bid].state.rvw[0][:2]))
                for bid in own_remaining
            ]
            if distances:
                min_dist = min(distances)
                bonus += (0.6 - min(min_dist, 0.6)) * 40.0

                closest_id = own_remaining[int(np.argmin(distances))]
                target_pos = np.array(balls_after[closest_id].state.rvw[0][:2])
                phi, _, ghost_pos = self._ghost_ball_solution(cue_pos, target_pos, target_pos)
                _, clearance_ratio = self._check_path_clear(
                    cue_pos,
                    ghost_pos,
                    balls_after,
                    ignore_ids={"cue", closest_id},
                )
                bonus += clearance_ratio * 20.0
        else:
            eight_ball = balls_after.get("8")
            if eight_ball is not None and eight_ball.state.s != 4:
                dist_eight = np.linalg.norm(
                    cue_pos - np.array(eight_ball.state.rvw[0][:2])
                )
                bonus += (0.8 - min(dist_eight, 0.8)) * 45.0

        opp_targets = self._infer_opponent_targets()
        if opp_targets:
            opp_remaining = [
                bid
                for bid in opp_targets
                if bid in balls_after and balls_after[bid].state.s != 4
            ]
            if opp_remaining:
                opp_distances = [
                    np.linalg.norm(cue_pos - np.array(balls_after[bid].state.rvw[0][:2]))
                    for bid in opp_remaining
                ]
                if opp_distances:
                    min_opp = min(opp_distances)
                    if min_opp < 0.28:
                        bonus -= (0.28 - min_opp) * 90.0
                    elif min_opp > 0.65:
                        bonus += min(min_opp - 0.65, 0.4) * 45.0

        cushion_dist = self._distance_to_cushion(cue_pos, table)
        if cushion_dist < 0.08:
            bonus += (0.08 - cushion_dist) * 120.0

        if info.get("strategy") == "safety" and bonus > 0:
            bonus *= 1.15

        return float(bonus)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    def _classify_phase(self, balls: BallDict, my_targets: List[str]) -> str:
        remaining = [
            bid
            for bid in my_targets
            if bid != "8" and bid in balls and balls[bid].state.s != 4
        ]
        if not remaining:
            return "eight"
        if len(remaining) >= 5:
            return "early"
        if len(remaining) >= 2:
            return "mid"
        return "late"

    def _active_targets(self, balls: BallDict, my_targets: Iterable[str]) -> List[str]:
        return [
            bid
            for bid in my_targets
            if bid in balls and balls[bid].state.s != 4 and bid != "8"
        ]

    def _ghost_ball_solution(
        self,
        cue_pos: np.ndarray,
        obj_pos: np.ndarray,
        pocket_pos: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        vec_to_pocket = pocket_pos - obj_pos
        dist_obj_to_pocket = np.linalg.norm(vec_to_pocket)
        if dist_obj_to_pocket <= 1e-6:
            return 0.0, 0.0, obj_pos
        unit_vec = vec_to_pocket / dist_obj_to_pocket
        ghost_pos = obj_pos - unit_vec * (2 * self.ball_radius)
        vec_to_ghost = ghost_pos - cue_pos
        distance = np.linalg.norm(vec_to_ghost)
        phi = math.degrees(math.atan2(vec_to_ghost[1], vec_to_ghost[0])) % 360
        return phi, distance, ghost_pos

    def _check_path_clear(
        self,
        cue_pos: np.ndarray,
        ghost_pos: np.ndarray,
        balls: BallDict,
        ignore_ids: Iterable[str],
    ) -> Tuple[bool, float]:
        ignore = set(ignore_ids)
        segment = ghost_pos - cue_pos
        seg_len = np.linalg.norm(segment)
        if seg_len <= 1e-6:
            return True, 1.0
        unit = segment / seg_len
        clearance_penalty = 0.0
        blocked = False
        for bid, ball in balls.items():
            if bid in ignore or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2])
            rel = pos - cue_pos
            proj = np.dot(rel, unit)
            if proj <= 0 or proj >= seg_len:
                continue
            closest = cue_pos + unit * proj
            distance = np.linalg.norm(pos - closest)
            if distance < self.ball_radius * 2.05:
                blocked = True
            clearance_penalty += max(0.0, (self.ball_radius * 2.5 - distance))
        clearance_ratio = math.exp(-clearance_penalty)
        return (not blocked), float(np.clip(clearance_ratio, 0.0, 1.0))

    def _extract_first_hit(self, shot: pt.System) -> Optional[str]:
        valid_ids = {str(i) for i in range(1, 16)} | {"8"}
        for event in shot.events:
            identifiers = getattr(event, "ids", [])
            if "cue" not in identifiers:
                continue
            others = [bid for bid in identifiers if bid != "cue" and bid in valid_ids]
            if others:
                return others[0]
        return None

    def _only_eight_left(self, last_state: BallDict, my_targets: List[str]) -> bool:
        non_eight = [bid for bid in my_targets if bid != "8"]
        if not non_eight:
            return True
        return all(last_state[bid].state.s == 4 for bid in non_eight)

    def _check_no_rail_foul(
        self,
        shot: pt.System,
        first_contact: Optional[str],
        new_pocketed: List[str],
    ) -> bool:
        if first_contact is None or new_pocketed:
            return False
        cue_hit = False
        target_hit = False
        for event in shot.events:
            labels = getattr(event, "ids", [])
            event_type = str(event.event_type).lower()
            if "cushion" not in event_type:
                continue
            if "cue" in labels:
                cue_hit = True
            if first_contact in labels:
                target_hit = True
        return not (cue_hit or target_hit)

    def _pos_from_table(self, table: pt.Table) -> Tuple[float, float, float, float]:
        xs = [pocket.center[0] for pocket in table.pockets.values()]
        ys = [pocket.center[1] for pocket in table.pockets.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _distance_to_cushion(self, position: np.ndarray, table: pt.Table) -> float:
        xmin, xmax, ymin, ymax = self._pos_from_table(table)
        x, y = position
        return min(abs(x - xmin), abs(x - xmax), abs(y - ymin), abs(y - ymax))

    def _pack_action(self, V0: float, phi: float, theta: float, a: float, b: float) -> ActionDict:
        return {
            "V0": float(np.clip(V0, 0.5, 8.0)),
            "phi": float(phi % 360.0),
            "theta": float(np.clip(theta, 0.0, 90.0)),
            "a": float(np.clip(a, -0.5, 0.5)),
            "b": float(np.clip(b, -0.5, 0.5)),
        }

    def _apply_eval_noise(self, action: ActionDict) -> ActionDict:
        noisy = {
            key: action[key] + random.gauss(0.0, self.internal_noise[key])
            for key in ("V0", "phi", "theta", "a", "b")
        }
        noisy["phi"] = noisy["phi"] % 360.0
        return self._pack_action(**noisy)

    def _mutate_action(self, action: ActionDict, scale: float) -> ActionDict:
        mutated = {
            "V0": action["V0"] + random.uniform(-0.5, 0.5) * scale,
            "phi": action["phi"] + random.uniform(-2.5, 2.5) * scale,
            "theta": action["theta"] + random.uniform(-1.5, 1.5) * scale,
            "a": action["a"] + random.uniform(-0.3, 0.3) * scale,
            "b": action["b"] + random.uniform(-0.3, 0.3) * scale,
        }
        return self._pack_action(**mutated)

    def _infer_opponent_targets(self) -> Optional[List[str]]:
        if self._initial_group is None:
            return None
        if not self._initial_group:
            return [str(i) for i in range(1, 8)] + [str(i) for i in range(9, 16)]
        try:
            first_value = int(self._initial_group[0])
        except ValueError:
            return None
        if 1 <= first_value <= 7:
            return [str(i) for i in range(9, 16)]
        return [str(i) for i in range(1, 8)]
