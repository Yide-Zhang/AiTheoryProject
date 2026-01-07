import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pooltool as pt

from .agent import Agent
from .decision_logger import logger, Candidate


ActionDict = Dict[str, float]
BallDict = Dict[str, Any]


class BasePoolAgent(Agent):
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

    def decision(
        self,
        balls: Optional[BallDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
    ) -> ActionDict:
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

        # --- Decision Logging ---
        refined_results.sort(key=lambda item: item[0], reverse=True)
        log_cands: List[Candidate] = []
        for sc, _act, inf in refined_results[:30]:
            log_cands.append(Candidate(
                score=sc,
                strategy=str(inf.get("strategy", "unknown")),
                confidence=float(inf.get("clearance", 0.0)),
                metadata=f"bal={inf.get('_balanced_priority',0.0):.2f}" if "_balanced_priority" in inf else ""
            ))
        
        logger.log_decision(
            step_type=f"Base_{phase}",
            chosen_score=best_score,
            candidates=log_cands,
            best_strategy=str(best_info.get("strategy", "unknown"))
        )
        # ------------------------

        print(
            f"[BasePoolAgent] strategy={best_info.get('strategy')} target={best_info.get('target')} "
            f"pocket={best_info.get('pocket')} score={best_score:.1f}"
        )
        return best_action

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


class BreakShotPlanner:
    """生成针对开球局面的高效候选动作."""

    def __init__(self, agent: "BasePoolAgent") -> None:
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


class BreakAgent(BasePoolAgent):
    """集成常规策略与开球专家的高速混合智能体."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
    ) -> None:
        super().__init__(max_candidates, evaluation_repeats, refine_top_k, refine_trials)
        self.break_expert = BreakShotPlanner(self)
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

        # --- Break Logging ---
        log_cands: List[Candidate] = []
        scored.sort(key=lambda item: item[0], reverse=True)
        for sc, _, inf in scored:
            log_cands.append(Candidate(
                score=sc,
                strategy=str(inf.get("strategy", "break")),
                confidence=float(inf.get("clearance", 1.0)),
                metadata=f"var={inf.get('variant')}"
            ))
        logger.log_decision(
            step_type="Break",
            chosen_score=best[0],
            candidates=log_cands,
            best_strategy="break_expert"
        )
        # ---------------------

        print(
            f"[BreakAgent] variant={best[2].get('variant')} score={best[0]:.1f} strategy=break"
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


class SafetyPlanner:
    """生成以黑8安全为核心的候选动作."""

    def __init__(self, agent: BreakAgent) -> None:
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


class SafetyAgent(BreakAgent):
    """具备黑8安全保护的 Fast Hybrid Agent 版本."""

    def __init__(
        self,
        max_candidates: int = 48,
        evaluation_repeats: int = 3,
        refine_top_k: int = 6,
        refine_trials: int = 8,
    ) -> None:
        super().__init__(max_candidates, evaluation_repeats, refine_top_k, refine_trials)
        self.safe_generator = SafetyPlanner(self)
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

        # --- Safety Logging ---
        log_cands: List[Candidate] = []
        evaluated.sort(key=lambda item: item[0], reverse=True)
        for sc, _, inf in evaluated[:30]:
            log_cands.append(Candidate(
                score=sc,
                strategy=str(inf.get("strategy", "safety")),
                confidence=float(inf.get("clearance", 0.0)),
                metadata=f"tgt={inf.get('target')} var={inf.get('variant')}"
            ))
        logger.log_decision(
            step_type="Safety",
            chosen_score=best_score,
            candidates=log_cands,
            best_strategy=str(best_info.get("strategy"))
        )
        # ----------------------

        if self._needs_eight_protection(balls, my_targets) and self._is_aiming_directly_at_eight(best_action, balls):
            return self._fallback_defensive_action(balls, table)

        print(
            f"[SafetyAgent] safe-mode strategy={best_info.get('strategy')} target={best_info.get('target')} score={best_score:.1f}"
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


@dataclass
class RiskGameState:
    own_remaining: int
    opponent_remaining: int
    easy_shots: int
    cue_risk: float
    eight_danger: float
    offensive_window: float


class RiskPlanner:
    """Generates supplemental actions that balance risk and offense."""

    def __init__(self, agent: SafetyAgent) -> None:
        self._agent = agent
        self.easy_threshold = 0.38
        self.medium_threshold = 0.58

    def analyze_state(
        self,
        balls: BallDict,
        my_targets: Optional[Sequence[str]],
        table: pt.Table,
    ) -> RiskGameState:
        if not balls or my_targets is None:
            return RiskGameState(0, 0, 0, 0.0, 0.0, 0.0)

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

        return RiskGameState(
            own_remaining=own_remaining,
            opponent_remaining=opponent_remaining,
            easy_shots=easy_shots,
            cue_risk=cue_risk,
            eight_danger=eight_danger,
            offensive_window=offensive_window,
        )

    def select_strategy(
        self,
        game_state: RiskGameState,
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
        game_state: RiskGameState,
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
        game_state: RiskGameState,
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
        game_state: RiskGameState,
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


class RiskAgent(SafetyAgent):
    """SafetyAgent with explicit risk-balanced planning."""

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
        self.balanced_planner = RiskPlanner(self)
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
            f"[RiskAgent] strategy={strategy} easy={game_state.easy_shots} "
            f"own={game_state.own_remaining} opp={game_state.opponent_remaining}"
        )
        return merged


class NewAgentFinal(RiskAgent):
    """整合所有功能的最终智能体."""

    pass
