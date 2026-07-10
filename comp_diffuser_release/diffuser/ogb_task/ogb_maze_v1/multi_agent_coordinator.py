import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MultiAgentRelayConfig:
    carrier_order: List[int]
    handoff_points: List[np.ndarray]
    final_goal_xy: np.ndarray
    retreat_points: List[np.ndarray] = field(default_factory=list)
    retreat_threshold: float = 2.0
    ball_handoff_threshold: float = 3.0
    ball_goal_threshold: float = 2.0

    # Backward compatibility for the original two-ant launch scripts.
    retreat_xy: Optional[np.ndarray] = None
    ant1_retreat_threshold: Optional[float] = None


class MultiAgentRelayCoordinator:
    ROLE_IDLE = "idle"
    ROLE_CARRY_TO_HANDOFF = "carry_ball_to_handoff"
    ROLE_RETREAT = "agent_retreating"
    ROLE_CARRY_TO_FINAL = "carry_ball_to_final"

    STAGE_DONE = "done"

    def __init__(self, config: MultiAgentRelayConfig, mode: str = "mode_b_sequential_handoff"):
        self.config = config
        self.mode = mode
        self._validate_config()
        self.reset()

    def _validate_config(self):
        self.config.final_goal_xy = np.asarray(self.config.final_goal_xy, dtype=np.float32)
        self.config.handoff_points = [
            np.asarray(p, dtype=np.float32).reshape(2) for p in self.config.handoff_points
        ]

        num_agents = len(self.config.carrier_order)
        expected_handoffs = max(num_agents - 1, 0)
        if len(self.config.handoff_points) != expected_handoffs:
            raise ValueError(
                f"Expected {expected_handoffs} handoff points for "
                f"{num_agents} agents, got {len(self.config.handoff_points)}"
            )

        if self.config.retreat_xy is not None:
            self.config.retreat_xy = np.asarray(self.config.retreat_xy, dtype=np.float32).reshape(2)

        if not self.config.retreat_points:
            if self.config.retreat_xy is not None:
                default_retreat = self.config.retreat_xy.copy()
            else:
                default_retreat = np.array([2.0, 14.0], dtype=np.float32)
            self.config.retreat_points = [
                default_retreat.copy() for _ in range(max(num_agents - 1, 0))
            ]
        else:
            self.config.retreat_points = [
                np.asarray(p, dtype=np.float32).reshape(2) for p in self.config.retreat_points
            ]

        if len(self.config.retreat_points) != max(num_agents - 1, 0):
            raise ValueError(
                f"Expected {max(num_agents - 1, 0)} retreat points for "
                f"{num_agents} agents, got {len(self.config.retreat_points)}"
            )

        if self.config.ant1_retreat_threshold is not None:
            self.config.retreat_threshold = float(self.config.ant1_retreat_threshold)

    def reset(self):
        self.segment_idx = 0
        self.phase = "carry"
        self.done = False
        self.last_ball_to_target_dist = None
        self.last_switch_info = None

    @property
    def current_stage_name(self) -> str:
        if self.done:
            return self.STAGE_DONE

        carrier_id = self.config.carrier_order[self.segment_idx]
        if self.phase == "retreat":
            return f"ant{carrier_id}_retreat"

        if self.segment_idx < len(self.config.handoff_points):
            return f"ant{carrier_id}_to_handoff_{self.segment_idx + 1}"

        return f"ant{carrier_id}_to_final"

    @property
    def current_carrier_id(self) -> Optional[int]:
        if self.done:
            return None
        return self.config.carrier_order[self.segment_idx]

    @property
    def current_target_xy(self) -> Optional[np.ndarray]:
        if self.done:
            return None

        if self.phase == "retreat":
            return self.config.retreat_points[self.segment_idx]

        if self.segment_idx < len(self.config.handoff_points):
            return self.config.handoff_points[self.segment_idx]

        return self.config.final_goal_xy

    def update(self, world_state: dict):
        self.last_switch_info = None
        if self.done:
            return

        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32)
        carrier_id = self.current_carrier_id
        agent_xy = np.asarray(world_state["agent_xy"][carrier_id], dtype=np.float32)
        prev_stage = self.current_stage_name

        if self.phase == "carry":
            target_xy = self.current_target_xy
            dist = float(np.linalg.norm(ball_xy - target_xy))
            self.last_ball_to_target_dist = dist

            if self.segment_idx < len(self.config.handoff_points):
                if dist <= self.config.ball_handoff_threshold:
                    self.phase = "retreat"
                    print(
                        f"[MA Coordinator] Ball reached handoff {self.segment_idx + 1} "
                        f"(dist={dist:.2f}). Ant {carrier_id} retreating..."
                    )
            else:
                if dist <= self.config.ball_goal_threshold:
                    self.done = True
                    self.phase = "carry"
                    print(
                        f"[MA Coordinator] Ball reached final goal (dist={dist:.2f}). "
                        "Relay complete."
                    )

        elif self.phase == "retreat":
            target_xy = self.current_target_xy
            dist = float(np.linalg.norm(agent_xy - target_xy))
            self.last_ball_to_target_dist = dist

            if dist <= self.config.retreat_threshold:
                next_carrier_id = self.config.carrier_order[self.segment_idx + 1]
                self.segment_idx += 1
                self.phase = "carry"
                print(
                    f"[MA Coordinator] Ant {carrier_id} parked (dist={dist:.2f}). "
                    f"Ant {next_carrier_id} WAKING UP!"
                )

        if prev_stage != self.current_stage_name or (
            prev_stage != self.STAGE_DONE and self.done
        ):
            self.last_switch_info = {"switched": True}

    def get_tasks(self, world_state: dict) -> Dict[int, dict]:
        agent_ids = list(world_state["agent_xy"].keys())
        tasks = {
            agent_id: {
                "active": False,
                "role": self.ROLE_IDLE,
                "target_xy": None,
                "stage_name": self.current_stage_name,
                "ball_to_target_dist": self.last_ball_to_target_dist,
            }
            for agent_id in agent_ids
        }

        if self.done:
            return tasks

        carrier_id = self.current_carrier_id
        if self.phase == "retreat":
            role = self.ROLE_RETREAT
        elif self.segment_idx < len(self.config.handoff_points):
            role = self.ROLE_CARRY_TO_HANDOFF
        else:
            role = self.ROLE_CARRY_TO_FINAL

        tasks[carrier_id] = {
            "active": True,
            "role": role,
            "target_xy": self.current_target_xy,
            "stage_name": self.current_stage_name,
            "ball_to_target_dist": self.last_ball_to_target_dist,
        }

        return tasks

    def is_done(self) -> bool:
        return self.done


class ClosestAntRelayCoordinator:
    """
    Parallel closest-ant relay (mode_d).

    At every timestep two roles are assigned dynamically:
        - Carrier: ant closest to the ball pushes it toward the current handoff.
        - Meet: ant closest to the current handoff (other than carrier) goes there
          to receive the ball.

    When the ball reaches a handoff point, advance to the next one (then final).
    Role assignment is recomputed every update(), so the closest ants can change
    as agents move.
    """

    ROLE_CARRY = "carry_ball"
    ROLE_MEET = "move_to_handoff"
    STAGE_DONE = "done"

    def __init__(
        self,
        config: MultiAgentRelayConfig,
        mode: str = "mode_d_closest_ant_parallel",
    ):
        self.config = config
        self.mode = mode
        self._validate_config()
        self.reset()

    def _validate_config(self):
        self.config.final_goal_xy = np.asarray(
            self.config.final_goal_xy, dtype=np.float32
        ).reshape(2)
        self.config.handoff_points = [
            np.asarray(p, dtype=np.float32).reshape(2) for p in self.config.handoff_points
        ]

    def reset(self):
        self.handoff_idx = 0
        self.done = False
        self.carrier_id = None
        self.meet_id = None
        self.last_ball_to_target_dist = None
        self.last_switch_info = None

    @property
    def current_target_xy(self) -> np.ndarray:
        if self.handoff_idx < len(self.config.handoff_points):
            return self.config.handoff_points[self.handoff_idx]
        return self.config.final_goal_xy

    @property
    def current_stage_name(self) -> str:
        if self.done:
            return self.STAGE_DONE
        if self.handoff_idx < len(self.config.handoff_points):
            return f"handoff_{self.handoff_idx + 1}"
        return "to_final"

    def _agent_ids(self, world_state: dict) -> List[int]:
        if self.config.carrier_order:
            return [int(aid) for aid in self.config.carrier_order]
        return [int(aid) for aid in world_state["agent_xy"].keys()]

    def _assign_roles(self, world_state: dict) -> tuple:
        agent_xy = world_state["agent_xy"]
        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32)
        agent_ids = self._agent_ids(world_state)

        carrier_id = min(
            agent_ids,
            key=lambda aid: float(np.linalg.norm(agent_xy[aid] - ball_xy)),
        )

        target_xy = self.current_target_xy
        meet_candidates = [aid for aid in agent_ids if aid != carrier_id]
        if meet_candidates:
            meet_id = min(
                meet_candidates,
                key=lambda aid: float(np.linalg.norm(agent_xy[aid] - target_xy)),
            )
        else:
            meet_id = carrier_id

        return int(carrier_id), int(meet_id)

    def update(self, world_state: dict):
        self.last_switch_info = None
        if self.done:
            return

        prev_carrier = self.carrier_id
        prev_meet = self.meet_id
        self.carrier_id, self.meet_id = self._assign_roles(world_state)

        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32)
        target_xy = self.current_target_xy
        dist = float(np.linalg.norm(ball_xy - target_xy))
        self.last_ball_to_target_dist = dist

        if self.handoff_idx < len(self.config.handoff_points):
            if dist <= self.config.ball_handoff_threshold:
                self.handoff_idx += 1
                print(
                    f"[MA Closest Coordinator] Ball reached handoff "
                    f"{self.handoff_idx}/{len(self.config.handoff_points)} "
                    f"(dist={dist:.2f}). Advancing target."
                )
        elif dist <= self.config.ball_goal_threshold:
            self.done = True
            print(
                f"[MA Closest Coordinator] Ball reached final goal "
                f"(dist={dist:.2f}). Relay complete."
            )

        if (
            prev_carrier != self.carrier_id
            or prev_meet != self.meet_id
            or (prev_carrier is None and self.carrier_id is not None)
        ):
            self.last_switch_info = {
                "switched": True,
                "carrier_id": self.carrier_id,
                "meet_id": self.meet_id,
            }

    def get_tasks(self, world_state: dict) -> Dict[int, dict]:
        agent_ids = self._agent_ids(world_state)
        stage_name = self.current_stage_name
        target_xy = self.current_target_xy

        tasks = {
            agent_id: {
                "active": False,
                "frozen": True,
                "role": "idle",
                "target_xy": None,
                "stage_name": stage_name,
                "ball_to_target_dist": self.last_ball_to_target_dist,
            }
            for agent_id in agent_ids
        }

        if self.done:
            for task in tasks.values():
                task["stage_name"] = self.STAGE_DONE
            return tasks

        if self.carrier_id is None:
            self.carrier_id, self.meet_id = self._assign_roles(world_state)

        tasks[self.carrier_id] = {
            "active": True,
            "frozen": False,
            "role": self.ROLE_CARRY,
            "target_xy": target_xy.copy(),
            "stage_name": f"ant{self.carrier_id}_carry_{stage_name}",
            "ball_to_target_dist": self.last_ball_to_target_dist,
        }

        if self.meet_id != self.carrier_id:
            tasks[self.meet_id] = {
                "active": True,
                "frozen": False,
                "role": self.ROLE_MEET,
                "target_xy": target_xy.copy(),
                "stage_name": f"ant{self.meet_id}_meet_{stage_name}",
                "ball_to_target_dist": self.last_ball_to_target_dist,
            }

        return tasks

    def is_done(self) -> bool:
        return self.done


class ParallelRelayCoordinator:
    """
    Concurrent (non-sequential) two-agent relay coordinator.

    Both agents are active from timestep 0:
        Ant1 carries the ball toward the handoff point, then freezes
        (stops moving) once the ball arrives there.
        Ant2 walks toward the handoff point immediately (to arrive early
        and wait), then switches its target to the final goal once the
        ball has arrived at the handoff point.

    Unlike MultiAgentRelayCoordinator.get_tasks(), which marks exactly
    one agent "active" per step, get_tasks() here always returns both
    agents as active; the per-agent "frozen" flag signals when an agent
    should stop moving (get zero action) instead.

    This is a speed-only change (no sequential hand-off wait) and is
    unrelated to ant-ant collision avoidance.
    """

    STAGE_TO_HANDOFF = "ant2_to_handoff"
    STAGE_TO_FINAL = "ant2_to_final"

    def __init__(self, config: MultiAgentRelayConfig, mode: str = "mode_c_parallel_handoff"):
        self.config = config
        self.mode = mode
        self._validate_config()
        self.reset()

    def _validate_config(self):
        self.config.final_goal_xy = np.asarray(self.config.final_goal_xy, dtype=np.float32)
        self.config.handoff_points = [np.asarray(p, dtype=np.float32) for p in self.config.handoff_points]

    def reset(self):
        self.ant1_frozen = False
        self.done = False
        self.last_ball_to_target_dist = None
        self.last_switch_info = None

    def update(self, world_state: dict):
        self.last_switch_info = None
        if self.done:
            return

        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32)
        was_frozen = self.ant1_frozen

        if not self.ant1_frozen:
            dist = float(np.linalg.norm(ball_xy - self.config.handoff_points[0]))
            self.last_ball_to_target_dist = dist
            if dist <= self.config.ball_handoff_threshold:
                self.ant1_frozen = True
                print(f"[MA Parallel Coordinator] Ball reached handoff zone (dist={dist:.2f}). Ant1 freezing, Ant2 -> final goal.")

        if self.ant1_frozen:
            dist = float(np.linalg.norm(ball_xy - self.config.final_goal_xy))
            self.last_ball_to_target_dist = dist
            if dist <= self.config.ball_goal_threshold:
                self.done = True

        if self.ant1_frozen and not was_frozen:
            self.last_switch_info = {"switched": True}

    def get_tasks(self, world_state: dict) -> Dict[int, dict]:
        agent_ids = list(world_state["agent_xy"].keys())
        carrier1, carrier2 = self.config.carrier_order[0], self.config.carrier_order[1]

        tasks = {
            agent_id: {
                "active": False,
                "frozen": True,
                "role": "idle",
                "target_xy": None,
                "stage_name": "done" if self.done else "idle",
                "ball_to_target_dist": self.last_ball_to_target_dist,
            }
            for agent_id in agent_ids
        }

        if self.done:
            return tasks

        tasks[carrier1] = {
            "active": True,
            "frozen": self.ant1_frozen,
            "role": "carry_ball_to_handoff",
            "target_xy": self.config.handoff_points[0],
            "stage_name": "ant1_to_handoff",
            "ball_to_target_dist": self.last_ball_to_target_dist,
        }
        tasks[carrier2] = {
            "active": True,
            "frozen": False,
            "role": "carry_ball_to_final" if self.ant1_frozen else "move_to_handoff",
            "target_xy": self.config.final_goal_xy if self.ant1_frozen else self.config.handoff_points[0],
            "stage_name": self.STAGE_TO_FINAL if self.ant1_frozen else self.STAGE_TO_HANDOFF,
            "ball_to_target_dist": self.last_ball_to_target_dist,
        }

        return tasks

    def is_done(self) -> bool:
        return self.done
