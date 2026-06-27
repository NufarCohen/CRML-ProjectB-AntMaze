import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MultiAgentRelayConfig:
    """
    Generic relay config for sequential ball handoff.

    carrier_order:
        Ordered list of ants that will carry the ball.
        Example for two ants: [1, 2]

    handoff_points:
        Intermediate ball target locations.
        Must have len(carrier_order) - 1 points.
        For two ants this is one point: the middle/handoff point.

    final_goal_xy:
        Final target location of the ball.

    ball_handoff_threshold:
        Radius around the current handoff point. When the ball enters this
        radius, the next ant becomes active automatically.

    ball_goal_threshold:
        Radius around the final goal. When the ball enters this radius, the
        relay is marked done.
    """

    carrier_order: List[int]
    handoff_points: List[np.ndarray]
    final_goal_xy: np.ndarray
    ball_handoff_threshold: float = 3.0
    ball_goal_threshold: float = 2.0


class MultiAgentRelayCoordinator:
    ROLE_IDLE = "idle"
    ROLE_CARRY_TO_HANDOFF = "carry_ball_to_handoff"
    ROLE_CARRY_TO_FINAL = "carry_ball_to_final"

    MODE_B_SEQUENTIAL_HANDOFF = "mode_b_sequential_handoff"
    STAGE_DONE = "done"

    def __init__(self, config: MultiAgentRelayConfig, mode: str = MODE_B_SEQUENTIAL_HANDOFF):
        self.config = config
        self.mode = mode
        self._validate_config()
        self.reset()

    def _validate_config(self):
        if len(self.config.carrier_order) < 1:
            raise ValueError("carrier_order must contain at least one agent")

        expected_handoffs = max(0, len(self.config.carrier_order) - 1)
        if len(self.config.handoff_points) != expected_handoffs:
            raise ValueError(
                f"Expected {expected_handoffs} handoff_points for "
                f"carrier_order={self.config.carrier_order}, "
                f"got {len(self.config.handoff_points)}"
            )

        self.config.final_goal_xy = np.asarray(self.config.final_goal_xy, dtype=np.float32)
        self.config.handoff_points = [
            np.asarray(p, dtype=np.float32) for p in self.config.handoff_points
        ]

        if self.mode != self.MODE_B_SEQUENTIAL_HANDOFF:
            raise NotImplementedError(f"Unsupported multi-agent mode for now: {self.mode}")

        if self.config.ball_handoff_threshold <= 0:
            raise ValueError(
                f"ball_handoff_threshold must be positive, got "
                f"{self.config.ball_handoff_threshold}"
            )
        if self.config.ball_goal_threshold <= 0:
            raise ValueError(
                f"ball_goal_threshold must be positive, got "
                f"{self.config.ball_goal_threshold}"
            )

    def reset(self):
        # stage_idx is the index of the current carrier in carrier_order.
        # If stage_idx < len(handoff_points), the current target is
        # handoff_points[stage_idx]. Otherwise, the current target is final_goal_xy.
        self.stage_idx = 0
        self.done = False
        self.last_ball_to_target_dist = None
        self.last_switch_info = None

    @property
    def current_carrier_id(self) -> Optional[int]:
        if self.done:
            return None
        return self.config.carrier_order[self.stage_idx]

    @property
    def current_target_xy(self) -> Optional[np.ndarray]:
        if self.done:
            return None
        if self.stage_idx < len(self.config.handoff_points):
            return self.config.handoff_points[self.stage_idx]
        return self.config.final_goal_xy

    @property
    def current_stage_name(self) -> str:
        if self.done:
            return self.STAGE_DONE
        if self.stage_idx < len(self.config.handoff_points):
            return f"carrier_{self.current_carrier_id}_to_handoff_{self.stage_idx}"
        return f"carrier_{self.current_carrier_id}_to_final"

    def _current_threshold(self) -> float:
        if self.stage_idx < len(self.config.handoff_points):
            return float(self.config.ball_handoff_threshold)
        return float(self.config.ball_goal_threshold)

    def update(self, world_state: dict):
        """
        Update relay stage according to ball distance from the current target.

        Expected world_state structure:
            {
                "ball_xy": np.ndarray shape (2,),
                "agent_xy": {1: xy, 2: xy, ...}
            }

        In Mode B, this is the automatic wake-up rule:
            If Ant1 is carrying to handoff and
            ||ball_xy - handoff_xy|| <= ball_handoff_threshold,
            then Ant2 becomes active automatically.
        """
        self.last_switch_info = None

        if self.done:
            return

        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32)
        target_xy = np.asarray(self.current_target_xy, dtype=np.float32)
        threshold = self._current_threshold()

        prev_stage_idx = self.stage_idx
        prev_stage_name = self.current_stage_name
        prev_carrier_id = self.current_carrier_id

        dist = float(np.linalg.norm(ball_xy - target_xy))
        self.last_ball_to_target_dist = dist

        if dist <= threshold:
            self.stage_idx += 1

            if self.stage_idx >= len(self.config.carrier_order):
                self.done = True

            self.last_switch_info = {
                "prev_stage_idx": prev_stage_idx,
                "new_stage_idx": self.stage_idx,
                "prev_stage_name": prev_stage_name,
                "new_stage_name": self.current_stage_name,
                "prev_carrier_id": prev_carrier_id,
                "new_carrier_id": self.current_carrier_id,
                "ball_xy": ball_xy.copy(),
                "target_xy": target_xy.copy(),
                "distance": dist,
                "threshold": threshold,
            }

    def get_tasks(self, world_state: dict) -> Dict[int, dict]:
        """
        Returns a task dict for every agent in world_state.

        In Mode B, exactly one carrier is active at a time.
        All other agents are idle.
        """
        agent_ids = list(world_state["agent_xy"].keys())

        tasks = {
            agent_id: {
                "active": False,
                "role": self.ROLE_IDLE,
                "target_xy": None,
                "stage_idx": self.stage_idx,
                "stage_name": self.current_stage_name,
                "ball_to_target_dist": self.last_ball_to_target_dist,
            }
            for agent_id in agent_ids
        }

        if self.done:
            return tasks

        carrier_id = self.current_carrier_id
        target_xy = self.current_target_xy

        if self.stage_idx < len(self.config.handoff_points):
            role = self.ROLE_CARRY_TO_HANDOFF
        else:
            role = self.ROLE_CARRY_TO_FINAL

        tasks[carrier_id] = {
            "active": True,
            "role": role,
            "target_xy": target_xy,
            "stage_idx": self.stage_idx,
            "stage_name": self.current_stage_name,
            "ball_to_target_dist": self.last_ball_to_target_dist,
        }

        return tasks

    def is_done(self) -> bool:
        return self.done
