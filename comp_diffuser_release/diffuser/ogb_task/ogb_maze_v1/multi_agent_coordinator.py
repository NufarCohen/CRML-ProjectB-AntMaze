import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MultiAgentRelayConfig:
    """
    Generic relay config.

    carrier_order:
        Ordered list of agents that will carry the ball.
        Example for two ants: [1, 2]
        Example for four ants: [1, 2, 3, 4]

    handoff_points:
        List of intermediate ball handoff locations.
        Must have len(carrier_order) - 1 elements.

    final_goal_xy:
        Final target for the ball.

    Mode B default:
        Only current carrier is active.
        Receiver waits.
    """
    carrier_order: List[int]
    handoff_points: List[np.ndarray]
    final_goal_xy: np.ndarray
    ball_handoff_threshold: float = 1.0
    ball_goal_threshold: float = 1.0


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
            np.asarray(p, dtype=np.float32)
            for p in self.config.handoff_points
        ]

        if self.mode != self.MODE_B_SEQUENTIAL_HANDOFF:
            raise NotImplementedError(f"Unsupported multi-agent mode for now: {self.mode}")

    def reset(self):
        # stage_idx = index of current carrier in carrier_order.
        # If stage_idx < len(handoff_points), target is handoff_points[stage_idx].
        # Else target is final_goal_xy.
        self.stage_idx = 0
        self.done = False

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

    def update(self, world_state: dict):
        if self.done:
            return

        ball_xy = world_state["ball_xy"]
        target_xy = self.current_target_xy

        dist = np.linalg.norm(ball_xy - target_xy)

        if self.stage_idx < len(self.config.handoff_points):
            threshold = self.config.ball_handoff_threshold
        else:
            threshold = self.config.ball_goal_threshold

        if dist < threshold:
            self.stage_idx += 1

            if self.stage_idx >= len(self.config.carrier_order):
                self.done = True

    def get_tasks(self, world_state: dict) -> Dict[int, dict]:
        """
        Returns a task dict for every agent in world_state.

        In Mode B:
            exactly one active carrier at a time.
            all other agents are idle.
        """
        agent_ids = list(world_state["agent_xy"].keys())

        tasks = {
            agent_id: {
                "active": False,
                "role": self.ROLE_IDLE,
                "target_xy": None,
                "stage_idx": self.stage_idx,
                "stage_name": self.current_stage_name,
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
        }

        return tasks

    def is_done(self) -> bool:
        return self.done