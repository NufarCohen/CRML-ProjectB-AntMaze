import numpy as np


class MultiAgentRelayCoordinator:
    """
    Mode B sequential handoff coordinator.

    Correct intended behavior:
        1. Ant1 is active.
        2. Ant1 pushes the ball to handoff_xy = (x1, y1).
        3. Only when the BALL is really close to handoff_xy, Ant2 wakes up.
        4. Ant2 first goes to the current ball/handoff location.
        5. Once Ant2 is close enough to the ball, Ant2 pushes it to final_goal_xy.
        6. Done when the BALL is close enough to final_goal_xy.

    Stages:
        ant1_to_handoff
        ant2_to_handoff
        ant2_to_final
        done
    """

    STAGE_ANT1_TO_HANDOFF = "ant1_to_handoff"
    STAGE_ANT2_TO_HANDOFF = "ant2_to_handoff"
    STAGE_ANT2_TO_FINAL = "ant2_to_final"
    STAGE_DONE = "done"

    def __init__(
        self,
        mode="mode_b_sequential_handoff",
        carrier_order=(1, 2),
        handoff_points=None,
        final_goal_xy=None,
        handoff_ball_radius=1.0,
        ant2_reach_ball_radius=1.5,
        ball_goal_threshold=1.0,
        ball_handoff_threshold=None,
        verbose=True,
        **kwargs,
    ):
        self.mode = mode
        self.verbose = bool(verbose)

        self.carrier_order = [int(x) for x in carrier_order]
        if len(self.carrier_order) < 2:
            raise ValueError(
                f"Mode B expects at least two agents. Got carrier_order={self.carrier_order}"
            )

        self.ant1_id = self.carrier_order[0]
        self.ant2_id = self.carrier_order[1]

        if handoff_points is None:
            handoff_points = [np.array([5.0, 2.0], dtype=np.float32)]

        if len(handoff_points) < 1:
            raise ValueError("handoff_points must contain at least one point")

        self.handoff_points = [
            np.asarray(p, dtype=np.float32).reshape(2) for p in handoff_points
        ]
        self.handoff_xy = self.handoff_points[0].copy()

        if final_goal_xy is None:
            raise ValueError("final_goal_xy must be provided")

        self.final_goal_xy = np.asarray(final_goal_xy, dtype=np.float32).reshape(2)

        # Backward compatibility:
        # Old code may still pass ball_handoff_threshold.
        # In the new logic this means the strict BALL-to-handoff radius.
        if ball_handoff_threshold is not None:
            handoff_ball_radius = float(ball_handoff_threshold)

        self.handoff_ball_radius = float(handoff_ball_radius)
        self.ant2_reach_ball_radius = float(ant2_reach_ball_radius)
        self.ball_goal_threshold = float(ball_goal_threshold)

        self.current_stage_name = self.STAGE_ANT1_TO_HANDOFF
        self.active_agent_id = self.ant1_id
        self.target_xy = self.handoff_xy.copy()

        self.stage_switched_last_update = False
        self.last_debug_info = {}

    def reset(self):
        self.current_stage_name = self.STAGE_ANT1_TO_HANDOFF
        self.active_agent_id = self.ant1_id
        self.target_xy = self.handoff_xy.copy()
        self.stage_switched_last_update = False
        self.last_debug_info = {}

    def is_done(self):
        return self.current_stage_name == self.STAGE_DONE

    def get_active_agent_id(self):
        if self.is_done():
            return None
        return self.active_agent_id

    def get_active_task(self):
        if self.is_done():
            return {
                "active": False,
                "role": "done",
                "stage": self.STAGE_DONE,
                "active_agent_id": None,
                "target_xy": None,
                "is_done": True,
            }

        return {
            "active": True,
            "role": self.current_stage_name,
            "stage": self.current_stage_name,
            "active_agent_id": self.active_agent_id,
            "target_xy": np.asarray(self.target_xy, dtype=np.float32).copy(),
            "is_done": False,
        }

    def get_tasks(self):
        """
        Return task dict compatible with the existing planner:
            tasks = {
                1: {"active": True/False, ...},
                2: {"active": True/False, ...},
            }

        Exactly one task is active unless done.
        """
        tasks = {}

        for agent_id in self.carrier_order:
            if self.is_done():
                tasks[agent_id] = {
                    "active": False,
                    "role": "done",
                    "stage": self.STAGE_DONE,
                    "active_agent_id": agent_id,
                    "target_xy": None,
                    "is_done": True,
                }
                continue

            is_active = agent_id == self.active_agent_id

            tasks[agent_id] = {
                "active": bool(is_active),
                "role": self.current_stage_name if is_active else "waiting",
                "stage": self.current_stage_name,
                "active_agent_id": agent_id,
                "target_xy": (
                    None
                    if self.target_xy is None
                    else np.asarray(self.target_xy, dtype=np.float32).copy()
                ),
                "is_done": False,
            }

        return tasks

    def update(self, world_state):
        """
        Expected world_state:
            {
                "ball_xy": np.array([x, y]),
                "agent_xy": {
                    1: np.array([x, y]),
                    2: np.array([x, y]),
                }
            }

        Returns:
            tasks dict.
        """
        ball_xy, agent_xy = self._parse_world_state(world_state)

        ant1_xy = agent_xy[self.ant1_id]
        ant2_xy = agent_xy[self.ant2_id]

        ball_to_handoff = float(np.linalg.norm(ball_xy - self.handoff_xy))
        ant1_to_ball = float(np.linalg.norm(ant1_xy - ball_xy))
        ant2_to_ball = float(np.linalg.norm(ant2_xy - ball_xy))
        ball_to_goal = float(np.linalg.norm(ball_xy - self.final_goal_xy))

        old_stage = self.current_stage_name

        if self.current_stage_name == self.STAGE_ANT1_TO_HANDOFF:
            self.active_agent_id = self.ant1_id
            self.target_xy = self.handoff_xy.copy()

            # Important:
            # The BALL must be really close to x1,y1.
            # Do not use 3.5/4.0 for this in final behavior.
            if ball_to_handoff <= self.handoff_ball_radius:
                self.current_stage_name = self.STAGE_ANT2_TO_HANDOFF
                self.active_agent_id = self.ant2_id
                self.target_xy = ball_xy.copy()

        elif self.current_stage_name == self.STAGE_ANT2_TO_HANDOFF:
            self.active_agent_id = self.ant2_id
            self.target_xy = ball_xy.copy()

            # Ant2 first reaches the ball, then we ask it to push to final.
            if ant2_to_ball <= self.ant2_reach_ball_radius:
                self.current_stage_name = self.STAGE_ANT2_TO_FINAL
                self.active_agent_id = self.ant2_id
                self.target_xy = self.final_goal_xy.copy()

        elif self.current_stage_name == self.STAGE_ANT2_TO_FINAL:
            self.active_agent_id = self.ant2_id
            self.target_xy = self.final_goal_xy.copy()

            if ball_to_goal <= self.ball_goal_threshold:
                self.current_stage_name = self.STAGE_DONE
                self.active_agent_id = None
                self.target_xy = None

        elif self.current_stage_name == self.STAGE_DONE:
            self.active_agent_id = None
            self.target_xy = None

        else:
            raise ValueError(f"Unknown stage: {self.current_stage_name}")

        self.stage_switched_last_update = old_stage != self.current_stage_name

        self.last_debug_info = {
            "old_stage": old_stage,
            "stage": self.current_stage_name,
            "active_agent_id": self.active_agent_id,
            "ball_xy": ball_xy.copy(),
            "ant1_xy": ant1_xy.copy(),
            "ant2_xy": ant2_xy.copy(),
            "handoff_xy": self.handoff_xy.copy(),
            "final_goal_xy": self.final_goal_xy.copy(),
            "ball_to_handoff": ball_to_handoff,
            "ant1_to_ball": ant1_to_ball,
            "ant2_to_ball": ant2_to_ball,
            "ball_to_goal": ball_to_goal,
            "handoff_ball_radius": self.handoff_ball_radius,
            "ant2_reach_ball_radius": self.ant2_reach_ball_radius,
            "ball_goal_threshold": self.ball_goal_threshold,
        }

        if self.stage_switched_last_update and self.verbose:
            print(f"[MA Relay] stage switch: {old_stage} -> {self.current_stage_name}")

        return self.get_tasks()

    def update_and_get_tasks(self, world_state):
        return self.update(world_state)

    def get_stage_switched_last_update(self):
        return self.stage_switched_last_update

    def _parse_world_state(self, world_state):
        if world_state is None:
            raise ValueError("world_state is None")

        if "ball_xy" not in world_state:
            raise KeyError(f"world_state missing 'ball_xy'. keys={world_state.keys()}")

        if "agent_xy" not in world_state:
            raise KeyError(f"world_state missing 'agent_xy'. keys={world_state.keys()}")

        ball_xy = np.asarray(world_state["ball_xy"], dtype=np.float32).reshape(2)

        raw_agent_xy = world_state["agent_xy"]
        agent_xy = {}

        for agent_id in self.carrier_order:
            if agent_id not in raw_agent_xy:
                raise KeyError(
                    f"world_state['agent_xy'] missing agent_id={agent_id}. "
                    f"available={list(raw_agent_xy.keys())}"
                )

            agent_xy[agent_id] = np.asarray(
                raw_agent_xy[agent_id],
                dtype=np.float32,
            ).reshape(2)

        return ball_xy, agent_xy


# Backward-compatible aliases.
MultiAgentCoordinator = MultiAgentRelayCoordinator
RelayCoordinator = MultiAgentRelayCoordinator
ModeBRelayCoordinator = MultiAgentRelayCoordinator
MultiAgentModeBCoordinator = MultiAgentRelayCoordinator