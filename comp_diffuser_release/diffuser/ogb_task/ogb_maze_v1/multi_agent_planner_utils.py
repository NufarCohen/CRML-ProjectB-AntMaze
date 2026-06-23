import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


ANT_QPOS_DIM = 15
ANT_QVEL_DIM = 14
ANT_ACTION_DIM = 8

BALL_QPOS_DIM = 7
BALL_QVEL_DIM = 6

SINGLE_AGENT_OBS_DIM = 42


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    qpos_slice: slice
    qvel_obs_slice: slice
    qvel_data_slice: slice
    action_slice: slice


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    qpos_slice: slice
    qvel_obs_slice: slice
    qvel_data_slice: slice


@dataclass(frozen=True)
class MultiAgentLayout:
    num_agents: int
    agents: Dict[int, AgentSpec]
    ball: ObjectSpec
    qpos_dim: int
    qvel_dim: int
    obs_dim: int
    action_dim: int


def make_multi_agent_layout(num_agents: int) -> MultiAgentLayout:
    """
    Generic flat layout for AntSoccer with N ants + one ball.

    Observation layout:
        [all qpos, all qvel]

    qpos:
        ant1 qpos, ant2 qpos, ..., antN qpos, ball qpos

    qvel:
        ant1 qvel, ant2 qvel, ..., antN qvel, ball qvel

    action:
        ant1 action, ant2 action, ..., antN action
    """
    if num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {num_agents}")

    qpos_agents_dim = num_agents * ANT_QPOS_DIM
    qvel_agents_dim = num_agents * ANT_QVEL_DIM

    qpos_dim = qpos_agents_dim + BALL_QPOS_DIM
    qvel_dim = qvel_agents_dim + BALL_QVEL_DIM

    obs_dim = qpos_dim + qvel_dim
    action_dim = num_agents * ANT_ACTION_DIM

    qvel_obs_offset = qpos_dim

    agents = {}

    for agent_id in range(1, num_agents + 1):
        agent_idx = agent_id - 1

        qpos_start = agent_idx * ANT_QPOS_DIM
        qpos_end = qpos_start + ANT_QPOS_DIM

        qvel_data_start = agent_idx * ANT_QVEL_DIM
        qvel_data_end = qvel_data_start + ANT_QVEL_DIM

        qvel_obs_start = qvel_obs_offset + qvel_data_start
        qvel_obs_end = qvel_obs_offset + qvel_data_end

        action_start = agent_idx * ANT_ACTION_DIM
        action_end = action_start + ANT_ACTION_DIM

        agents[agent_id] = AgentSpec(
            agent_id=agent_id,
            qpos_slice=slice(qpos_start, qpos_end),
            qvel_obs_slice=slice(qvel_obs_start, qvel_obs_end),
            qvel_data_slice=slice(qvel_data_start, qvel_data_end),
            action_slice=slice(action_start, action_end),
        )

    ball_qpos_start = qpos_agents_dim
    ball_qpos_end = ball_qpos_start + BALL_QPOS_DIM

    ball_qvel_data_start = qvel_agents_dim
    ball_qvel_data_end = ball_qvel_data_start + BALL_QVEL_DIM

    ball_qvel_obs_start = qvel_obs_offset + ball_qvel_data_start
    ball_qvel_obs_end = qvel_obs_offset + ball_qvel_data_end

    ball = ObjectSpec(
        name="ball",
        qpos_slice=slice(ball_qpos_start, ball_qpos_end),
        qvel_obs_slice=slice(ball_qvel_obs_start, ball_qvel_obs_end),
        qvel_data_slice=slice(ball_qvel_data_start, ball_qvel_data_end),
    )

    return MultiAgentLayout(
        num_agents=num_agents,
        agents=agents,
        ball=ball,
        qpos_dim=qpos_dim,
        qvel_dim=qvel_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


def validate_env_matches_layout(env, layout: MultiAgentLayout) -> None:
    model = env.model

    errors = []
    if model.nq != layout.qpos_dim:
        errors.append(f"nq env={model.nq}, layout={layout.qpos_dim}")
    if model.nv != layout.qvel_dim:
        errors.append(f"nv env={model.nv}, layout={layout.qvel_dim}")
    if model.nu != layout.action_dim:
        errors.append(f"nu env={model.nu}, layout={layout.action_dim}")

    if errors:
        raise RuntimeError(
            "Environment does not match multi-agent layout: "
            + " | ".join(errors)
        )


def get_agent_single_obs_from_multi_obs(
    obs_multi: np.ndarray,
    layout: MultiAgentLayout,
    agent_id: int,
) -> np.ndarray:
    """
    Convert multi-agent obs to the original single-ant obs42.

    obs42:
        0:15     selected ant qpos
        15:22    ball qpos
        22:36    selected ant qvel
        36:42    ball qvel
    """
    if agent_id not in layout.agents:
        raise ValueError(
            f"Unknown agent_id={agent_id}. Available={list(layout.agents.keys())}"
        )

    obs_multi = np.asarray(obs_multi, dtype=np.float32)

    if obs_multi.shape[-1] != layout.obs_dim:
        raise ValueError(
            f"Expected multi-agent obs dim {layout.obs_dim}, "
            f"got {obs_multi.shape[-1]}"
        )

    agent = layout.agents[agent_id]

    obs_single = np.zeros(SINGLE_AGENT_OBS_DIM, dtype=np.float32)
    obs_single[0:15] = obs_multi[agent.qpos_slice]
    obs_single[15:22] = obs_multi[layout.ball.qpos_slice]
    obs_single[22:36] = obs_multi[agent.qvel_obs_slice]
    obs_single[36:42] = obs_multi[layout.ball.qvel_obs_slice]

    return obs_single


def compose_multi_agent_action(
    actions_by_agent: Dict[int, Optional[np.ndarray]],
    layout: MultiAgentLayout,
) -> np.ndarray:
    """
    Compose joint action from per-agent action8.

    None means zero action for that agent.
    """
    joint_action = np.zeros(layout.action_dim, dtype=np.float32)

    for agent_id, action in actions_by_agent.items():
        if agent_id not in layout.agents:
            raise ValueError(
                f"Unknown agent_id={agent_id}. Available={list(layout.agents.keys())}"
            )

        if action is None:
            continue

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (ANT_ACTION_DIM,):
            raise ValueError(
                f"Agent {agent_id}: expected action shape "
                f"({ANT_ACTION_DIM},), got {action.shape}"
            )

        joint_action[layout.agents[agent_id].action_slice] = action

    return joint_action


def make_single_active_joint_action(
    action8: np.ndarray,
    active_agent_id: int,
    layout: MultiAgentLayout,
) -> np.ndarray:
    actions_by_agent = {agent_id: None for agent_id in layout.agents}
    actions_by_agent[active_agent_id] = action8
    return compose_multi_agent_action(actions_by_agent, layout)


def get_agent_qpos(obs_multi: np.ndarray, layout: MultiAgentLayout, agent_id: int) -> np.ndarray:
    return np.asarray(obs_multi[layout.agents[agent_id].qpos_slice], dtype=np.float32).copy()


def get_agent_qvel(obs_multi: np.ndarray, layout: MultiAgentLayout, agent_id: int) -> np.ndarray:
    return np.asarray(obs_multi[layout.agents[agent_id].qvel_obs_slice], dtype=np.float32).copy()


def get_ball_qpos(obs_multi: np.ndarray, layout: MultiAgentLayout) -> np.ndarray:
    return np.asarray(obs_multi[layout.ball.qpos_slice], dtype=np.float32).copy()


def get_ball_qvel(obs_multi: np.ndarray, layout: MultiAgentLayout) -> np.ndarray:
    return np.asarray(obs_multi[layout.ball.qvel_obs_slice], dtype=np.float32).copy()


def get_agent_xy(obs_multi: np.ndarray, layout: MultiAgentLayout, agent_id: int) -> np.ndarray:
    return get_agent_qpos(obs_multi, layout, agent_id)[0:2]


def get_ball_xy(obs_multi: np.ndarray, layout: MultiAgentLayout) -> np.ndarray:
    return get_ball_qpos(obs_multi, layout)[0:2]


def get_multi_agent_world_state(obs_multi: np.ndarray, layout: MultiAgentLayout) -> dict:
    obs_multi = np.asarray(obs_multi, dtype=np.float32)

    if obs_multi.shape[-1] != layout.obs_dim:
        raise ValueError(
            f"Expected multi-agent obs dim {layout.obs_dim}, "
            f"got {obs_multi.shape[-1]}"
        )

    return {
        "num_agents": layout.num_agents,
        "agent_xy": {
            agent_id: get_agent_xy(obs_multi, layout, agent_id)
            for agent_id in layout.agents
        },
        "ball_xy": get_ball_xy(obs_multi, layout),
        "agent_qpos": {
            agent_id: get_agent_qpos(obs_multi, layout, agent_id)
            for agent_id in layout.agents
        },
        "agent_qvel": {
            agent_id: get_agent_qvel(obs_multi, layout, agent_id)
            for agent_id in layout.agents
        },
        "ball_qpos": get_ball_qpos(obs_multi, layout),
        "ball_qvel": get_ball_qvel(obs_multi, layout),
    }


def set_multi_agent_env_from_single_agent_obs(
    env,
    single_obs: np.ndarray,
    layout: MultiAgentLayout,
    active_agent_id: int,
    inactive_agent_xy: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    Initialize an N-agent env from original single-ant AntSoccer obs42.

    The selected active agent receives the single-ant state.
    The ball receives the single-ant ball state.
    Other agents keep their qpos unless inactive_agent_xy is provided;
    their velocities are zeroed.
    """
    if active_agent_id not in layout.agents:
        raise ValueError(
            f"Unknown active_agent_id={active_agent_id}. "
            f"Available={list(layout.agents.keys())}"
        )

    single_obs = np.asarray(single_obs, dtype=np.float32)

    if single_obs.shape[-1] != SINGLE_AGENT_OBS_DIM:
        raise ValueError(
            f"Expected single-agent obs dim {SINGLE_AGENT_OBS_DIM}, "
            f"got {single_obs.shape[-1]}"
        )

    inactive_agent_xy = inactive_agent_xy or {}

    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()

    if qpos.shape[0] != layout.qpos_dim:
        raise ValueError(f"qpos dim mismatch: env={qpos.shape[0]}, layout={layout.qpos_dim}")
    if qvel.shape[0] != layout.qvel_dim:
        raise ValueError(f"qvel dim mismatch: env={qvel.shape[0]}, layout={layout.qvel_dim}")

    active_agent = layout.agents[active_agent_id]

    qpos[active_agent.qpos_slice] = single_obs[0:15]
    qvel[active_agent.qvel_data_slice] = single_obs[22:36]

    qpos[layout.ball.qpos_slice] = single_obs[15:22]
    qvel[layout.ball.qvel_data_slice] = single_obs[36:42]

    for agent_id, agent in layout.agents.items():
        if agent_id == active_agent_id:
            continue

        if agent_id in inactive_agent_xy:
            xy = np.asarray(inactive_agent_xy[agent_id], dtype=np.float32)
            if xy.shape != (2,):
                raise ValueError(
                    f"inactive_agent_xy[{agent_id}] must have shape (2,), got {xy.shape}"
                )
            qpos[agent.qpos_slice.start:agent.qpos_slice.start + 2] = xy

        qvel[agent.qvel_data_slice] = 0.0

    env.set_state(qpos, qvel)
    return env.get_ob().copy()
