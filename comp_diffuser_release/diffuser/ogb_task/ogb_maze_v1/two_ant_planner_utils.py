import numpy as np


TWO_ANT_OBS_DIM = 71
SINGLE_ANTSOCCER_OBS_DIM = 42

TWO_ANT_ACTION_DIM = 16
SINGLE_ANT_ACTION_DIM = 8


def two_ant_obs_to_single_ant_obs(obs71, active_agent_id):
    """
    two-ant obs71:
        qpos:
            0:15    ant1 qpos
            15:30   ant2 qpos
            30:37   ball qpos

        qvel:
            37:51   ant1 qvel
            51:65   ant2 qvel
            65:71   ball qvel

    single AntSoccer obs42:
        0:15    ant qpos
        15:22   ball qpos
        22:36   ant qvel
        36:42   ball qvel
    """
    obs71 = np.asarray(obs71)

    if obs71.shape[-1] != TWO_ANT_OBS_DIM:
        raise ValueError(f"Expected two-ant obs dim 71, got {obs71.shape}")

    if active_agent_id == 1:
        ant_qpos = obs71[..., 0:15]
        ant_qvel = obs71[..., 37:51]
    elif active_agent_id == 2:
        ant_qpos = obs71[..., 15:30]
        ant_qvel = obs71[..., 51:65]
    else:
        raise ValueError(f"active_agent_id must be 1 or 2, got {active_agent_id}")

    ball_qpos = obs71[..., 30:37]
    ball_qvel = obs71[..., 65:71]

    obs42 = np.concatenate(
        [ant_qpos, ball_qpos, ant_qvel, ball_qvel],
        axis=-1,
    )

    if obs42.shape[-1] != SINGLE_ANTSOCCER_OBS_DIM:
        raise RuntimeError(f"Expected converted obs dim 42, got {obs42.shape}")

    return obs42


def single_ant_action_to_two_ant_action(action8, active_agent_id):
    action8 = np.asarray(action8, dtype=np.float32)

    if action8.shape[-1] != SINGLE_ANT_ACTION_DIM:
        raise ValueError(f"Expected action dim 8, got {action8.shape}")

    action16 = np.zeros(TWO_ANT_ACTION_DIM, dtype=np.float32)

    if active_agent_id == 1:
        action16[:8] = action8
    elif active_agent_id == 2:
        action16[8:] = action8
    else:
        raise ValueError(f"active_agent_id must be 1 or 2, got {active_agent_id}")

    return action16


def set_two_ant_env_from_single_ant_obs(env, obs42, active_agent_id, inactive_ant_xy=None):
    """
    Put the two-ant env into a state that corresponds to a single-ant AntSoccer obs42.

    The active ant receives the original ant state.
    The ball receives the original ball state.
    The inactive ant is placed away from the ball.
    """
    obs42 = np.asarray(obs42)

    if obs42.shape[-1] != SINGLE_ANTSOCCER_OBS_DIM:
        raise ValueError(f"Expected single AntSoccer obs dim 42, got {obs42.shape}")
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    print(
        "[set_two_ant_env_from_single_ant_obs] "
        f"model.nq={base_env.model.nq}, "
        f"model.nv={base_env.model.nv}, "
        f"model.nu={base_env.model.nu}, "
        f"qpos.shape={base_env.data.qpos.shape}, "
        f"qvel.shape={base_env.data.qvel.shape}"
    )

    if base_env.model.nq != 37 or base_env.model.nv != 34 or base_env.model.nu != 16:
        raise RuntimeError(
            "Expected real two-ant env with nq=37, nv=34, nu=16, "
            f"but got nq={base_env.model.nq}, nv={base_env.model.nv}, nu={base_env.model.nu}. "
            "This means the planner is still using the original single-ant env."
        )
    if inactive_ant_xy is None:
        inactive_ant_xy = np.array([4.0, 4.0], dtype=np.float64)

    qpos = env.data.qpos.copy()
    qvel = np.zeros_like(env.data.qvel)

    ant_qpos = obs42[0:15]
    ball_qpos = obs42[15:22]
    ant_qvel = obs42[22:36]
    ball_qvel = obs42[36:42]

    if active_agent_id == 1:
        qpos[0:15] = ant_qpos
        qvel[0:14] = ant_qvel

        qpos[15:17] = inactive_ant_xy
        qvel[14:28] = 0.0

    elif active_agent_id == 2:
        qpos[15:30] = ant_qpos
        qvel[14:28] = ant_qvel

        qpos[0:2] = inactive_ant_xy
        qvel[0:14] = 0.0

    else:
        raise ValueError(f"active_agent_id must be 1 or 2, got {active_agent_id}")

    qpos[30:37] = ball_qpos
    qvel[28:34] = ball_qvel

    env.set_state(qpos, qvel)


def two_ant_xy_from_obs(obs71):
    obs71 = np.asarray(obs71)

    if obs71.shape[-1] != TWO_ANT_OBS_DIM:
        raise ValueError(f"Expected two-ant obs dim 71, got {obs71.shape}")

    ant1_xy = obs71[..., 0:2]
    ant2_xy = obs71[..., 15:17]
    ball_xy = obs71[..., 30:32]

    return ant1_xy, ant2_xy, ball_xy