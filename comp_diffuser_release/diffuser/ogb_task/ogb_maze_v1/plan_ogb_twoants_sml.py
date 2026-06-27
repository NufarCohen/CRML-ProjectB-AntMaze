import sys
import os

sys.path.append('./')

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

import copy
import os.path as osp
from datetime import datetime

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.datasets.d4rl import Is_OgB_Robot_Env
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_multi_agent_planner_v1 import (
    OgB_Stgl_Sml_MultiAgents_MazeEnvPlanner_V1,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)

np.set_printoptions(precision=3, suppress=True)


class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    pl_seeds: str = '-1'
    plan_n_ep: int = -100


def main(args_train, args):
    """Run the two-ant sequential handoff planner."""
    ld_config = dict()

    planner = OgB_Stgl_Sml_MultiAgents_MazeEnvPlanner_V1(args_train, args=args)
    planner.setup_load(ld_config=ld_config)

    print(
        "[TwoAnt Relay Planner]\n"
        f"  multi_agent_env_name     = {args.multi_agent_env_name}\n"
        f"  ma_mode                  = {args.ma_mode}\n"
        f"  carrier_order            = {args.carrier_order}\n"
        f"  handoff_xy               = ({args.handoff_x}, {args.handoff_y})\n"
        f"  ball_handoff_threshold   = {args.ball_handoff_threshold}\n"
        f"  ball_goal_threshold      = {args.ball_goal_threshold}\n"
        f"  ep_st_idx                = {args.ep_st_idx}\n"
        f"  n_act_per_waypnt         = {args.n_act_per_waypnt}\n"
        f"  is_replan                = {args.is_replan}\n"
    )

    pl_seeds = args.pl_seeds

    if len(pl_seeds) == 1:
        if Is_OgB_Robot_Env:
            if pl_seeds[0] == -1:
                avg_result_dict = planner.ogb_plan_once(pl_seed=None)
            else:
                avg_result_dict = planner.ogb_plan_once(pl_seed=pl_seeds[0])
        else:
            utils.print_color(f'{args.pl_seeds=}')
            raise NotImplementedError
    else:
        raise NotImplementedError("This launch file currently supports one planning seed only.")

    try:
        planner.env.close()
    except Exception:
        pass

    try:
        del planner.env
    except Exception:
        pass

    try:
        del planner.renderer.env
    except Exception:
        pass

    return avg_result_dict


def configure_common_eval_args(args, args_train):
    """Common sampling / saving config used by the original planner."""
    loadpath = args.logbase, args.dataset, args_train.exp_name
    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds)

    args.n_batch_acc_probs = 4
    args.is_replan = None
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False
    args.is_rd_agv = False

    args.b_size_per_prob = 40
    args.ev_top_n = 5
    args.ev_pick_type = 'first'
    args.tjb_blend_type = 'exp'
    args.tjb_exp_beta = 2

    args.var_temp = 1.0
    args.cond_w = 2.0
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50

    return loadpath


def configure_two_ant_soccer_relay(args, args_train):
    """
    Configure Mode B sequential handoff:

        Ant1 carries ball to handoff_xy.
        Ant2 automatically wakes up when ball is inside ball_handoff_threshold.
        Ant2 carries ball from handoff_xy to final_goal_xy.

    Important:
        final_goal_xy is still loaded from the OGBench eval problem gl_pos[15:17].
        The handoff point is manually defined here by handoff_x/handoff_y.
    """
    dfu_ndim = len(args_train.dataset_config['obs_select_dim'])

    if 'antsoccer' not in args.dataset.lower():
        raise NotImplementedError(
            "This launch file is intentionally only for AntSoccer two-ant relay. "
            f"Got dataset={args.dataset}"
        )

    if 'arena' not in args.dataset.lower():
        raise NotImplementedError(
            "This launch file was prepared for antsoccer arena datasets. "
            f"Got dataset={args.dataset}"
        )

    if dfu_ndim not in [4, 17]:
        raise NotImplementedError(
            f"Expected AntSoccer diffusion dim 4 or 17, got dfu_ndim={dfu_ndim}"
        )

    # Real rollout env: two ants, one ball.
    args.num_agents = 2
    args.multi_agent_env_name = "antsoccer-twoants-arena-v0"

    # Mode B: only one active carrier at a time.
    # Ant2 wakes up only after the ball reaches the handoff radius.
    args.ma_mode = "mode_b_sequential_handoff"
    args.carrier_order = [1, 2]
    args.initial_active_agent_id = 1

    # Manual middle point for the relay.
    # Change these two numbers to control where Ant1 should bring the ball.
    args.handoff_x = 12.0
    args.handoff_y = 12.0

    # Wake-up radius for Ant2.
    # If Ant2 never wakes up, increase this to 3.5 or 4.0.
    # If Ant2 wakes too early, decrease this to 2.0 or 2.5.
    args.ball_handoff_threshold = 3.0

    # Success radius around final target.
    args.ball_goal_threshold = 2.0

    # Evaluation problem index.
    # This picks start_state[i] and final goal_pos[i] from the OGBench HDF5.
    # It is not itself a coordinate.
    args.ep_st_idx = 12

    args.ev_n_comp = 5
    args.ev_cp_infer_t_type = 'interleave'
    args.is_use_subgoal_marker = False

    args.is_replan = 'ada_dist'
    args.n_act_per_waypnt = 1
    args.repl_ada_dist_cfg = dict(
        max_n_repl=10,
        thres=4,
        type='m_2',
        ada_dist_minus_n_wp=50,
        cond_2_extra=50,
        n_max_steps=5000,
        used_idxs=(0, 1),
    )

    args.inv_epoch = 'latest'
    args.is_inv_train_mode = True
    args.repl_wp_cfg = {}

    print(
        "[configure_two_ant_soccer_relay]\n"
        f"  dfu_ndim                = {dfu_ndim}\n"
        f"  carrier_order          = {args.carrier_order}\n"
        f"  handoff_xy             = ({args.handoff_x}, {args.handoff_y})\n"
        f"  handoff_radius         = {args.ball_handoff_threshold}\n"
        f"  goal_radius            = {args.ball_goal_threshold}\n"
        f"  ep_st_idx              = {args.ep_st_idx}\n"
    )


if __name__ == '__main__':
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')

    assert Is_OgB_Robot_Env

    loadpath = configure_common_eval_args(args, args_train)
    configure_two_ant_soccer_relay(args, args_train)

    latest_e = utils.get_latest_epoch(loadpath)
    depoch_list = [latest_e]

    if args.is_replan == 'ada_dist':
        args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']
    else:
        args.env_n_max_steps = None

    sub_dir = (
        f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}'
        f'-twoantRelay'
        f'-nm{int(args.plan_n_ep)}'
        f'-ems{args.env_n_max_steps // 1000}k'
        f'-ncp{args.ev_n_comp}'
        f'-{args.ev_cp_infer_t_type}'
        f'-handoff{args.handoff_x:.1f}_{args.handoff_y:.1f}'
        f'-rad{args.ball_handoff_threshold:.1f}'
        f'-evSd{",".join([str(sd) for sd in args.pl_seeds])}'
    )

    if args.is_save_pkl:
        sub_dir += '-pkl'
    if hasattr(args, 'ep_st_idx'):
        sub_dir += f'-st{args.ep_st_idx}'
    if args.is_rd_agv:
        sub_dir += '-agv'

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for epoch in depoch_list:
        args_train.diffusion_epoch = epoch
        args.diffusion_epoch = epoch
        result = main(copy.deepcopy(args_train), copy.deepcopy(args))
        result_list.append(result)
