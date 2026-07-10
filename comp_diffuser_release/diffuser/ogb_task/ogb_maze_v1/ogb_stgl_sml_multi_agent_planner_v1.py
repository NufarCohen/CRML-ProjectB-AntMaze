import numpy as np
import gymnasium as gym
import ogbench
from os.path import join
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
import diffuser.datasets as datasets
import diffuser.utils as utils
from datetime import datetime
import os.path as osp
import copy, pdb, json, pdb, torch, os, mujoco, math, socket
from diffuser.guides.render_m2d import Maze2dRenderer_V2
from collections import OrderedDict
from diffuser.datasets.d4rl import Is_Gym_Robot_Env, Is_OgB_Robot_Env
from diffuser.datasets.ogb_dset.ogb_utils import ogb_load_env, ogb_xy_to_ij, \
        ogb_get_rowcol_obs_trajs_from_xy, ogb_get_rowcol_obs_trajs_from_xy_list, ogb_load_env_kwargs
from diffuser.ogb_task.ogb_maze_v1.multi_agent_planner_utils import (
    make_multi_agent_layout,
    validate_env_matches_layout,
    get_agent_single_obs_from_multi_obs,
    compose_multi_agent_action,
    make_single_active_joint_action,
    get_multi_agent_world_state,
    set_multi_agent_env_from_single_agent_obs,
    overlay_subtitles_on_image,
)
from diffuser.ogb_task.ogb_maze_v1.multi_agent_coordinator import (
    MultiAgentRelayCoordinator,
    MultiAgentRelayConfig,
    ParallelRelayCoordinator,
    ClosestAntRelayCoordinator,
)


class OgB_Stgl_Sml_MultiAgents_MazeEnvPlanner_V1:
    """
    A Class to support evaluation using CompDiffuser.
    This class can be used for OGBench Maze and AntSoccer.

    Multi-agent Mode B behavior (N agents):
        1. Ant1 is active.
        2. Ant1 pushes the ball to handoff_xy = (x1, y1).
        3. Ant1 retreats; Ant2 wakes up when Ant1 is parked.
        4. Ant2 pushes the ball to (x2, y2), retreats; Ant3 wakes up.
        5. ... continues through all relay segments ...
        6. The last ant pushes the ball to final_goal_xy.
        7. Done when the BALL is close enough to final_goal_xy.
    """

    def __init__(self, args_train, args) -> None:
        self.args_train = args_train
        self.args = args

        # Multi-agent rollout settings.
        self.num_agents = int(getattr(args, "num_agents", 2))
        self.initial_active_agent_id = int(getattr(args, "initial_active_agent_id", 1))
        self.multi_agent_env_name = getattr(
            args,
            "multi_agent_env_name",
            "antsoccer-twoants-arena-v0",
        )

        self.ma_layout = make_multi_agent_layout(self.num_agents)

        self.ma_mode = getattr(args, "ma_mode", "mode_b_sequential_handoff")
        self.carrier_order = getattr(
            args,
            "carrier_order",
            list(range(1, self.num_agents + 1)),
        )
        self.carrier_order = [int(x) for x in self.carrier_order]

        self.handoff_x = float(getattr(args, "handoff_x", 5.0))
        self.handoff_y = float(getattr(args, "handoff_y", 2.0))
        self.handoff_points = self._parse_handoff_points(args)

        # Final Mode B thresholds.
        # Important: this is the strict BALL-to-handoff radius.
        # Do NOT use 3.5/4.0 here for final behavior; that caused early switching.
        self.handoff_ball_radius = float(getattr(args, "handoff_ball_radius", 1.0))

        # Ant2 first has to reach the ball/handoff before pushing to final.
        self.ant2_reach_ball_radius = float(getattr(args, "ant2_reach_ball_radius", 1.5))

        # Final success threshold: ball-to-final-goal.
        self.ball_goal_threshold = float(getattr(args, "ball_goal_threshold", 1.0))

        # Backward compatibility for old logs/code paths.
        self.ball_handoff_threshold = self.handoff_ball_radius

        self.retreat_points = self._parse_retreat_points(args)
        self.retreat_threshold = float(
            getattr(args, "retreat_threshold", getattr(args, "ant1_retreat_threshold", 2.0))
        )

        self.relay_coordinator = None
        self.ma_stage_switched_last_update = False
        self.ma_stage_target_xy = self.handoff_points[0].copy()

        # Concurrent relay state; unused by the sequential path.
        self.parallel_relay_coordinator = None
        self.closest_ant_relay_coordinator = None
        self.ma_parallel_stage_switched_last_update = False

        utils.print_color(
            f"[MultiAgent Planner] num_agents={self.num_agents}, "
            f"initial_active_agent_id={self.initial_active_agent_id}, "
            f"multi_agent_env_name={self.multi_agent_env_name}, "
            f"handoff_points={self.handoff_points}",
            c="c",
        )

        self.plan_n_ep = args.plan_n_ep
        self.b_size_per_prob = args.b_size_per_prob

        self.n_batch_acc_probs = args.n_batch_acc_probs
        self.is_replan = args.is_replan
        self.repl_wp_cfg = args.repl_wp_cfg
        self.repl_ada_dist_cfg = getattr(self.args, "repl_ada_dist_cfg", {})

        self.is_inv_train_mode = getattr(self.args, "is_inv_train_mode", False)
        self.ep_st_idx = getattr(self.args, "ep_st_idx", 0)

        self.is_save_pkl = getattr(self.args, "is_save_pkl", False)
        self.rd_resol = getattr(self.args, "rd_resol", 200)
        self.is_use_subgoal_marker = getattr(self.args, "is_use_subgoal_marker", True)
        self.is_rd_agv = getattr(self.args, "is_rd_agv", False)
        self.is_vid_subtitles = bool(int(getattr(args, "is_vid_subtitles", 1)))

        if self.plan_n_ep == 100:
            assert self.ep_st_idx == 0

        self.vis_trajs_per_img = 10
        self.score_low_limit = 100
        self.act_control = "pred_inv"
        self.n_act_per_waypnt = args.n_act_per_waypnt

        tmp_fps = getattr(self.args, "vid_fps", False)
        if tmp_fps:
            pass
        elif "soccer" in self.args.dataset:
            tmp_fps = 80
        else:
            tmp_fps = 120

        self.vid_fps = tmp_fps * self.n_act_per_waypnt
        self.is_soccer_task = False

        if "antmaze" in args_train.dataset:
            self.extra_env_steps = 150
        elif "humanoidmaze" in args_train.dataset:
            self.extra_env_steps = 400
        elif "antsoccer" in args_train.dataset:
            self.extra_env_steps = 300
            self.is_soccer_task = True
        elif "pointmaze" in args_train.dataset:
            self.extra_env_steps = 50
            self.act_control = "pd_ogb"
        else:
            raise NotImplementedError

    @staticmethod
    def _parse_handoff_points(args):
        raw_points = getattr(args, "handoff_points", None)
        if raw_points is not None:
            return [np.asarray(p, dtype=np.float32).reshape(2) for p in raw_points]

        expected_handoffs = max(int(getattr(args, "num_agents", 2)) - 1, 0)
        if expected_handoffs <= 1:
            handoff_x = float(getattr(args, "handoff_x", 5.0))
            handoff_y = float(getattr(args, "handoff_y", 2.0))
            return [np.array([handoff_x, handoff_y], dtype=np.float32)]

        prefix = []
        for idx in range(1, expected_handoffs + 1):
            x_key = f"handoff{idx}_x"
            y_key = f"handoff{idx}_y"
            if hasattr(args, x_key) and hasattr(args, y_key):
                prefix.append(
                    np.array(
                        [float(getattr(args, x_key)), float(getattr(args, y_key))],
                        dtype=np.float32,
                    )
                )

        if len(prefix) == expected_handoffs:
            return prefix

        raise ValueError(
            f"Expected {expected_handoffs} handoff points for "
            f"num_agents={getattr(args, 'num_agents', 2)}. "
            "Provide args.handoff_points or handoff1_x/handoff1_y, ..."
        )

    @staticmethod
    def _parse_retreat_points(args):
        raw_points = getattr(args, "retreat_points", None)
        if raw_points is not None:
            return [np.asarray(p, dtype=np.float32).reshape(2) for p in raw_points]

        num_agents = int(getattr(args, "num_agents", 2))
        expected_retreats = max(num_agents - 1, 0)
        if expected_retreats == 0:
            return []

        default_positions = [
            [2.0, 14.0],
            [2.0, 2.0],
            [14.0, 2.0],
            [20.0, 14.0],
        ]

        retreat_x = float(getattr(args, "retreat_x", default_positions[0][0]))
        retreat_y = float(getattr(args, "retreat_y", default_positions[0][1]))
        default_positions[0] = [retreat_x, retreat_y]

        return [
            np.array(default_positions[idx], dtype=np.float32)
            for idx in range(expected_retreats)
        ]

    def setup_load(self, ld_config):
        """
        Used in a separate launch evaluation, where model should be loaded from file.
        """
        args_train = self.args_train
        args = self.args

        if self.args.env_n_max_steps is not None:
            assert self.is_replan == "ada_dist"

        dfu_exp = utils.load_stgl_sml_diffusion(
            args.logbase,
            args_train.dataset,
            args_train.exp_name,
            epoch=args.diffusion_epoch,
            ld_config=ld_config,
        )

        self.train_normalizer = utils.load_ogb_maze_datasetNormalizer(args_train)
        self.full_normalizer = utils.load_ogb_maze_datasetNormalizer(
            args_train,
            obs_dim_idxs="full",
        )

        self.check_is_same_nmlizer()

        self.diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1 = dfu_exp.ema
        self.diffusion.var_temp = args.var_temp
        self.diffusion.condition_guidance_w = args.cond_w
        self.diffusion.use_ddim = args.use_ddim
        self.diffusion.ddim_eta = args.ddim_eta
        self.diffusion.ddim_num_inference_steps = args.ddim_steps

        self.dataset = dfu_exp.dataset
        self.renderer: Maze2dRenderer_V2
        self.renderer = dfu_exp.renderer
        self.trainer = dfu_exp.trainer

        if self.diffusion.is_inv_dyn_dfu:
            dfu_name = getattr(self.diffusion, "dfu_name", "our_stgl_sml")

            if dfu_name == "dd_maze":
                self.pol_config = {}
                self.tj_blder_config = {}
                for k in args.__dict__.keys():
                    if "ev_" in k:
                        self.pol_config[k] = args.__dict__[k]

                from diffuser.baselines.dd_maze.dd_maze_policy_v1 import DD_Maze_Policy_V1

                self.policy = DD_Maze_Policy_V1(
                    self.diffusion,
                    self.train_normalizer,
                    self.pol_config,
                )

            else:
                self.pol_config = {}

                for k in args.__dict__.keys():
                    if "ev_" in k:
                        self.pol_config[k] = args.__dict__[k]

                self.tj_blder_config = dict(
                    blend_type=args.tjb_blend_type,
                    exp_beta=args.tjb_exp_beta,
                )

                self.policy = Stgl_Sml_Policy_V1(
                    self.diffusion,
                    self.train_normalizer,
                    self.pol_config,
                    self.tj_blder_config,
                )

        else:
            assert False, "not implemented"

        self.savepath = args.savepath
        self.epoch = dfu_exp.epoch

        utils.print_color(f"Load From {self.epoch=}", c="y")

        self.create_env()

        self.setup_general()
        self.dataset_config = args_train.dataset_config
        self.obs_select_dim = self.dataset_config["obs_select_dim"]
        self.dfu_ndim = len(self.obs_select_dim)
        self.dset_type = self.dataset_config["dset_type"]
        self.ld_config = ld_config
        self.load_inv_model()

    def create_env(self):
        if hasattr(self, "env"):
            del self.env
        if hasattr(self, "base_env"):
            del self.base_env

        if Is_OgB_Robot_Env:
            utils.print_color(
                f"[MultiAgent Planner] loading real rollout env with gym.make: "
                f"{self.multi_agent_env_name}",
                c="c",
            )

            self.env = gym.make(
                self.multi_agent_env_name,
                render_mode="rgb_array",
                width=self.rd_resol,
                height=self.rd_resol,
            )

            self.base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

            if hasattr(self.base_env, "set_seed_addn"):
                self.base_env.set_seed_addn(0)

            # Keep original single-agent dataset name for normalizer/invdyn/eval problems.
            self.base_env.name = self.args.dataset
            self.env.name = self.args.dataset

            validate_env_matches_layout(self.base_env, self.ma_layout)

            utils.print_color(
                f"[MultiAgent Planner] real env loaded. "
                f"num_agents={self.num_agents}, "
                f"model.nq={self.base_env.model.nq}, "
                f"model.nv={self.base_env.model.nv}, "
                f"model.nu={self.base_env.model.nu}, "
                f"expected_nq={self.ma_layout.qpos_dim}, "
                f"expected_nv={self.ma_layout.qvel_dim}, "
                f"expected_nu={self.ma_layout.action_dim}, "
                f"qpos.shape={self.base_env.data.qpos.shape}, "
                f"qvel.shape={self.base_env.data.qvel.shape}, "
                f"env.name kept as {self.base_env.name}",
                c="c",
            )

        else:
            raise NotImplementedError

        if not hasattr(self.base_env, "max_episode_steps"):
            self.base_env.max_episode_steps = getattr(
                self.env.spec,
                "max_episode_steps",
                1000,
            )

        utils.print_color(
            f"[setup_load] self.base_env.max_episode_steps={self.base_env.max_episode_steps}"
        )

    def setup_general(self):
        utils.mkdir(self.savepath)
        self.savepath_root = self.savepath
        self.load_ev_problems()

    def load_inv_model(self):
        """
        Load the inverse dynamics model.
        """
        if hasattr(self.args, "inv_model_path"):
            self.inv_model_path = self.args.inv_model_path
        else:
            self.inv_model_path = utils.ogb_get_inv_model_path(
                self.base_env.name,
                gl_dim=self.dfu_ndim,
            )

        if "pointmaze" in self.args.dataset:
            import types

            self.inv_epoch = None
            self.inv_model = types.SimpleNamespace()
            self.inv_model.training = None
            utils.print_color(
                f"\n{self.args.dataset=}. *** No Inv Dyn Model *** \n",
                c="y",
            )

        else:
            from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import (
                ogb_load_invdyn_maze_v1,
            )

            inv_model, inv_ema, inv_epoch = ogb_load_invdyn_maze_v1(
                self.inv_model_path,
                epoch=self.args.inv_epoch,
            )
            self.inv_epoch = inv_epoch
            self.inv_model = inv_model

            if self.is_inv_train_mode:
                self.inv_model.train()
                assert "soccer" in self.args.dataset, "temporary, can be removed"

            utils.print_color(f"\n{self.inv_model.training=}\n", c="y")

    def load_ev_problems(self):
        """
        Load pre-collected evaluation problems.
        """
        self.problems_h5path = utils.get_ogb_maze_ev_probs_fname(self.base_env.name)
        self.problems_dict = utils.load_stgl_lh_ev_probs_hdf5(
            h5path=self.problems_h5path
        )
        return

    def ogb_env_get_obs(self) -> np.ndarray:
        active_agent_id = self.ma_get_current_active_agent_id(update_relay=False)
        if active_agent_id is None:
            active_agent_id = self.carrier_order[-1]

        raw_obs = self.base_env.get_ob()
        return get_agent_single_obs_from_multi_obs(
            raw_obs,
            layout=self.ma_layout,
            agent_id=active_agent_id,
        )

    def ma_setup_relay_coordinator_for_episode(self, final_goal_xy):
        config = MultiAgentRelayConfig(
            carrier_order=self.carrier_order,
            handoff_points=self.handoff_points,
            final_goal_xy=np.asarray(final_goal_xy, dtype=np.float32),
            retreat_points=self.retreat_points,
            retreat_threshold=self.retreat_threshold,
            retreat_xy=np.array(
                [
                    float(getattr(self.args, "retreat_x", self.retreat_points[0][0])),
                    float(getattr(self.args, "retreat_y", self.retreat_points[0][1])),
                ],
                dtype=np.float32,
            ) if self.retreat_points else None,
            ant1_retreat_threshold=float(
                getattr(self.args, "ant1_retreat_threshold", self.retreat_threshold)
            ),
            ball_handoff_threshold=float(getattr(self.args, "ball_handoff_threshold", 3.0)),
            ball_goal_threshold=self.ball_goal_threshold,
        )

        self.relay_coordinator = MultiAgentRelayCoordinator(
            config=config,
            mode=self.ma_mode,
        )

        self.ma_stage_switched_last_update = False
        self.ma_stage_target_xy = self.handoff_points[0].copy()

        utils.print_color(
            f"[MA Relay] setup: "
            f"mode={self.ma_mode}, "
            f"carrier_order={self.carrier_order}, "
            f"handoff_points={self.handoff_points}, "
            f"final_goal_xy={np.asarray(final_goal_xy, dtype=np.float32)}, "
            f"handoff_ball_radius={self.handoff_ball_radius}, "
            f"ant2_reach_ball_radius={self.ant2_reach_ball_radius}, "
            f"ball_goal_threshold={self.ball_goal_threshold}",
            c="g",
        )

    def ma_uses_parallel_step(self) -> bool:
        return self.ma_mode in (
            "mode_c_parallel_handoff",
            "mode_d_closest_ant_parallel",
        )

    def ma_setup_closest_ant_relay_coordinator_for_episode(self, final_goal_xy):
        config = MultiAgentRelayConfig(
            carrier_order=self.carrier_order,
            handoff_points=self.handoff_points,
            final_goal_xy=np.asarray(final_goal_xy, dtype=np.float32),
            ball_handoff_threshold=float(
                getattr(self.args, "ball_handoff_threshold", 3.0)
            ),
            ball_goal_threshold=self.ball_goal_threshold,
        )

        self.closest_ant_relay_coordinator = ClosestAntRelayCoordinator(
            config=config,
            mode=self.ma_mode,
        )
        self.ma_parallel_stage_switched_last_update = False

        utils.print_color(
            f"[MA Closest-Ant Relay] setup: "
            f"mode={self.ma_mode}, "
            f"num_agents={self.num_agents}, "
            f"handoff_points={self.handoff_points}, "
            f"final_goal_xy={np.asarray(final_goal_xy, dtype=np.float32)}, "
            f"ball_handoff_threshold={float(getattr(self.args, 'ball_handoff_threshold', 3.0))}, "
            f"ball_goal_threshold={self.ball_goal_threshold}",
            c="g",
        )

    def ma_setup_parallel_relay_coordinator_for_episode(self, final_goal_xy):
        """
        Concurrent-mode counterpart of ma_setup_relay_coordinator_for_episode.
        Only used when self.ma_mode == "mode_c_parallel_handoff".
        """
        config = MultiAgentRelayConfig(
            carrier_order=self.carrier_order,
            handoff_points=self.handoff_points,
            final_goal_xy=np.asarray(final_goal_xy, dtype=np.float32),
            # retreat_xy/ant1_retreat_threshold are required by the shared
            # config dataclass but unused in parallel mode (no retreat step).
            retreat_points=self.retreat_points,
            retreat_threshold=self.retreat_threshold,
            retreat_xy=np.array(
                [
                    float(getattr(self.args, "retreat_x", self.retreat_points[0][0] if self.retreat_points else 2.0)),
                    float(getattr(self.args, "retreat_y", self.retreat_points[0][1] if self.retreat_points else 14.0)),
                ],
                dtype=np.float32,
            ),
            ant1_retreat_threshold=float(
                getattr(self.args, "ant1_retreat_threshold", self.retreat_threshold)
            ),
            ball_handoff_threshold=float(getattr(self.args, "ball_handoff_threshold", 3.0)),
            ball_goal_threshold=self.ball_goal_threshold,
        )

        self.parallel_relay_coordinator = ParallelRelayCoordinator(
            config=config,
            mode=self.ma_mode,
        )
        self.ma_parallel_stage_switched_last_update = False

        utils.print_color(
            f"[MA Parallel Relay] setup: "
            f"mode={self.ma_mode}, "
            f"carrier_order={self.carrier_order}, "
            f"handoff_points={self.handoff_points}, "
            f"final_goal_xy={np.asarray(final_goal_xy, dtype=np.float32)}, "
            f"ball_handoff_threshold={float(getattr(self.args, 'ball_handoff_threshold', 3.0))}, "
            f"ball_goal_threshold={self.ball_goal_threshold}",
            c="g",
        )

    def ma_get_world_state(self) -> dict:
        raw_obs = self.base_env.get_ob().copy()
        return get_multi_agent_world_state(raw_obs, layout=self.ma_layout)

    def ma_update_relay_and_get_tasks(self, update_relay: bool = True) -> dict:
        """
        Return current per-agent tasks.

        Important:
            update_relay=True should be called only once per env timestep.
            After that, reuse the returned active_agent_id for obs/action in the same timestep.
        """
        if self.relay_coordinator is None:
            return {
                int(self.carrier_order[0]): {
                    "active": True,
                    "role": "initial",
                    "stage": "initial",
                    "active_agent_id": int(self.carrier_order[0]),
                    "target_xy": None,
                    "is_done": False,
                }
            }

        world_state = self.ma_get_world_state()

        if update_relay:
            self.relay_coordinator.update(world_state)
            self.ma_stage_switched_last_update = self.relay_coordinator.last_switch_info is not None

        tasks = self.relay_coordinator.get_tasks(world_state)
        return tasks

    def ma_get_current_active_agent_id(self, update_relay: bool = False):
        active_agent_id, _ = self.ma_get_active_task(update_relay=update_relay)
        return active_agent_id

    def ma_get_active_task(self, update_relay: bool = False) -> tuple:
        """
        Return (active_agent_id, active_task).

        Mode B expects exactly one active agent unless the relay is done.
        """
        if self.relay_coordinator is None:
            active_agent_id = int(self.carrier_order[0])
            return active_agent_id, {
                "active": True,
                "role": "initial",
                "stage": "initial",
                "active_agent_id": active_agent_id,
                "target_xy": None,
                "is_done": False,
            }

        tasks = self.ma_update_relay_and_get_tasks(update_relay=update_relay)

        active_items = [
            (agent_id, task)
            for agent_id, task in tasks.items()
            if task.get("active", False)
        ]

        if len(active_items) != 1:
            if self.relay_coordinator.is_done():
                return None, {
                    "active": False,
                    "role": "done",
                    "stage": "done",
                    "active_agent_id": None,
                    "target_xy": None,
                    "is_done": True,
                }

            raise RuntimeError(
                f"Mode B expects exactly one active task, got {active_items}. "
                f"tasks={tasks}"
            )

        active_agent_id, active_task = active_items[0]
        active_task = dict(active_task)
        active_task.setdefault("active", True)
        active_task.setdefault("active_agent_id", active_agent_id)
        active_task.setdefault("is_done", False)
        active_task.setdefault(
            "stage",
            getattr(
                self.relay_coordinator,
                "current_stage_name",
                active_task.get("role", "unknown"),
            ),
        )

        return active_agent_id, active_task

    def ma_get_parallel_tasks(self, update_relay: bool = True) -> dict:
        """
        Return per-agent tasks for parallel relay modes (mode_c / mode_d).

        Unlike ma_get_active_task, multiple agents may be active at once.
        """
        world_state = self.ma_get_world_state()

        if self.ma_mode == "mode_c_parallel_handoff":
            coordinator = self.parallel_relay_coordinator
        elif self.ma_mode == "mode_d_closest_ant_parallel":
            coordinator = self.closest_ant_relay_coordinator
        else:
            raise RuntimeError(f"ma_get_parallel_tasks called with {self.ma_mode=}")

        if update_relay:
            coordinator.update(world_state)
            self.ma_parallel_stage_switched_last_update = (
                coordinator.last_switch_info is not None
            )

        return coordinator.get_tasks(world_state)

    def ma_format_target_label(self, target_xy) -> str:
        if target_xy is None:
            return "?"
        target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
        return f"({target_xy[0]:.1f}, {target_xy[1]:.1f})"

    def ma_get_ball_target_label(self) -> str:
        """Human-readable description of where the ball should go next."""
        if (
            self.ma_mode == "mode_d_closest_ant_parallel"
            and self.closest_ant_relay_coordinator is not None
        ):
            coord = self.closest_ant_relay_coordinator
            if coord.done:
                return "done"
            if coord.handoff_idx < len(self.handoff_points):
                return f"handoff {coord.handoff_idx + 1}"
            return "final goal"

        if self.relay_coordinator is not None:
            coord = self.relay_coordinator
            if coord.done:
                return "done"
            if coord.phase == "carry":
                if coord.segment_idx < len(self.handoff_points):
                    return f"handoff {coord.segment_idx + 1}"
                return "final goal"
            if coord.segment_idx < len(self.handoff_points):
                return f"handoff {coord.segment_idx + 1} (wait)"
            return "final goal (wait)"

        if len(self.handoff_points) == 1:
            return "handoff"
        return f"handoff {len(self.handoff_points)}"

    def ma_format_agent_action_label(
        self,
        agent_id: int,
        role: str = "",
        stage: str = "",
    ) -> str:
        stage = stage or ""
        role = role or ""

        if "retreat" in stage or role == "agent_retreating":
            return f"Ant{agent_id}: parking"
        if role == "move_to_handoff":
            return f"Ant{agent_id}: going to handoff"
        if role in ("carry_ball", "carry_ball_to_handoff", "carry_ball_to_final"):
            if "to_final" in stage or role == "carry_ball_to_final":
                return f"Ant{agent_id}: carrying ball to goal"
            if "to_handoff" in stage:
                handoff_num = stage.rsplit("_", 1)[-1]
                if handoff_num.isdigit():
                    return f"Ant{agent_id}: carrying ball to handoff {handoff_num}"
            return f"Ant{agent_id}: carrying ball"
        if "to_final" in stage:
            return f"Ant{agent_id}: carrying ball to goal"
        if "to_handoff" in stage:
            handoff_num = stage.rsplit("_", 1)[-1]
            if handoff_num.isdigit():
                return f"Ant{agent_id}: carrying ball to handoff {handoff_num}"
            return f"Ant{agent_id}: carrying ball"
        if "handoff" in stage:
            return f"Ant{agent_id}: carrying ball"

        return f"Ant{agent_id}: active"

    def ma_get_relay_subtitle_lines(
        self,
        tasks=None,
        active_agent_id=None,
        active_task=None,
    ):
        mode_label = {
            "mode_b_sequential_handoff": "Sequential relay",
            "mode_d_closest_ant_parallel": "Closest-ant relay",
            "mode_c_parallel_handoff": "Parallel relay",
        }.get(self.ma_mode, self.ma_mode)

        lines = [f"{mode_label} | Ball -> {self.ma_get_ball_target_label()}"]

        if self.ma_uses_parallel_step() and tasks is not None:
            active_parts = []
            for agent_id in self.carrier_order:
                task = tasks.get(agent_id, {})
                if not task.get("active", False) or task.get("frozen", False):
                    continue
                active_parts.append(
                    self.ma_format_agent_action_label(
                        agent_id,
                        role=task.get("role", ""),
                        stage=task.get("stage_name", ""),
                    )
                )

            if len(active_parts) == 1:
                lines.append(active_parts[0])
            elif active_parts:
                lines.append(" | ".join(active_parts))
            else:
                lines.append("Waiting")
            return lines

        if active_agent_id is None:
            lines.append("Done" if active_task and active_task.get("is_done") else "Waiting")
            return lines

        stage = active_task.get("stage") or active_task.get("stage_name", "")
        role = active_task.get("role", "")
        lines.append(
            self.ma_format_agent_action_label(
                active_agent_id,
                role=role,
                stage=stage,
            )
        )
        return lines

    def ma_render_frame_with_subtitles(
        self,
        tasks=None,
        active_agent_id=None,
        active_task=None,
    ):
        img = self.base_env.render()
        if not self.is_vid_subtitles:
            return img

        lines = self.ma_get_relay_subtitle_lines(
            tasks=tasks,
            active_agent_id=active_agent_id,
            active_task=active_task,
        )
        return overlay_subtitles_on_image(img, lines, font_size=11)

    def ma_build_stage_goal_obs(self, base_goal_obs: np.ndarray, active_task: dict) -> np.ndarray:
        """
        Builds a goal observation where BOTH the ant position and ball position 
        are overridden to match the current relay stage.
        """
        goal_obs = np.asarray(base_goal_obs, dtype=np.float32).copy()

        if active_task is None:
            return goal_obs

        target_xy = active_task.get("target_xy", None)
        stage_name = active_task.get("stage_name", "")

        if target_xy is not None:
            target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
            
            if stage_name.endswith("_retreat") or stage_name == "ant1_retreat":
                goal_obs[0:2] = target_xy
                goal_obs[15:17] = goal_obs[15:17]
            elif (
                active_task.get("role") == "move_to_handoff"
                or "_meet_" in stage_name
            ):
                goal_obs[0:2] = target_xy
            else:
                goal_obs[0:2] = target_xy
                goal_obs[15:17] = target_xy

        return goal_obs

    def ma_run_parallel_step(
        self,
        i_ep,
        i_et,
        gl_pos,
        n_comp_full,
        fused_traj_ep_by_agent,
        all_plan_trajs_ep_by_agent,
        cnt_repl_by_agent,
        prev_dfu_wp_idx_by_agent,
        prev_n_comp_by_agent,
        goal_cur_by_agent,
        last_pick_traj_by_agent,
        is_suc,
        total_reward,
        cnt_extras,
        rollout,
        imgs_rout,
        imgs_rout_agv,
    ):
        """
        One env timestep of the concurrent (mode_c_parallel_handoff) relay:
        both agents are independently planned/replanned/followed (each
        keeping its own fused_traj_ep/wp-tracking state), then a single
        joint action drives one self.base_env.step() call.

        Mirrors the per-agent body of the sequential loop
        (ogb_plan_once, lines ~787-1063) but looped over both agents;
        does not call/modify ma_get_active_task or the sequential
        relay_coordinator.
        """
        tasks = self.ma_get_parallel_tasks(update_relay=True)
        force_stage_replan = bool(
            getattr(self, "ma_parallel_stage_switched_last_update", False)
        )

        wp_idx = i_et // self.n_act_per_waypnt
        is_wp_not_start = (wp_idx * self.n_act_per_waypnt) == i_et

        if i_et % 50 == 0:
            world_state_dbg = self.ma_get_world_state()
            utils.print_color(
                f"[MA PARALLEL STATE DEBUG] {i_ep=} {i_et=} "
                f"agent_xy={world_state_dbg['agent_xy']} "
                f"ball_xy={world_state_dbg['ball_xy']} "
                f"stages={ {aid: (t['stage_name'], t.get('frozen')) for aid, t in tasks.items()} }",
                c="m",
            )

        actions_by_agent = {}

        for agent_id in self.carrier_order:
            task = tasks[agent_id]

            if not task.get("active", False) or task.get("frozen", True):
                actions_by_agent[agent_id] = None
                continue

            raw_obs_cur = self.base_env.get_ob().copy()
            obs_cur = get_agent_single_obs_from_multi_obs(
                raw_obs_cur,
                layout=self.ma_layout,
                agent_id=agent_id,
            )
            obs_cur_nm = self.full_normalizer.normalize(obs_cur, "observations")

            fused_traj_ep = fused_traj_ep_by_agent[agent_id]
            all_plan_trajs_ep = all_plan_trajs_ep_by_agent[agent_id]
            cnt_repl = cnt_repl_by_agent[agent_id]

            force_replan = force_stage_replan
            if self.ma_mode == "mode_c_parallel_handoff":
                force_replan = force_stage_replan and agent_id == self.carrier_order[1]

            if i_et > 0 and agent_id in goal_cur_by_agent:
                ada_used_dist = np.linalg.norm(
                    obs_cur[self.obs_select_dim,][self.ada_dist_used_idxs,]
                    - goal_cur_by_agent[agent_id][self.ada_dist_used_idxs,]
                )
            else:
                ada_used_dist = None

            if self.is_replan == "at_given_t":
                is_do_repl_et = wp_idx in self.repl_wp_cfg
            elif self.is_replan == "ada_dist" and ada_used_dist is not None:
                repl_ada_cond_1 = ada_used_dist > self.ada_dist_thres
                prev_dfu_wp_idx = prev_dfu_wp_idx_by_agent.get(agent_id, 0)
                prev_traj_len = len(all_plan_trajs_ep[-1]) if all_plan_trajs_ep else 0
                repl_ada_cond_2 = (
                    wp_idx - prev_dfu_wp_idx - self.ada_dist_cond_2_extra
                ) > prev_traj_len
                is_do_repl_et = (
                    repl_ada_cond_1 or repl_ada_cond_2
                ) and cnt_repl < self.ada_dist_max_n_repl
            else:
                is_do_repl_et = False

            is_do_repl_et = is_do_repl_et and is_wp_not_start

            if force_replan:
                is_do_repl_et = True

            if i_et == 0 or is_do_repl_et:
                input_st = obs_cur[self.obs_select_dim,]

                stage_goal_obs = self.ma_build_stage_goal_obs(
                    base_goal_obs=gl_pos,
                    active_task=task,
                )
                gl_pos_for_dfu = stage_goal_obs[self.obs_select_dim,]

                g_cond = {
                    "st_gl": np.array(
                        [input_st[None,], gl_pos_for_dfu[None,]],
                        dtype=np.float32,
                    )
                }

                if is_do_repl_et and i_et > 0:
                    cnt_repl += 1

                    if self.is_replan == "at_given_t":
                        self.policy.n_comp = self.repl_wp_cfg[wp_idx]

                    elif self.is_replan == "ada_dist":
                        if self.ada_dist_type == "m_1":
                            raise NotImplementedError

                        elif self.ada_dist_type == "m_2":
                            if (
                                agent_id not in prev_dfu_wp_idx_by_agent
                                or not all_plan_trajs_ep
                            ):
                                tmp_n_comp = n_comp_full
                            else:
                                prev_dfu_wp_idx = prev_dfu_wp_idx_by_agent[agent_id]
                                prev_n_comp = prev_n_comp_by_agent.get(
                                    agent_id, n_comp_full
                                )
                                tmp_cnt_wp = wp_idx - prev_dfu_wp_idx
                                tmp_v1 = max(0, (tmp_cnt_wp - self.ada_dist_minus_n_wp))
                                prev_hzn = len(all_plan_trajs_ep[-1])
                                tmp_n_comp = math.ceil(
                                    (1 - tmp_v1 / prev_hzn) * prev_n_comp
                                )
                                tmp_n_comp = max(1, tmp_n_comp)

                        self.policy.n_comp = tmp_n_comp
                    else:
                        raise NotImplementedError
                else:
                    self.policy.n_comp = n_comp_full

                m_out = self.policy.gen_cond_stgl(
                    g_cond=g_cond,
                    b_s=self.b_size_per_prob,
                )
                pick_traj = m_out.pick_traj

                utils.print_color(f"[ Run Parallel Planner ] {i_et=} {wp_idx=} {agent_id=}", c="y")

                tmp_tj_end_idx = wp_idx + len(pick_traj)
                if tmp_tj_end_idx > len(fused_traj_ep):
                    pick_traj = pick_traj[: len(fused_traj_ep) - wp_idx]
                    tmp_tj_end_idx = wp_idx + len(pick_traj)

                fused_traj_ep[wp_idx:tmp_tj_end_idx] = pick_traj
                fused_traj_ep[tmp_tj_end_idx:] = pick_traj[-1]

                all_plan_trajs_ep.append(pick_traj)
                last_pick_traj_by_agent[agent_id] = pick_traj

                prev_dfu_wp_idx_by_agent[agent_id] = wp_idx
                prev_n_comp_by_agent[agent_id] = self.policy.n_comp
                cnt_repl_by_agent[agent_id] = cnt_repl

            wp_idx_clamped = min(wp_idx, len(fused_traj_ep) - 1)
            goal_cur = fused_traj_ep[wp_idx_clamped]
            goal_cur_by_agent[agent_id] = goal_cur.copy()

            if self.is_use_subgoal_marker:
                self.base_env.set_subgoal_waypnt(goal_cur[:2])

            goal_cur_nm = self.train_normalizer.normalize(goal_cur, "observations")

            obs_cur_nm_t = utils.to_torch(obs_cur_nm)[None,]
            goal_cur_nm_t = utils.to_torch(goal_cur_nm)[None,]
            act_pred_nm = self.inv_model(obs_cur_nm_t, goal_cur_nm_t).cpu().numpy()[0]
            act_pred = self.full_normalizer.unnormalize(act_pred_nm, "actions")

            actions_by_agent[agent_id] = act_pred

        joint_action = compose_multi_agent_action(actions_by_agent, self.ma_layout)

        if i_et % 50 == 0:
            utils.print_color(
                f"[MA PARALLEL ACTION DEBUG] {i_ep=} {i_et=} "
                f"norm_ant1={np.linalg.norm(joint_action[:8]):.3f} "
                f"norm_ant2={np.linalg.norm(joint_action[8:16]):.3f}",
                c="m",
            )

        raw_obs_cur, rew, terminated, truncated, info = self.base_env.step(joint_action)

        is_suc = bool(info["success"]) or is_suc
        total_reward += rew

        rollout.append(self.base_env.get_qpos_qvel())
        subtitle_tasks = self.ma_get_parallel_tasks(update_relay=False)
        imgs_rout.append(
            self.ma_render_frame_with_subtitles(tasks=subtitle_tasks)
        )

        if self.is_rd_agv:
            imgs_rout_agv.append(self.render_agv_img())

        do_break = False
        if self.ma_mode == "mode_d_closest_ant_parallel":
            if self.closest_ant_relay_coordinator.is_done():
                utils.print_color(
                    f"[MA Closest Relay] done at {i_ep=} {i_et=}. Ending episode.",
                    c="g",
                )
                is_suc = True
                total_reward += 1.0
                do_break = True

        if is_suc:
            utils.print_color(f"{i_ep=} {i_et=} {is_suc=}")
            cnt_extras += 1
            if cnt_extras == 30:
                do_break = True

        return is_suc, total_reward, cnt_extras, do_break

    def ogb_env_interact_1_ep(self, pick_traj, start_state, target):
        """
        OGBench interaction for one episode.
        Kept mostly for compatibility. Main online planning path is ogb_plan_once().
        """
        active_agent_id = self.ma_get_current_active_agent_id(update_relay=False)
        cur_obs42 = get_agent_single_obs_from_multi_obs(
            self.base_env.get_ob(),
            layout=self.ma_layout,
            agent_id=active_agent_id,
        )
        assert np.isclose(start_state, cur_obs42).all()
        assert np.isclose(target, self.base_env.cur_goal_xy).all()

        is_suc = False
        total_reward = 0
        rollout = [start_state.copy()]
        imgs_rout = [self.base_env.render()]

        self.n_max_steps = len(pick_traj) * self.n_act_per_waypnt + 30

        for i_et in range(self.n_max_steps):
            active_agent_id, active_task = self.ma_get_active_task(update_relay=True)

            if active_agent_id is None:
                utils.print_color(
                    f"[MA Relay] done at i_et={i_et}. Ending episode.",
                    c="g",
                )
                is_suc = True
                total_reward += 1.0
                break

            raw_obs_cur = self.base_env.get_ob().copy()
            obs_cur = get_agent_single_obs_from_multi_obs(
                raw_obs_cur,
                layout=self.ma_layout,
                agent_id=active_agent_id,
            )
            obs_cur_nm = self.full_normalizer.normalize(obs_cur, "observations")

            assert len(obs_cur) in [4, 29, 42, 69], "2d, ant, antsoccer, humanoid"

            if self.act_control == "pred_inv":
                wp_idx = i_et // self.n_act_per_waypnt
                wp_idx = min(wp_idx, len(pick_traj) - 1)

                goal_cur = pick_traj[wp_idx]
                goal_cur_nm = self.train_normalizer.normalize(goal_cur, "observations")

                self.base_env.set_subgoal_waypnt(goal_cur[:2])

                obs_cur_nm = utils.to_torch(obs_cur_nm)[None,]
                goal_cur_nm = utils.to_torch(goal_cur_nm)[None,]
                act_pred = self.inv_model(obs_cur_nm, goal_cur_nm).cpu().numpy()[0]

            elif self.act_control == "dfu_force":
                raise NotImplementedError()

            else:
                raise NotImplementedError(f"Unknown act_control={self.act_control}")

            joint_action = make_single_active_joint_action(
                act_pred,
                active_agent_id=active_agent_id,
                layout=self.ma_layout,
            )

            raw_obs_cur, rew, terminated, truncated, info = self.base_env.step(joint_action)

            obs_cur = get_agent_single_obs_from_multi_obs(
                raw_obs_cur,
                layout=self.ma_layout,
                agent_id=active_agent_id,
            )

            is_suc = bool(info["success"]) or is_suc
            total_reward += info["success"]
            score = 0

            if i_et % 100 == 0:
                print(
                    f"t: {i_et} | r: {rew:.2f} |  R: {total_reward:.2f} | "
                    f"pos: {obs_cur[:2]} | "
                    f"Max Steps: {self.n_max_steps}"
                )

            rollout.append(obs_cur.copy())
            imgs_rout.append(self.base_env.render())

        imgs_rout = np.array(imgs_rout)
        rollout = np.array(rollout)

        return is_suc, total_reward, score, rollout, imgs_rout

    def ogb_plan_once(self, pl_seed=None, given_probs=None):
        """
        Replanning-capable OGBench planner.
        """
        assert Is_OgB_Robot_Env
        n_comp_full = self.pol_config["ev_n_comp"]

        if pl_seed is not None:
            utils.set_seed(pl_seed)

        if given_probs is not None:
            num_ep = len(given_probs)
        else:
            num_probs = len(self.problems_dict["start_state"])
            num_ep = num_probs if self.plan_n_ep == -100 else self.plan_n_ep

        utils.print_color(f"[ogb_plan_once]: {num_ep=}")

        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = []
        ep_pred_obss_full = []
        ep_rollouts = []
        ep_rollouts_full = []
        ep_targets = []
        ep_is_suc = []
        ep_titles_obs, ep_titles_act = [], []
        ep_cnt_repls = []
        ep_cnt_env_steps = []
        ep_all_plan_trajs_100 = {}
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)

        for i_ep in range(self.ep_st_idx, self.ep_st_idx + num_ep):
            is_suc = False

            if given_probs is not None:
                raise NotImplementedError
            else:
                st_state = self.problems_dict["start_state"][i_ep,]
                gl_pos = self.problems_dict["goal_pos"][i_ep]
                self.check_obs_dim(gl_pos, "full")

            self.base_env.reset()

            if "antmaze" in self.base_env.name:
                self.base_env.set_state_with_obs(st_state)
            elif "humanoidmaze" in self.base_env.name:
                self.base_env.set_state_with_full(st_state)
            elif "antsoccer" in self.base_env.name:
                initial_carrier_id = self.carrier_order[0]
                set_multi_agent_env_from_single_agent_obs(
                    self.base_env,
                    single_obs=st_state,
                    layout=self.ma_layout,
                    active_agent_id=initial_carrier_id,
                )
            elif "pointmaze" in self.base_env.name:
                self.base_env.set_xy_with_0vel(xy=st_state)
            else:
                raise NotImplementedError

            if "antsoccer" in self.base_env.name:
                self.base_env.set_goal(goal_xy=gl_pos[(15, 16),])
                self.base_env.set_ball_start_marker(st_state[(15, 16),])

                final_goal_xy = gl_pos[(15, 16),]
                if self.ma_mode == "mode_c_parallel_handoff":
                    self.ma_setup_parallel_relay_coordinator_for_episode(final_goal_xy=final_goal_xy)
                elif self.ma_mode == "mode_d_closest_ant_parallel":
                    self.ma_setup_closest_ant_relay_coordinator_for_episode(final_goal_xy=final_goal_xy)
                else:
                    self.ma_setup_relay_coordinator_for_episode(final_goal_xy=final_goal_xy)
            else:
                self.base_env.set_goal(goal_xy=gl_pos[:2])

            mujoco.mj_forward(self.base_env.model, self.base_env.data)

            raw_st_state_mj = self.base_env.get_ob()
            st_state_mj = get_agent_single_obs_from_multi_obs(
                raw_st_state_mj,
                layout=self.ma_layout,
                agent_id=self.carrier_order[0],
            )
            target_mj = self.base_env.cur_goal_xy

            if "antmaze" in self.base_env.name:
                assert np.isclose(st_state_mj, st_state).all()
            elif "antsoccer" in self.base_env.name:
                assert np.isclose(st_state_mj, st_state).all()
            elif "pointmaze" in self.base_env.name:
                assert np.isclose(st_state_mj, st_state).all()
            else:
                assert np.isclose(self.base_env.get_state_full(), st_state).all()

            self.base_env.set_start_marker(st_state_mj[:2])

            if self.ma_uses_parallel_step():
                _init_active_agent_id = self.carrier_order[0]
            else:
                _init_active_agent_id = self.ma_get_current_active_agent_id(update_relay=False)

            raw_obs_cur = self.base_env.get_ob().copy()
            obs_cur = get_agent_single_obs_from_multi_obs(
                raw_obs_cur,
                layout=self.ma_layout,
                agent_id=_init_active_agent_id,
            )
            self.check_obs_dim(obs_cur, "obs")

            repl_wp_cfg = self.repl_wp_cfg
            repl_wp_list = sorted(repl_wp_cfg.keys())
            self.policy.n_comp = n_comp_full

            if self.is_replan == "at_given_t":
                last_repl_wpnt = repl_wp_list[-1]
                full_hzn_with_repl = last_repl_wpnt + self.get_comp_hzn(
                    repl_wp_cfg[last_repl_wpnt]
                )

                utils.print_color(
                    f"[ {i_ep=} ] {repl_wp_cfg=}\n"
                    f"{last_repl_wpnt=} {full_hzn_with_repl=}",
                    c="y",
                )
                tot_hzn = full_hzn_with_repl

            elif self.is_replan == "ada_dist":
                self.ada_dist_max_n_repl = self.repl_ada_dist_cfg["max_n_repl"]
                self.ada_dist_thres = self.repl_ada_dist_cfg["thres"]
                self.ada_dist_type = self.repl_ada_dist_cfg["type"]

                if self.ada_dist_type == "m_1":
                    self.ada_dist_comp_sch = self.repl_ada_dist_cfg.get("comp", {})
                    assert len(self.ada_dist_comp_sch) == self.ada_dist_max_n_repl

                elif self.ada_dist_type == "m_2":
                    self.ada_dist_minus_n_wp = self.repl_ada_dist_cfg["ada_dist_minus_n_wp"]
                    self.ada_dist_cond_2_extra = self.repl_ada_dist_cfg["cond_2_extra"]

                    if "used_idxs" not in self.repl_ada_dist_cfg:
                        self.repl_ada_dist_cfg["used_idxs"] = tuple(
                            range(len(self.obs_select_dim))
                        )

                    self.ada_dist_used_idxs = self.repl_ada_dist_cfg["used_idxs"]
                    assert len(self.ada_dist_used_idxs) in [2, 4], "temporary sanity check"

                tot_hzn = 10000
                utils.print_color(f"{tot_hzn=}", c="c")
                utils.print_color(f"{self.repl_ada_dist_cfg=}", c="c")

            elif self.is_replan == False:
                tot_hzn = self.get_comp_hzn(self.policy.n_comp)
                assert self.repl_wp_cfg == {}

            else:
                raise NotImplementedError

            fused_traj_ep = np.zeros(shape=(tot_hzn, self.dfu_ndim), dtype=np.float32)
            all_plan_trajs_ep = []
            imgs_rout = []
            imgs_rout_agv = []

            rollout = [self.base_env.get_qpos_qvel()]
            target_xy = self.base_env.cur_goal_xy
            ep_targets.append(target_xy)

            assert len(target_xy) == 2

            cnt_extras = 0
            total_reward = 0

            if self.is_replan == "ada_dist":
                self.n_max_steps = self.repl_ada_dist_cfg["n_max_steps"]
            else:
                self.n_max_steps = tot_hzn * self.n_act_per_waypnt + self.extra_env_steps

            wp_idx = 0
            cnt_repl = 0

            if self.ma_uses_parallel_step():
                fused_traj_ep_by_agent = {
                    aid: np.zeros(shape=(tot_hzn, self.dfu_ndim), dtype=np.float32)
                    for aid in self.carrier_order
                }
                all_plan_trajs_ep_by_agent = {aid: [] for aid in self.carrier_order}
                cnt_repl_by_agent = {aid: 0 for aid in self.carrier_order}
                prev_dfu_wp_idx_by_agent = {}
                prev_n_comp_by_agent = {}
                goal_cur_by_agent = {}
                last_pick_traj_by_agent = {}

            for i_et in range(self.n_max_steps):
                if self.ma_uses_parallel_step():
                    is_suc, total_reward, cnt_extras, do_break = self.ma_run_parallel_step(
                        i_ep=i_ep,
                        i_et=i_et,
                        gl_pos=gl_pos,
                        n_comp_full=n_comp_full,
                        fused_traj_ep_by_agent=fused_traj_ep_by_agent,
                        all_plan_trajs_ep_by_agent=all_plan_trajs_ep_by_agent,
                        cnt_repl_by_agent=cnt_repl_by_agent,
                        prev_dfu_wp_idx_by_agent=prev_dfu_wp_idx_by_agent,
                        prev_n_comp_by_agent=prev_n_comp_by_agent,
                        goal_cur_by_agent=goal_cur_by_agent,
                        last_pick_traj_by_agent=last_pick_traj_by_agent,
                        is_suc=is_suc,
                        total_reward=total_reward,
                        cnt_extras=cnt_extras,
                        rollout=rollout,
                        imgs_rout=imgs_rout,
                        imgs_rout_agv=imgs_rout_agv,
                    )
                    wp_idx = i_et // self.n_act_per_waypnt
                    score = 0
                    if do_break:
                        break
                    continue

                # Update relay exactly once per env timestep and reuse active_agent_id.
                active_agent_id, active_task = self.ma_get_active_task(update_relay=True)
                force_ma_stage_replan = bool(
                    getattr(self, "ma_stage_switched_last_update", False)
                )
                if i_et % 50 == 0 or force_ma_stage_replan:
                    world_state_dbg = self.ma_get_world_state()
                    dist_to_target = getattr(self.relay_coordinator, "last_ball_to_target_dist", None)
                    dist_str = f"{dist_to_target:.3f}" if dist_to_target is not None else "None"
                    
                    utils.print_color(
                        f"[MA STATE DEBUG] {i_ep=} {i_et=} "
                        f"stage={active_task.get('stage', None)} "
                        f"active_agent_id={active_agent_id} "
                        f"target_xy={active_task.get('target_xy', None)} "
                        f"ball_xy={world_state_dbg['ball_xy']} "
                        f"agent_xy={world_state_dbg['agent_xy']} "
                        f"ball_to_target_dist={dist_str} "
                        f"force_ma_stage_replan={force_ma_stage_replan}",
                        c="m",
                    )
                if active_agent_id is None:
                    utils.print_color(
                        f"[MA Relay] done at {i_ep=} {i_et=}. Ending episode.",
                        c="g",
                    )
                    is_suc = True
                    total_reward += 1.0
                    break

                raw_obs_cur = self.base_env.get_ob().copy()
                obs_cur = get_agent_single_obs_from_multi_obs(
                    raw_obs_cur,
                    layout=self.ma_layout,
                    agent_id=active_agent_id,
                )
                obs_cur_nm = self.full_normalizer.normalize(obs_cur, "observations")

                wp_idx = i_et // self.n_act_per_waypnt
                is_wp_not_start = (wp_idx * self.n_act_per_waypnt) == i_et

                assert len(all_plan_trajs_ep) <= 20, "sanity check, never use so much repl"

                if i_et > 0:
                    cur_obs_subgl_l2_dist = np.linalg.norm(
                        obs_cur[self.obs_select_dim,][self.ada_dist_used_idxs,]
                        - goal_cur[self.ada_dist_used_idxs,]
                    )
                    ada_used_dist = cur_obs_subgl_l2_dist

                if self.is_replan == "at_given_t":
                    is_do_repl_et = wp_idx in repl_wp_cfg

                elif self.is_replan == "ada_dist" and i_et > 0:
                    repl_ada_cond_1 = ada_used_dist > self.ada_dist_thres
                    repl_ada_cond_2 = (
                        wp_idx - prev_dfu_wp_idx - self.ada_dist_cond_2_extra
                    ) > len(all_plan_trajs_ep[-1])

                    is_do_repl_et = (
                        repl_ada_cond_1 or repl_ada_cond_2
                    ) and cnt_repl < self.ada_dist_max_n_repl

                else:
                    is_do_repl_et = False

                is_do_repl_et = is_do_repl_et and is_wp_not_start

                # Stage switch must force immediate replan.
                if force_ma_stage_replan:
                    is_do_repl_et = True

                if i_et == 0 or is_do_repl_et:
                    input_st = obs_cur[self.obs_select_dim,]

                    # Build stage-specific target:
                    # ant1_to_handoff: handoff_xy
                    # ant2_to_handoff: current ball_xy
                    # ant2_to_final: final_goal_xy
                    stage_goal_obs = self.ma_build_stage_goal_obs(
                        base_goal_obs=gl_pos,
                        active_task=active_task,
                    )
                    gl_pos_for_dfu = stage_goal_obs[self.obs_select_dim,]

                    if i_et % 50 == 0 or force_ma_stage_replan:
                        utils.print_color(
                            f"[MA GOAL DEBUG] {i_ep=} {i_et=} "
                            f"stage={active_task.get('stage', None)} "
                            f"active_agent_id={active_agent_id} "
                            f"target_xy={active_task.get('target_xy', None)} "
                            f"base_goal_ball_xy={gl_pos[15:17]} "
                            f"stage_goal_ball_xy={stage_goal_obs[15:17]} "
                            f"gl_pos_for_dfu_ball_xy={gl_pos_for_dfu[15:17]}",
                            c="c",
                        )

                    g_cond = {
                        "st_gl": np.array(
                            [input_st[None,], gl_pos_for_dfu[None,]],
                            dtype=np.float32,
                        )
                    }

                    if is_do_repl_et and i_et > 0:
                        cnt_repl += 1

                        if force_ma_stage_replan and not self.is_replan:
                            self.policy.n_comp = n_comp_full

                        elif self.is_replan == "at_given_t":
                            self.policy.n_comp = repl_wp_cfg[wp_idx]

                        elif self.is_replan == "ada_dist":
                            if self.ada_dist_type == "m_1":
                                raise NotImplementedError

                            elif self.ada_dist_type == "m_2":
                                tmp_cnt_wp = wp_idx - prev_dfu_wp_idx
                                tmp_v1 = max(0, (tmp_cnt_wp - self.ada_dist_minus_n_wp))
                                prev_hzn = len(all_plan_trajs_ep[-1])
                                tmp_n_comp = math.ceil(
                                    (1 - tmp_v1 / prev_hzn) * prev_n_comp
                                )
                                tmp_n_comp = max(1, tmp_n_comp)
                                tmp_cur_hzn = self.get_comp_hzn(num_comp=tmp_n_comp)

                                utils.print_color(
                                    f"{i_et=} {wp_idx=} {prev_dfu_wp_idx=} "
                                    f"{tmp_cnt_wp=} {tmp_v1=}"
                                )
                                utils.print_color(
                                    f"{i_et=} {wp_idx=} {prev_hzn=} "
                                    f"{tmp_n_comp=} {tmp_cur_hzn=} "
                                    f"{ada_used_dist=:.2f}"
                                )

                            self.policy.n_comp = tmp_n_comp

                        else:
                            raise NotImplementedError

                    else:
                        assert self.policy.n_comp == n_comp_full

                    m_out = self.policy.gen_cond_stgl(
                        g_cond=g_cond,
                        b_s=self.b_size_per_prob,
                    )

                    pick_traj = m_out.pick_traj

                    utils.print_color(f"[ Run Planner ]{i_et=} {wp_idx=}", c="y")
                    utils.print_color(f"{pick_traj.shape=}")

                    tmp_tj_end_idx = wp_idx + len(pick_traj)

                    if tmp_tj_end_idx > len(fused_traj_ep):
                        pick_traj = pick_traj[: len(fused_traj_ep) - wp_idx]
                        tmp_tj_end_idx = wp_idx + len(pick_traj)

                    if self.is_replan == "at_given_t" and repl_wp_list[-1] == wp_idx:
                        assert tmp_tj_end_idx == len(fused_traj_ep)

                    elif self.is_replan == "ada_dist":
                        pass

                    elif self.is_replan == False:
                        if not (is_do_repl_et and i_et > 0):
                            assert tmp_tj_end_idx == len(fused_traj_ep)

                    fused_traj_ep[wp_idx:tmp_tj_end_idx] = pick_traj
                    fused_traj_ep[tmp_tj_end_idx:] = pick_traj[-1]

                    all_plan_trajs_ep.append(pick_traj)

                    prev_dfu_wp_idx = wp_idx
                    prev_n_comp = self.policy.n_comp

                if self.act_control == "pred_inv":
                    wp_idx = min(wp_idx, len(fused_traj_ep) - 1)

                    goal_cur = fused_traj_ep[wp_idx]

                    subgoal_xy = goal_cur[:2].copy()
                    current_stage_name = active_task.get("stage_name", "")
                    if active_agent_id == 2 and current_stage_name in ["ant2_to_handoff", "ant2_to_final"]:
                        world_state = self.ma_get_world_state()
                        ant1_xy = world_state["agent_xy"][1]
                        ant2_xy = world_state["agent_xy"][2]

                        dist_between_ants = np.linalg.norm(ant2_xy - ant1_xy)
                        if dist_between_ants < 3.0:
                            repulsion_vector = ant2_xy - ant1_xy
                            repulsion_vector = repulsion_vector / (np.linalg.norm(repulsion_vector) + 1e-5)

                            subgoal_xy += repulsion_vector * 1.5
                    # --------------------------------------------------

                    if self.is_use_subgoal_marker:
                        self.base_env.set_subgoal_waypnt(subgoal_xy)

                    goal_cur_nm = self.train_normalizer.normalize(goal_cur, "observations")

                    obs_cur_nm = utils.to_torch(obs_cur_nm)[None,]
                    goal_cur_nm = utils.to_torch(goal_cur_nm)[None,]
                    act_pred_nm = self.inv_model(obs_cur_nm, goal_cur_nm).cpu().numpy()[0]
                    act_pred = self.full_normalizer.unnormalize(act_pred_nm, "actions")

                elif self.act_control == "pd_ogb":
                    wp_idx = min(wp_idx, len(fused_traj_ep) - 1)

                    goal_cur = fused_traj_ep[wp_idx]
                    self.base_env.set_subgoal_waypnt(goal_cur[:2])

                    act_pred_nm = (goal_cur - obs_cur) * 5
                    act_pred = self.full_normalizer.unnormalize(act_pred_nm, "actions")

                elif self.act_control == "dfu_force":
                    raise NotImplementedError

                else:
                    raise NotImplementedError(f"Unknown act_control={self.act_control}")

                joint_action = make_single_active_joint_action(
                    act_pred,
                    active_agent_id=active_agent_id,
                    layout=self.ma_layout,
                )

                if i_et % 50 == 0 or force_ma_stage_replan:
                    utils.print_color(
                        f"[MA ACTION DEBUG] {i_ep=} {i_et=} "
                        f"stage={active_task.get('stage', None)} "
                        f"active_agent_id={active_agent_id} "
                        f"norm_ant1={np.linalg.norm(joint_action[:8]):.3f} "
                        f"norm_ant2={np.linalg.norm(joint_action[8:16]):.3f}",
                        c="m",
                    )

                raw_obs_cur, rew, terminated, truncated, info = self.base_env.step(joint_action)

                obs_cur = get_agent_single_obs_from_multi_obs(
                    raw_obs_cur,
                    layout=self.ma_layout,
                    agent_id=active_agent_id,
                )

                is_suc = bool(info["success"]) or is_suc

                total_reward += rew
                score = 0

                if i_et % 100 == 0 and i_et != 0:
                    tmp_dist_1 = np.linalg.norm(
                        obs_cur[self.obs_select_dim,] - goal_cur
                    ).item()

                    print(
                        f"t: {i_et} |"
                        f"pos: {ogb_get_rowcol_obs_trajs_from_xy(self.base_env, obs_cur[None, :2])[0]} | "
                        f"obs_subgl_dist: {tmp_dist_1:.3f} | "
                        f"Max Steps: {self.n_max_steps}"
                    )

                rollout.append(self.base_env.get_qpos_qvel())
                post_task_agent_id, post_task = self.ma_get_active_task(update_relay=False)
                imgs_rout.append(
                    self.ma_render_frame_with_subtitles(
                        active_agent_id=post_task_agent_id,
                        active_task=post_task,
                    )
                )

                if self.is_rd_agv:
                    imgs_rout_agv.append(self.render_agv_img())

                if is_suc:
                    utils.print_color(f"{i_ep=} {i_et=} {is_suc=}")
                    cnt_extras += 1
                    if cnt_extras == 30:
                        break

            if self.ma_uses_parallel_step():
                if self.ma_mode == "mode_d_closest_ant_parallel":
                    _report_agent_id = self.closest_ant_relay_coordinator.carrier_id
                    if _report_agent_id is None:
                        _report_agent_id = self.carrier_order[0]
                else:
                    _report_agent_id = self.carrier_order[1]
                fused_traj_ep = fused_traj_ep_by_agent[_report_agent_id]
                all_plan_trajs_ep = all_plan_trajs_ep_by_agent[_report_agent_id]
                cnt_repl = cnt_repl_by_agent[_report_agent_id]
                pick_traj = last_pick_traj_by_agent.get(
                    _report_agent_id,
                    last_pick_traj_by_agent.get(self.carrier_order[0]),
                )

            ep_all_plan_trajs_100[i_ep] = all_plan_trajs_ep

            tmp_dir_path = self.get_sample_savedir(i_ep)
            tmp_obs_trajs = ogb_get_rowcol_obs_trajs_from_xy_list(
                self.base_env,
                all_plan_trajs_ep,
            )

            tmp_trajs_ball = self.extract_ball_trajs_ev(
                all_plan_trajs_ep,
                do_to_ij=True,
            )

            img_obs = self.renderer.composite(
                f"{tmp_dir_path}/ep{i_ep}_et{i_et}_wp{wp_idx}_"
                f"cp{self.policy.n_comp}_h{len(pick_traj)}_re{cnt_repl}_allpred.png",
                tmp_obs_trajs,
                ncol=n_col,
                trajs_ball=tmp_trajs_ball,
            )

            rollout = np.array(rollout)
            imgs_rout = np.array(imgs_rout)

            tmp_dir_path = self.get_sample_savedir(i_ep)
            utils.save_imgs_to_mp4(
                imgs=imgs_rout,
                save_path=f"{tmp_dir_path}/ep{i_ep}_{is_suc}.mp4",
                fps=self.vid_fps,
                n_repeat_first=10,
            )

            if self.is_rd_agv and is_suc:
                utils.save_imgs_to_mp4(
                    imgs=imgs_rout_agv,
                    save_path=f"{tmp_dir_path}/ep{i_ep}_{is_suc}_agv.mp4",
                    fps=self.vid_fps,
                    n_repeat_first=10,
                )

            ep_pred_obss_full.append(fused_traj_ep)
            ep_rollouts_full.append(rollout)

            fused_traj_ep = ogb_get_rowcol_obs_trajs_from_xy(self.base_env, fused_traj_ep)
            rollout_ij2d = ogb_get_rowcol_obs_trajs_from_xy(self.base_env, rollout[:, :2])

            ep_pred_obss.append(fused_traj_ep)
            ep_rollouts.append(rollout_ij2d)
            ep_titles_obs.append(f"PredObs: {i_ep}_o{self.dfu_ndim}_{is_suc}")
            ep_titles_act.append(f"Act: {i_ep}_{is_suc}")

            ep_is_suc.append(is_suc)

            ep_scores.append(score)
            ep_total_rewards.append(total_reward)
            ep_cnt_repls.append(cnt_repl)

            if len(ep_pred_obss) % trajs_per_img == 0 or i_ep == num_ep - 1:
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img
                tmp_st_idx -= self.ep_st_idx
                tmp_end_idx -= self.ep_st_idx

                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tgts = ogb_get_rowcol_obs_trajs_from_xy(self.base_env, tmp_tgts)

                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum()
                tmp_avg_sr = int(np.mean(ep_is_suc[tmp_st_idx:tmp_end_idx]) * 100)

                get_is_non_keypt = getattr(self.diffusion, "get_is_non_keypt", None)

                if get_is_non_keypt is not None:
                    raise NotImplementedError
                else:
                    is_non_keypt = None

                tmp_trajs_ball_pred = self.extract_ball_trajs_ev(
                    ep_pred_obss_full[tmp_st_idx:tmp_end_idx],
                    do_to_ij=True,
                )
                tmp_trajs_ball_rout = self.extract_ball_trajs_ev(
                    ep_rollouts_full[tmp_st_idx:tmp_end_idx],
                    do_to_ij=True,
                )

                img_obs, rows_obs = self.renderer.composite(
                    None,
                    np.array(ep_pred_obss[tmp_st_idx:tmp_end_idx]),
                    ncol=n_col,
                    goal=tmp_tgts,
                    titles=tmp_tls_obs,
                    return_rows=True,
                    is_non_keypt=is_non_keypt,
                    trajs_ball=tmp_trajs_ball_pred,
                )

                img_act, rows_act = self.renderer.composite(
                    None,
                    ep_rollouts[tmp_st_idx:tmp_end_idx],
                    ncol=n_col,
                    goal=tmp_tgts,
                    titles=tmp_tls_act,
                    return_rows=True,
                    trajs_ball=tmp_trajs_ball_rout,
                )

                f_path_3 = join(
                    self.savepath,
                    f"total/{tmp_st_idx}_act_obs_nns{tmp_num_f}_sr{tmp_avg_sr}.png",
                )
                n_rows = len(rows_obs)
                img_whole = []

                for i_r in range(n_rows):
                    img_whole.append(np.concatenate([rows_act[i_r], rows_obs[i_r]], axis=0))

                img_whole = np.concatenate(img_whole)
                utils.save_img(f_path_3, img_whole)

        self.policy.n_comp = n_comp_full

        utils.print_color(self.base_env.name)

        ep_is_suc = np.array(ep_is_suc)
        ep_srate = ep_is_suc.mean() * 100
        ep_fail_idxs = np.where(ep_is_suc == False)[0]
        assert len(ep_is_suc) == num_ep

        avg_ep_scores = np.mean(ep_scores)
        avg_ep_rewards = np.mean(ep_total_rewards)

        utils.print_color(f"[avg suc rate] {ep_srate=}")

        json_path = join(self.savepath, "00_rollout.json")

        sc_low_idxs = np.where(np.array(ep_scores) < self.score_low_limit)[0].tolist()
        sc_low_idxs_d = dict(
            zip(
                sc_low_idxs,
                np.round(ep_scores, decimals=2)[sc_low_idxs].tolist(),
            )
        )
        print(f"{sc_low_idxs_d=}")

        avg_t_dict = self.get_avg_sampling_time()

        ep_range = range(len(ep_scores))
        json_data = OrderedDict(
            [
                ("num_ep", num_ep),
                ("ep_srate", ep_srate),
                ("avg_ep_scores", avg_ep_scores),
                ("avg_ep_rewards", avg_ep_rewards),
                ("pl_seed", pl_seed),
            ]
        )

        json_data = self.update_j_data(json_data)
        json_data.update(
            [
                ("p_type", "plan_once"),
                ("avg_t_dict", avg_t_dict),
                ("ep_fail_idxs", ep_fail_idxs.tolist()),
                ("sc_low_idx", sc_low_idxs_d),
                ("ep_is_suc", ep_is_suc.tolist()),
                ("ep_cnt_repls", ep_cnt_repls),
                ("ep_scores", dict(zip(ep_range, ep_scores))),
                ("ep_total_rewards", dict(zip(ep_range, ep_total_rewards))),
                ("ncp_pred_time_list", self.policy.ncp_pred_time_list),
            ]
        )

        utils.save_json(json_data, json_path)

        if self.is_save_pkl:
            self.save_results_to_pkl(
                ep_all_plan_trajs_100=ep_all_plan_trajs_100,
                ep_pred_obss_full=ep_pred_obss_full,
                ep_rollouts_full=ep_rollouts_full,
            )

        new_savepath = f"{self.savepath.rstrip(os.sep)}-sr{int(ep_srate)}/"
        utils.rename_fn(self.savepath, new_savepath)
        new_json_path = json_path.replace(self.savepath, new_savepath)
        utils.print_color(f"new_json_path: {new_json_path} \n")

        return json_data

    def save_results_to_pkl(self, **kwargs):
        import pickle

        num_ep = len(kwargs["ep_rollouts_full"])
        pkl_path = join(self.savepath, "00_rout.pkl")

        for i_ep in range(num_ep):
            tmp_tj = kwargs["ep_pred_obss_full"][i_ep]
            kwargs["ep_pred_obss_full"][i_ep] = tmp_tj[
                : len(kwargs["ep_rollouts_full"][i_ep])
            ]

        with open(f"{pkl_path}", "wb") as file:
            pickle.dump(kwargs, file)

        print(f"[save to pickle] {pkl_path}")

    def get_avg_sampling_time(self):
        ncp_times = np.array(self.policy.ncp_pred_time_list)
        max_ncp = np.unique(ncp_times[:, 0]).max()
        is_max_ncp = np.isclose(max_ncp, ncp_times[:, 0])
        n_max_ncp = is_max_ncp.sum().item()
        max_idxs = np.where(is_max_ncp)[0]

        out_dict = {}
        n_rm = 2

        tmp_t_list = ncp_times[max_idxs[n_rm:], 1]
        if len(tmp_t_list) == 0:
            tmp_t_list = [0]

        out_dict[max_ncp] = {
            "n": n_max_ncp - n_rm,
            "t": np.round(np.mean(tmp_t_list), 4).item(),
            "e": np.round(np.std(tmp_t_list), 4).item(),
        }

        return out_dict

    def get_sample_savedir(self, i_ep):
        div_freq = 10
        subdir = str((i_ep // div_freq) * div_freq)
        sample_savedir = os.path.join(self.savepath, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir

    def check_is_same_nmlizer(self):
        tmp_1 = self.train_normalizer.normalizers["observations"].mins[:2]
        tmp_2 = self.full_normalizer.normalizers["observations"].mins[:2]
        assert np.isclose(tmp_1, tmp_2).all()

    def extract_ball_trajs_ev(self, all_plan_trajs_ep, do_to_ij):
        """
        Eval-time version of extracting ball trajectories.
        """
        if self.is_soccer_task:
            if all_plan_trajs_ep[0].shape[1] == 42:
                tmp_all_ball_tj_list = [tmp_tj[:, (15, 16)] for tmp_tj in all_plan_trajs_ep]
            else:
                tmp_all_ball_tj_list = [tmp_tj[:, -2:] for tmp_tj in all_plan_trajs_ep]

            if do_to_ij:
                tmp_trajs_2 = ogb_get_rowcol_obs_trajs_from_xy_list(
                    self.base_env,
                    tmp_all_ball_tj_list,
                )
            else:
                tmp_trajs_2 = tmp_all_ball_tj_list
        else:
            tmp_trajs_2 = None

        return tmp_trajs_2

    def update_j_data(self, json_data: OrderedDict):
        json_data.update(
            [
                ("epoch_diffusion", self.epoch),
                ("cond_w", self.diffusion.condition_guidance_w),
                ("p_h5path", self.problems_h5path),
                ("var_temp", self.diffusion.var_temp),
                ("use_ddim", self.diffusion.use_ddim),
                ("ddim_eta", self.diffusion.ddim_eta),
                ("ddim_steps", self.diffusion.ddim_num_inference_steps),
                ("is_replan", self.is_replan),
                ("repl_wp_cfg", self.repl_wp_cfg),
                ("repl_ada_dist_cfg", self.repl_ada_dist_cfg),
                ("extra_env_steps", self.extra_env_steps),
                ("b_size_per_prob", self.b_size_per_prob),
                ("pol_config", self.pol_config),
                ("tj_blder_config", self.tj_blder_config),
                ("n_batch_acc_probs", self.n_batch_acc_probs),
                ("max_episode_steps", self.n_max_steps),
                ("act_control", self.act_control),
                ("n_act_per_waypnt", self.n_act_per_waypnt),
                ("inv_model_path", self.inv_model_path),
                ("inv_epoch", self.inv_epoch),
                ("is_inv_train_mode", self.inv_model.training),
                ("ep_st_idx", self.ep_st_idx),
                ("hostname", socket.gethostname()),
                ("ma_mode", self.ma_mode),
                ("use_parallel_relay", self.ma_mode == "mode_d_closest_ant_parallel"),
                ("is_vid_subtitles", self.is_vid_subtitles),
                ("num_agents", self.num_agents),
                ("carrier_order", self.carrier_order),
                ("handoff_points", [p.tolist() for p in self.handoff_points]),
                ("retreat_points", [p.tolist() for p in self.retreat_points]),
                ("retreat_threshold", self.retreat_threshold),
                ("handoff_ball_radius", self.handoff_ball_radius),
                ("ant2_reach_ball_radius", self.ant2_reach_ball_radius),
                ("ball_goal_threshold", self.ball_goal_threshold),
            ]
        )
        return json_data

    def check_obs_dim(self, obs_in, o_type="obs"):
        if "antmaze" in self.base_env.name:
            assert obs_in.shape == (29,)
        elif "human" in self.base_env.name:
            if o_type == "obs":
                assert obs_in.shape == (69,)
            else:
                assert obs_in.shape == (55,)
        elif "antsoccer" in self.base_env.name:
            assert obs_in.shape == (42,)
        elif "pointmaze" in self.base_env.name:
            assert obs_in.shape == (2,)
        else:
            raise NotImplementedError

    def save_env_cur_img(self, sv_idx):
        tmp_img = self.base_env.render()
        utils.save_img(f"./luotest_{sv_idx}.png", tmp_img)

    def get_comp_hzn(self, num_comp):
        return self.diffusion.get_total_hzn(num_comp=num_comp)

    def render_agv_img(self):
        self.base_env.camera_name = "back_luo_v3"

        vc_name = "visual_circle"
        rgba_ori = self.base_env.model.geom(vc_name).rgba.copy()
        self.base_env.model.geom(vc_name).rgba = np.array([0.0, 0.0, 0.0, 0.0])
        img_r = self.base_env.render()

        self.base_env.camera_name = None
        self.base_env.model.geom(vc_name).rgba = rgba_ori

        return img_r