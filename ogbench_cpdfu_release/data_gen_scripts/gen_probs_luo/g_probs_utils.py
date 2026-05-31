
import numpy as np
import gymnasium ## as gym
from ogbench.luo_utils.eval_utils import print_color
import ogbench, pdb


def gen_ogb_probs_utils(rs_cfg,):
    """
    generate the eval problems in two for loops
    """
    np.set_printoptions(precision=3, suppress=True)
    
    dset_name = rs_cfg.dataset_name
    e_name = rs_cfg.env_name ## they are the same
    print_color(f'ogb env name: {e_name=}, {dset_name=}')
    
    wrapped_env = ogbench.make_env_and_datasets(e_name, env_only=True)
    env = wrapped_env.unwrapped

    ## seed for resetting, make sure reproducible
    ## env.set_seed_addn(0) ## move to below

    ## tasks_idxs = [1,2,3,4,5] ## rs_cfg.num_tasks 
    assert env.num_tasks == 5
    assert rs_cfg.num_tasks == env.num_tasks
    task_idxs = list( range(1, env.num_tasks + 1) )
    print_color( f'[gen_ogb_probs_utils] {task_idxs=}' )

    out_data = dict(start_state=[], goal_pos=[], 
                    goal_rd_img=[], img_init_obs=[],)

    # pdb.set_trace()
    cnt_prob = 0

    for tk_idx in task_idxs:
        
        print_color( f'[Current] {tk_idx=}', c='y' )
        for _ in range(rs_cfg.num_eval_ep):
            tmp_sd = rs_cfg.env_seed_start + cnt_prob

            env.set_seed_addn(tmp_sd)
            # Reset the environment and set the evaluation task.
            ob_cur, info = env.reset(
                options=dict(
                    task_id=tk_idx,  # Set the evaluation task. Each environment provides five
                                    # evaluation goals, and `task_id` must be in [1, 5].
                    render_goal=True,  # Set to `True` to get a rendered goal image (optional).
                )
            )

            ## NOTE: Feb 2: step one action so that the ant/agent will be in a more natural pose
            tmp_n_acts = getattr(rs_cfg, 'step_n_acts_after_init', 0)
            if tmp_n_acts > 0:
                for _ in range(tmp_n_acts):
                    env.step( np.zeros(shape=env.action_space.shape) )
                ob_cur = env.get_ob()
            # pdb.set_trace()



            ## a state: e.g., (29,)
            goal = info['goal']  # Get the goal observation to pass to the agent.
            ## an image: (h,w,3), for antmaze giant: (200,200,3)
            goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).
            img_init_obs = env.render()

            # pdb.set_trace() ## check ob
            
            ## just states
            st_state = ob_cur
            gl_pos = goal

            # pdb.set_trace()

            if len(st_state) == 29 : ## Ant
                pass
            elif len(st_state) == 69 and getattr(rs_cfg, 'is_save_full_jnt', False): ## Human
                ## replace with full qpos and qvel for humanoid
                # pdb.set_trace()
                ## st_state shape: (55,)
                st_state = np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()])
                gl_pos = info['goal_full_jnt_state']
                # pdb.set_trace()
            elif len(st_state) == 42:
                pass
            elif len(st_state) ==  2:
                pass
            else:
                raise NotImplementedError
            
            if cnt_prob % 5 == 0:
                print(f'{cnt_prob} After: {st_state=}, {gl_pos=}')
            
            # pdb.set_trace()


            out_data['start_state'].append( st_state )
            out_data['goal_pos'].append( gl_pos )
            out_data['goal_rd_img'].append( goal_rendered )
            out_data['img_init_obs'].append( img_init_obs )
            
            cnt_prob += 1

            assert cnt_prob == len(out_data['start_state'])

    env.close()

    print_color(f'total {cnt_prob=}', c='y')

    ## n_p, 29/69
    out_data['start_state'] = np.array( out_data['start_state'] )
    ## n_p, 29/69
    out_data['goal_pos'] = np.array( out_data['goal_pos'] )

    out_data['goal_rd_img'] = np.array( out_data['goal_rd_img'] )
    out_data['img_init_obs'] = np.array( out_data['img_init_obs'] )



    # pdb.set_trace()


    return out_data


### ----------------------

import os, imageio
import os.path as osp

def save_img(save_path: str, img: np.ndarray):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    imageio.imsave(save_path, img)
    print(f'[save_img] {save_path}')

def save_prob_imgs(rs_cfg, img_root_dir, imgs):
    
    print_color(f'[save_prob_imgs] {imgs.shape=}')

    num_probs = len(imgs)
    n_probs_per_tk = num_probs // rs_cfg.num_tasks
    
    # pdb.set_trace()

    for i_tk in range(rs_cfg.num_tasks):
        cur_dir = f'{img_root_dir}/task_{str(i_tk+1)}'
        os.makedirs(cur_dir, exist_ok=True)

        print(f'\n\n{cur_dir=}\n')
        for i_pr in range(n_probs_per_tk):
            tmp_path = f'{cur_dir}/prob_{i_pr}.png'
            save_img(tmp_path, imgs[ i_pr + i_tk*n_probs_per_tk, :,:,: ])
            # print(tmp_path)


    print(f'[save_prob_imgs] to {cur_dir}')
    
    return


###

