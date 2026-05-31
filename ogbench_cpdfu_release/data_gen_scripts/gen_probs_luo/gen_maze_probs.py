import random, sys, os; sys.path.append('./')
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
import torch, yaml, pdb

from tap import Tap
from types import SimpleNamespace
from data_gen_scripts.gen_probs_luo.g_probs_utils import gen_ogb_probs_utils, save_prob_imgs
from ogbench.luo_utils.eval_utils import print_color

class Parser(Tap):
    sub_conf: str
    h5_root: str

def main():
    args = Parser().parse_args()
    # seed_u = rs_cfg['seed_u']
    seed_u = 0
    random.seed(seed_u) ## useless actually
    np.random.seed(seed_u)
    torch.manual_seed(seed_u)

    file_path = 'data_gen_scripts/gen_probs_luo/gen_ogb_maze_probs_confs.yaml'
    # Open the YAML file and load its contents
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        rs_cfg = config[args.sub_conf]
        rs_cfg = SimpleNamespace(**rs_cfg)

    ## ----- Temporal Handling ---------
    ## Seems all maze envs just have 5 tasks
    rs_cfg.num_tasks = 5
    rs_cfg.env_name = rs_cfg.dataset_name
    ## ---------------------------------
    # pdb.set_trace()


    all_prob_dict = gen_ogb_probs_utils(rs_cfg,)

    # h5_root = '/coc/flash7/yluo470/robot2024/hi_diffuser/data/ogb_maze/ev_probs/'
    h5_root = args.h5_root
    h5_save_path = f'{h5_root}/{args.sub_conf}.hdf5'

    img_root = './cache/'
    img_save_path = f'{img_root}/{args.sub_conf}/'
    

    ## ----------------------------------
    import h5py
    ## Finished all, save to hdf5
    with h5py.File(h5_save_path, 'w') as file:
        file.create_dataset('start_state', data=all_prob_dict['start_state'])
        file.create_dataset('goal_pos', data=all_prob_dict['goal_pos'])

        # file.attrs['env_seed'] = None
        file.attrs['env_name'] = rs_cfg.env_name
        file.attrs['dataset_name'] = rs_cfg.dataset_name

        # pdb.set_trace()
        # assert file.keys() in all_prob_dict.keys()
        print( f'hdf5: {file.keys()=}' )
    
    ## lock file
    if 'luotest' not in args.sub_conf:
        os.chmod(h5_save_path, 0o444)


    # pdb.set_trace()
    ## goal_rd_img only contains the goal point
    ## img_init_obs also contains the agent inside the img
    # save_prob_imgs(rs_cfg, img_save_path, all_prob_dict['goal_rd_img'])
    save_prob_imgs(rs_cfg, img_save_path, all_prob_dict['img_init_obs'])

    
    print_color(f'[save to] {h5_save_path=}')




    return





if __name__ == "__main__":
    main()