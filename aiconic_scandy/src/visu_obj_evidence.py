import glob
import os.path
import pickle
from collections import Counter
import itertools
import random
import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import seaborn as sns

from data_processing_hq import generate_res_df, read_config_to_df, generate_colormap

random.seed(42)
custom_cmap = generate_colormap()

def go_through_frames(folder_path, save_objs_in_frames=None):
    plot_dir = os.path.join(folder_path, "indiv_obj_plots")
    os.makedirs(plot_dir, exist_ok=True)

    if os.path.isfile(os.path.join(folder_path, f'res_foveations.csv')):
        res_df = pd.read_csv(os.path.join(folder_path, f'res_foveations.csv'))
    else:
        res_df = generate_res_df(folder_path)

    evidence_keys = []
    evidence_dict_over_time = []
    pathes = sorted(glob.glob(os.path.join(folder_path, "*.pickle")))

    # save_objs_in_frames = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14]

    for f_idx, fp in enumerate(pathes):
        with open(fp, "rb") as f:
            data = pickle.load(f)
            # evidence_keys.append([int(float(key)) for key in data['evidence_dict'].keys()])
            evidence_keys.append(list(data['evidence_dict'].keys()))
            evidence_dict_over_time.append(data['evidence_dict'])
            if save_objs_in_frames is not None:
                max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                              data["gaze_img"].cpu().numpy().shape)
                fig, axs, = plt.subplots(3, 4, figsize=(10, 6.5))
                all_objects = data["obj_seg_img"].astype(np.uint32)
                for i, ax in enumerate(axs.flat):
                    sgl_obj_map = all_objects == save_objs_in_frames[i]
                    ax.imshow(sgl_obj_map * max(1, save_objs_in_frames[i]), cmap=custom_cmap, vmin=0, vmax=N_colors,
                              interpolation='none')
                    ax.set_title(f"Obj ID {save_objs_in_frames[i]}")
                    ax.set_axis_off()
                ax.set_title("All objects")
                ax.imshow(all_objects, cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
                ax.plot(max_coords[1], max_coords[0], 'rx', markersize=15)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, fp[-11:-7]))
                plt.close(fig)

    if save_objs_in_frames is None:
        item_count = Counter(itertools.chain(*evidence_keys))
        most_common_ids = [id for id, num in item_count.most_common(12)]
        print(item_count.most_common(12))
        return most_common_ids, evidence_dict_over_time
    else:
        return save_objs_in_frames, evidence_dict_over_time


def plot_evidence_over_time(vis_obj_ids=None, threshold=None, save=True):
    fig, ax = plt.subplots(figsize=(5, 3))
    if vis_obj_ids is None:
        ax.plot(df)
    else:
        for obj in vis_obj_ids:
            ax.plot(df[obj], label=f"Obj {obj}")
        ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Object Evidence')
    sns.despine()
    if threshold is not None:
        ax.axhline(y=float(threshold), ls=":", color="k")
    if save:
        plt.savefig(os.path.join(folder_path, "obj_evidence_over_time"))
    else:
        plt.show()

if __name__ == "__main__":
    run_dir = "2024-03-07-19-12-39_BOUNCYCASTLE_task0.001_entropy0.5_dv3.0_sig0.35"
    # run_dir = "2024-03-07-20-57-21_BOUNCYCASTLE_task0.001_entropy0.5_dv3.0_sig0.0"
    folder_path = os.path.join("/media/vito/scanpath_backup/scanpath_results_current/", run_dir, "config_0/-5FU8vEKtyE")

    vis_obj_ids, evidence_dict_over_time = go_through_frames(folder_path, None)
    df = pd.DataFrame(evidence_dict_over_time)
    df_conf = read_config_to_df(folder_path)
    decision_threshold = df_conf["decision_threshold"].iloc[0]

    plot_evidence_over_time(vis_obj_ids, threshold=decision_threshold, save=False) # threshold=decision_threshold

    # keys_set = set().union(*evidence_dict_over_time)
    # keys = list(keys_set)
    # num_keys = len(keys)
    # time_points = range(len(evidence_dict_over_time))
    # values = [[] for _ in range(num_keys)]
    #
    # for time_data in evidence_dict_over_time:
    #     for idx, key in enumerate(keys):
    #         values[idx].append(time_data.get(key, None))
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for idx, key in enumerate(keys):
    #     plt.plot(time_points, values[idx], label=key)
