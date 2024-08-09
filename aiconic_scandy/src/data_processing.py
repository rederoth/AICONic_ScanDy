import glob
import os.path
import pickle
import random
import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from data_processing_hq import generate_res_df

def generate_colormap_old(N=100):
    HSV_tuples = [(x * 1.0 / N, 0.7, 0.6) for x in range(N)]
    rgb_out = []
    for rgb in HSV_tuples:
        rgb_out.append(colorsys.hsv_to_rgb(*rgb))
    random.shuffle(rgb_out)
    rgb_colors = np.array([(0.0, 0.0, 0.0)] + rgb_out)
    custom_cmap = matplotlib.colors.ListedColormap(rgb_colors, N=N)
    return custom_cmap


random.seed(42)
custom_cmap = generate_colormap_old()


def process_experiment_data(folder_path):
    # store resulting df as csv
    res_df = generate_res_df(folder_path)
    # make plots for every frame
    pathes = glob.glob(os.path.join(folder_path, "*.pickle"))
    pathes.sort()
    plot_dir = os.path.join(folder_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for fp in pathes: #[-10:]
        with open(fp, "rb") as f:
            data = pickle.load(f)
            fig, axs, = plt.subplots(3, 3)
            axs[0, 0].imshow(data["sensitivity_img"].cpu().numpy())
            # plot a large red cross on the maximum of data["gaze_img"]
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()), data["gaze_img"].cpu().numpy().shape)
            axs[0,0].plot(max_coords[1], max_coords[0], 'rx', markersize=10)            
            axs[0, 0].set_title("Gaze & Sensitivity")
            # axs[0, 0].plot(data["gaze_history"][:, 1], data["gaze_history"][:, 0], "r")
            axs[0, 1].imshow(data["task_imp_img"], cmap="hot", vmin=0, vmax=1)
            axs[0, 1].set_title("Task relevance (saliency)")
            axs[0, 2].imshow(data["flow_visualized"], vmin=0, vmax=1)
            axs[0, 2].set_title("Optical flow")
            axs[1, 0].imshow(data["obj_seg_img"], cmap=custom_cmap, vmin=0, vmax=100, interpolation='none')
            axs[1, 0].plot(max_coords[1], max_coords[0], "rx", markersize=10)
            axs[1, 0].set_title("Resulting objects")
            axs[1, 1].imshow(data["obj_cert_img"], vmin=0, vmax=1, interpolation='none')
            axs[1, 1].set_title("Object certainty")
            axs[1, 2].imshow(np.round(data["obj_seg_measurements"][3].cpu().numpy()).astype(bool).astype(np.uint8) * 12 + np.round(data["obj_seg_measurements"][4].cpu().numpy()).astype(bool).astype(np.uint8) * 2, cmap=custom_cmap, vmin=0, vmax=100, interpolation='none')
            axs[1, 2].set_title("SAM object (gaze_loc)")
            axs[2, 0].imshow(data["obj_seg_measurements"][0].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=100, interpolation='none')
            axs[2, 0].set_title("Appearance-based objects")
            axs[2, 1].imshow(data["obj_seg_measurements"][1].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=100, interpolation='none')
            axs[2, 1].set_title("Motion-based objects")
            axs[2, 2].imshow(np.round(data["obj_seg_measurements"][2].cpu().numpy()).astype(np.uint8), cmap=custom_cmap, vmin=0, vmax=100, interpolation='none')
            axs[2, 2].set_title("SAM objects (global)")

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, fp[-11:-7]))
            plt.close(fig)



if __name__ == "__main__":
    # process_experiment_data("/media/vito/TOSHIBA EXT/scanpath_results/2023-09-13-13-31-14/config_0/-2HFZjPOCMk")
    pathes = glob.iglob("/media/vito/TOSHIBA EXT/scanpath_results/*")
    todo_sequences = []
    for p in pathes:
        single_sequence_pathes = glob.iglob(os.path.join(p, "*", "*"))
        for sp in single_sequence_pathes:
            if not os.path.isdir(sp):
                continue
            if os.path.isdir(os.path.join(sp, "plots")):
                continue
            todo_sequences.append(sp)
    todo_sequences = sorted(todo_sequences)
    todo_sequences.reverse()
    for p in tqdm(todo_sequences):
        print(p)
        process_experiment_data(p)
