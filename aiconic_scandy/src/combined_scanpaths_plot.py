import glob
import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_processing_hq import read_config_to_df, make_video_from_frames


def make_combined_scanpathes_vid(folder_paths):
    markersize = 15
    video_name = folder_paths[0].split('/')[-1]

    all_gaze_data = []
    for p in folder_paths:
        with open(os.path.join(p, "raw_results.pickle4"), "rb") as f:
            raw_results = pickle.load(f)
            gaze_data = np.array(raw_results[0], int)
            all_gaze_data.append(gaze_data)

    plot_dir = os.path.join(folder_paths[0], "videos")
    os.makedirs(plot_dir, exist_ok=True)

    frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data_all', video_name, 'images', f'{1:04d}.png')
    img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)
    shape = img.shape[:2]

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    orig_video_frames = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',]
    for i in tqdm(range(len(all_gaze_data[0]))):  # [-10:]
        frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data_all', video_name, 'images', f'{i:04d}.png')
        img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)
        imshow = plt.imshow(img, animated=True)
        all_coords = []
        for participant_i in range(len(all_gaze_data)):
            # plot a large red cross on the maximum of data["gaze_img"]
            coords = plt.plot(all_gaze_data[participant_i][i][1], all_gaze_data[participant_i][i][0], color=colors[participant_i], marker='x', markersize=markersize, animated=True)
            all_coords.extend(coords)
        orig_video_frames.append([imshow, *all_coords])
    make_video_from_frames(fig, orig_video_frames, os.path.join(plot_dir, "combined_participants_vid.mp4"))
    plt.close(fig)


def make_combined_scanpathes_vid_with_trails(folder_paths):
    markersize = 15
    video_name = folder_paths[0].split('/')[-1]

    all_gaze_data = []
    all_fov_tables = []
    for p in folder_paths:
        with open(os.path.join(p, "raw_results.pickle4"), "rb") as f:
            raw_results = pickle.load(f)
            gaze_data = np.array(raw_results[0], int)
            all_gaze_data.append(gaze_data)
        with open(os.path.join(p, "res_foveations.csv"), "rb") as f:
            res_df = pd.read_csv(f)
            all_fov_tables.append(res_df)

    plot_dir = os.path.join(folder_paths[0], "videos")
    os.makedirs(plot_dir, exist_ok=True)

    frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data_all', video_name, 'images', f'{1:04d}.png')
    img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)
    shape = img.shape[:2]

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    orig_video_frames = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',]
    ns_fovs = [0 for _ in range(len(all_fov_tables))]
    current_trail_start = [0 for _ in range(len(all_fov_tables))]
    for i in tqdm(range(len(all_gaze_data[0]))):  # [-10:]
        frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data_all', video_name, 'images', f'{i:04d}.png')
        img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)
        imshow = plt.imshow(img, animated=True)
        all_coords = []
        for participant_i in range(len(all_gaze_data)):
            coords = plt.plot(all_gaze_data[participant_i][i][1], all_gaze_data[participant_i][i][0],
                              color=colors[participant_i], marker='x', markersize=markersize, animated=True)
            all_coords.extend(coords)
            coords_trail = plt.plot(all_gaze_data[participant_i][current_trail_start[participant_i]:i, 1],
                                    all_gaze_data[participant_i][current_trail_start[participant_i]:i, 0],
                                    color=colors[participant_i], linestyle=(0, (1, 1)), linewidth=3, animated=True)
            all_coords.extend(coords_trail)

            current_fov_end = all_fov_tables[participant_i]["frame_end"][ns_fovs[participant_i]]
            if current_fov_end < i:
                ns_fovs[participant_i] += 1
                current_trail_start[participant_i] = i
                coords_sacc = plt.plot(all_gaze_data[participant_i][i-1:i+1, 1],
                                        all_gaze_data[participant_i][i-1:i+1, 0],
                                        color=colors[participant_i], linewidth=4, animated=True)
                all_coords.extend(coords_sacc)
        orig_video_frames.append([imshow, *all_coords])
    make_video_from_frames(fig, orig_video_frames, os.path.join(plot_dir, "trails_combined_participants_vid.mp4"))
    plt.close(fig)


if __name__ == "__main__":
    base_path = "/media/vito/scanpath_backup/scanpath_results_current/2024-05-06-16-44-22_TEST_base_task0.0_entropy0.11_dv3.0_sig0.3"
    relevant_video = "2DNG46ZD9Ss"
    pathes = glob.glob(os.path.join(base_path, "*"))
    pathes.sort()
    pathes = [os.path.join(p, relevant_video) for p in pathes]
    make_combined_scanpathes_vid_with_trails(pathes)