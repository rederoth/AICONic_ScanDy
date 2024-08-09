import glob
import os.path
import pickle
import random
import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 as cv
import yaml
import pandas as pd
from tqdm import tqdm



def generate_colormap(N_colors=200):
    HSV_tuples = [(x * 1.0 / N_colors, 0.7, 0.6) for x in range(N_colors)]
    rgb_out = []
    for rgb in HSV_tuples:
        rgb_out.append(colorsys.hsv_to_rgb(*rgb))
    random.shuffle(rgb_out)
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_colors = np.array([(0.0, 0.0, 0.0)]+rgb_out)
    N_colors = len(rgb_colors)
    # rgb_colors = np.random.rand(N, 3)
    return matplotlib.colors.ListedColormap(rgb_colors, N=N_colors)


def generate_colormap2(N_colors=200):
    HSV_tuples = [(x * 1.0 / N_colors, 0.7, 0.6) for x in range(N_colors)]
    rgb_out = []
    for rgb in HSV_tuples:
        rgb_out.append(colorsys.hsv_to_rgb(*rgb))
    random.shuffle(rgb_out)
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_out.extend([np.array(x) * 0.99 for x in rgb_out])
    rgb_colors = np.array([(0.0, 0.0, 0.0)]+rgb_out)
    N_colors = len(rgb_colors)
    # rgb_colors = np.random.rand(N, 3)
    return matplotlib.colors.ListedColormap(rgb_colors, N=N_colors), N_colors


random.seed(42)
N_colors = 200
custom_cmap, N_colors = generate_colormap2(N_colors)



def create_video_files(folder_path):
    markersize = 15
    # store resulting df as csv
    video_name = folder_path.split('/')[-1]
    res_df = generate_res_df(folder_path)
    # read in the config file
    df_conf = read_config_to_df(folder_path)
    if 'sac_momentum_max' in df_conf.columns:
        sensitivity_max = df_conf["sac_momentum_max"].iloc[0]
    else:
        sensitivity_max = 1.

    # make plots for every frame
    pathes = glob.glob(os.path.join(folder_path, "*.pickle"))
    pathes.sort()

    if len(pathes) == 0:
        return

    plot_dir = os.path.join(folder_path, "videos")
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
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]
        # print(f_idx, fp)
        frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data_all', video_name, 'images', f'{f_idx:04d}.png')
        img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)

        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)

            # plot a large red cross on the maximum of data["gaze_img"]
            im1 = plt.imshow(img, animated=True)
            im2 = plt.plot(max_coords[1], max_coords[0], 'rx', markersize=markersize, animated=True)
            orig_video_frames.append([im1, *im2])
    make_video_from_frames(fig, orig_video_frames, os.path.join(plot_dir, "source_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    task_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)

            task_contrib = data["task_imp_img"]

            task_video_frames.append([plt.imshow(task_contrib, cmap="inferno", vmin=0, vmax=1, animated=True)])
    make_video_from_frames(fig, task_video_frames, os.path.join(plot_dir, "task_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    uncertainty_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)

            uncert_contrib = data["obj_cert_img"]
            uncertainty_video_frames.append(
                [plt.imshow(uncert_contrib, vmin=0, vmax=1, interpolation='none', animated=True)])
    make_video_from_frames(fig, uncertainty_video_frames, os.path.join(plot_dir, "uncertainty_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    sensitivity_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]


        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)

            sens_contrib = data["sensitivity_img"].cpu().numpy()
            im1 = plt.imshow(sens_contrib, cmap="bone", vmin=0, vmax=sensitivity_max, animated=True)
            im2 = plt.plot(max_coords[1], max_coords[0], "rx", markersize=markersize, animated=True)
            sensitivity_video_frames.append([im1, *im2])
    make_video_from_frames(fig, sensitivity_video_frames, os.path.join(plot_dir, "sensitivity_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    evidence_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)
            sens_contrib = data["sensitivity_img"].cpu().numpy()
            uncert_contrib = data["obj_cert_img"]
            task_contrib = data["task_imp_img"]

            evidence_img = sens_contrib * task_contrib * uncert_contrib
            im1 = plt.imshow(evidence_img, cmap="magma", vmin=0, animated=True)
            im2 = plt.plot(max_coords[1], max_coords[0], "rx", markersize=markersize, animated=True)
            evidence_video_frames.append([im1, *im2])
    make_video_from_frames(fig, evidence_video_frames, os.path.join(plot_dir, "evidence_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    flow_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]
        with open(fp, "rb") as f:
            data = pickle.load(f)

            flow_video_frames.append([plt.imshow(data["flow_visualized"], vmin=0, vmax=1, animated=True)])
    make_video_from_frames(fig, flow_video_frames, os.path.join(plot_dir, "flow_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    segmentation_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        if df_conf["use_ground_truth_objects"].iloc[0] == "true":
            gt_objects = cv.cvtColor(cv.imread(
                os.path.join('/media/vito/TOSHIBA EXT/scanpath_data', video_name, 'mask', f'{f_idx:04d}.png')),
                cv.COLOR_BGR2GRAY)

        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)

            if df_conf["use_ground_truth_objects"].iloc[0] == "true":
                im1 = plt.imshow(gt_objects.astype(np.uint32), cmap=custom_cmap, vmin=0, vmax=N_colors,
                                 interpolation='none', animated=True)
            else:
                im1 = plt.imshow(data["obj_seg_img"].astype(np.uint32), cmap=custom_cmap, vmin=0, vmax=N_colors,
                                 interpolation='none', animated=True)
            im2 = plt.plot(max_coords[1], max_coords[0], 'rx', markersize=markersize, animated=True)
            segmentation_video_frames.append([im1, *im2])

    make_video_from_frames(fig, segmentation_video_frames, os.path.join(plot_dir, "segmentation_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    appearance_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)

            appearance_video_frames.append(
                [plt.imshow(data["obj_seg_measurements"][0].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=N_colors,
                            interpolation='none', animated=True)])  # appearance


    make_video_from_frames(fig, appearance_video_frames, os.path.join(plot_dir, "appearance_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    motion_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]
        with open(fp, "rb") as f:
            data = pickle.load(f)

            motion_video_frames.append(
                [plt.imshow(data["obj_seg_measurements"][1].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=N_colors,
                            interpolation='none', animated=True)])  # motion
    make_video_from_frames(fig, motion_video_frames, os.path.join(plot_dir, "motion_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    global_semantic_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)
            global_semantic_video_frames.append(
                [plt.imshow(np.round(data["obj_seg_measurements"][2].cpu().numpy()).astype(np.uint8), cmap=custom_cmap,
                            vmin=0, vmax=N_colors, interpolation='none', animated=True)])  # global sematic


    make_video_from_frames(fig, global_semantic_video_frames, os.path.join(plot_dir, "global_semantic_vid.mp4"))
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    foveated_video_frames = []
    for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]

        with open(fp, "rb") as f:
            data = pickle.load(f)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()),
                                          data["gaze_img"].cpu().numpy().shape)

            im1 = plt.imshow(
                np.round(data["obj_seg_measurements"][3].cpu().numpy()).astype(bool).astype(np.uint8) * 12 + np.round(
                    data["obj_seg_measurements"][4].cpu().numpy()).astype(bool).astype(np.uint8) * 2, cmap=custom_cmap,
                vmin=0, vmax=N_colors, interpolation='none', animated=True)
            im2 = plt.plot(max_coords[1], max_coords[0], 'rx', markersize=markersize,
                           animated=True)  # foveated sematics
            foveated_video_frames.append([im1, *im2])

    make_video_from_frames(fig, foveated_video_frames, os.path.join(plot_dir, "foveated_vid.mp4"))
    plt.close(fig)

    with open(pathes[0], "rb") as f:
        data = pickle.load(f)
        N_particles = len(data["particle_segmentations"])

    for i in range(N_particles):
        fig = plt.figure(frameon=False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        fig.set_size_inches(shape[1] / 100.0, shape[0] / 100.0)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        particle_video_frames = []
        for f_idx, fp in tqdm(enumerate(pathes)):  # [-10:]
            with open(fp, "rb") as f:
                data = pickle.load(f)

                particle_seg = data["particle_segmentations"][i].astype(np.uint32)
                particle_video_frames.append([plt.imshow(particle_seg.astype(np.uint32), cmap=custom_cmap, vmin=0, vmax=N_colors,
                                 interpolation='none', animated=True)])
        make_video_from_frames(fig, particle_video_frames, os.path.join(plot_dir, "particle_"+str(i)+"_vid.mp4"))
        plt.close(fig)


def make_video_from_frames(fig, frames, path):
    ani = animation.ArtistAnimation(fig, frames, interval=66, blit=True,
                              repeat=False)
    ani.save(path)
    print("Done Saving "+str(path))


def process_experiment_data(folder_path, dpi=150):
    markersize = 15
    # store resulting df as csv
    video_name = folder_path.split('/')[-1]
    res_df = generate_res_df(folder_path)
    # read in the config file
    df_conf = read_config_to_df(folder_path)
    if 'sac_momentum_max' in df_conf.columns:
        sensitivity_max = df_conf["sac_momentum_max"].iloc[0]
    else:
        sensitivity_max = 1.

    # make plots for every frame
    pathes = glob.glob(os.path.join(folder_path, "*.pickle"))
    pathes.sort()
    plot_dir = os.path.join(folder_path, "plots_hq")
    os.makedirs(plot_dir, exist_ok=True)
    for f_idx, fp in tqdm(enumerate(pathes)): #[-10:]
        # print(f_idx, fp)
        frame_path = os.path.join('/media/vito/TOSHIBA EXT/scanpath_data', video_name, 'images', f'{f_idx:04d}.png')
        img = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2RGB)
        if df_conf["use_ground_truth_objects"].iloc[0] == "true":
            gt_objects = cv.cvtColor(cv.imread(os.path.join('/media/vito/TOSHIBA EXT/scanpath_data', video_name, 'mask', f'{f_idx:04d}.png')), cv.COLOR_BGR2GRAY)

        # if (f_idx == 0) & (df_conf["center_bias"].iloc[0] == "true"):
        #     cb = anisotropic_centerbias_np(img.shape[1], img.shape[0])

        with open(fp, "rb") as f:
            data = pickle.load(f)
            fig, axs, = plt.subplots(3, 4, figsize=(10,6.5), dpi=dpi)
            max_coords = np.unravel_index(np.argmax(data["gaze_img"].cpu().numpy()), data["gaze_img"].cpu().numpy().shape)
            task_contrib = data["task_imp_img"]
            if df_conf["center_bias"].iloc[0] == "true":
                # done now directly in the task contrib!
                # task_contrib = cb * task_contrib
                task_title = "Task contrib. (saliency) * CB"
            else:
                task_title = "Task contrib. (saliency)"
            axs[0, 0].imshow(task_contrib, cmap="inferno", vmin=0, vmax=1)
            axs[0, 0].set_title(task_title)
            # # TODO do this directly in the task contrib!
            # if df_conf["use_uncertainty_in_gaze_evidences"].iloc[0] == "true":
            #     uncert_contrib = data["obj_cert_img"]
            # else:
            #     uncert_contrib = np.ones_like(task_contrib)
            uncert_contrib = data["obj_cert_img"]
            axs[0, 1].imshow(uncert_contrib, vmin=0, vmax=1, interpolation='none')
            axs[0, 1].set_title("Uncertainty contrib.")
            sens_contrib = data["sensitivity_img"].cpu().numpy()
            axs[0, 2].imshow(sens_contrib, cmap="bone", vmin=0, vmax=sensitivity_max)
            axs[0, 2].plot(max_coords[1], max_coords[0], "rx", markersize=markersize)
            axs[0, 2].set_title("Sensitivity contrib.")

            # evidence_img = np.zeros_like(data["obj_seg_img"])
            # for o_idx, obj_id in enumerate(data["obj_seg_img"]):
            #     mask = data["obj_seg_img"] == obj_id
            #     tmp = mask * sens_contrib * task_contrib * uncert_contrib
            evidence_img = sens_contrib * task_contrib * uncert_contrib
            axs[0, 3].imshow(evidence_img, cmap="magma", vmin=0)
            axs[0, 3].plot(max_coords[1], max_coords[0], "rx", markersize=markersize)
            axs[0, 3].set_title("Evidence update")

            # plot a large red cross on the maximum of data["gaze_img"]
            axs[1, 0].imshow(img)
            axs[1, 0].plot(max_coords[1], max_coords[0], 'rx', markersize=markersize)
            axs[1, 0].set_title(f"Frame {f_idx:03d}")
            axs[1, 1].imshow(data["flow_visualized"], vmin=0, vmax=1)
            axs[1, 1].set_title("Optical flow")
            axs[1, 2].imshow(data["particle_entropy"], vmin=0, vmax=1)
            axs[1, 2].set_title("Particle entropy")#
            if df_conf["use_ground_truth_objects"].iloc[0] == "true":
                axs[1, 3].imshow(gt_objects.astype(np.uint32), cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
                axs[1, 3].plot(max_coords[1], max_coords[0], 'rx', markersize=markersize)
                axs[1, 3].set_title("Ground truth objects")
            else:
                print(np.max(data["obj_seg_img"].astype(np.uint32)))
                print(rgb_colors[np.max(data["obj_seg_img"].astype(np.uint32))])
                axs[1, 3].imshow(data["obj_seg_img"].astype(np.uint32), cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
                axs[1, 3].plot(max_coords[1], max_coords[0], 'rx', markersize=markersize)
                axs[1, 3].set_title("Resulting objects")

            axs[2, 0].imshow(data["obj_seg_measurements"][0].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
            axs[2, 0].set_title("Appearance-based")
            axs[2, 1].imshow(data["obj_seg_measurements"][1].cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
            axs[2, 1].set_title("Motion-based")
            axs[2, 2].imshow(np.round(data["obj_seg_measurements"][2].cpu().numpy()).astype(np.uint8), cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
            axs[2, 2].set_title("Global SAM")
            axs[2, 3].imshow(np.round(data["obj_seg_measurements"][3].cpu().numpy()).astype(bool).astype(np.uint8) * 12 + np.round(data["obj_seg_measurements"][4].cpu().numpy()).astype(bool).astype(np.uint8) * 2, cmap=custom_cmap, vmin=0, vmax=N_colors, interpolation='none')
            axs[2, 3].plot(max_coords[1], max_coords[0], 'rx', markersize=markersize)
            axs[2, 3].set_title("Prompted SAM")

            for ax in axs.flat:
                ax.set_axis_off()

            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(plot_dir, fp[-11:-7]))
            # print(f'saved {os.path.join(plot_dir, fp[-11:-7])}')
            plt.close(fig)


def read_config_to_df(folder_path):
    config_path = os.path.join(folder_path, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)
    df_conf = pd.json_normalize(config)
    return df_conf

def generate_res_df(folder_path, store_res_df=True):
    res_path = os.path.join(folder_path, 'raw_results.pickle4')
    if os.path.isfile(res_path):
        seed = res_path.split("/")[-3].split("_")[-1]
        video = res_path.split("/")[-2]
        with open(res_path, "rb") as f:
            raw_results = pickle.load(f)
            res_df = raw_results[1]
            res_df["seed"] = seed
            res_df["video"] = video
            if store_res_df:
                res_df.to_csv(os.path.join(folder_path, f'res_foveations.csv'))
        return res_df
    else:
        print(f"No raw_results.pickle4 in {folder_path}")


if __name__ == "__main__":
    # process_experiment_data("/media/vito/scanpath_backup/scanpath_results_current/2023-11-15-17-44-38/config_0/-sGcmYcU_QI")
    # process_experiment_data("/media/vito/scanpath_backup/scanpath_results_current/2023-11-15-17-44-38/config_0/24cgfaG8WI0")

    base_path = "/media/vito/scanpath_backup/scanpath_results_current/2024-05-06-16-44-22_TEST_base_task0.0_entropy0.11_dv3.0_sig0.3/config_"
    for i in range(10):
        experiments = glob.glob(os.path.join(base_path+str(i), "*"))
        for p in experiments:
            # process_experiment_data(p)
            create_video_files(p)

    # pathes = glob.iglob("/media/vito/TOSHIBA EXT/scanpath_results/2023-10-26-18-48-22/*")
    # todo_sequences = []
    # for p in pathes:
    #     single_sequence_pathes = glob.iglob(os.path.join(p, "*", "*"))
    #     for sp in single_sequence_pathes:
    #         if not os.path.isdir(sp):
    #             continue
    #         if os.path.isdir(os.path.join(sp, "plots_hq")):
    #             continue
    #         todo_sequences.append(sp)
    # todo_sequences = sorted(todo_sequences)
    # todo_sequences.reverse()
    # for p in tqdm(todo_sequences):
    #     print(p)
    #     process_experiment_data(p)
