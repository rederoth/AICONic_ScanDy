import glob
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from collections import Counter

import scanpath_utils as su
import summary_plots as sp


def transform_gaze2px(tracker_val, video_min, video_max, target_min, target_max):
    try:
        num = float(tracker_val)
    except ValueError:
        return np.NaN
    # check if value exceeds boundaries of the video
    if (video_max <= num) or (num < video_min) or np.isnan(num):
        return np.NaN
    else:
        return int(
            (num - video_min + target_min)
            * target_max
            / (video_max - video_min + target_min)
        )


if __name__ == "__main__":
    # evaluate groundtruth data
    flag = "_all" #"_all"  # "" takes only the training set (10), _all takes 43 videos (including the training set of 10)!
    run_name = "Humans (all)"
    GROUND_TRUTH_PATH = "/home/vito/Documents/eye_data_EM-UnEye_2023-11-29/"
    GT_STORE_PATH = GROUND_TRUTH_PATH  # "/home/vito/Documents/eye_data_uneye_2023-10-15_processed/"
    VIDEO_PATH = f"/media/vito/scanpath_backup/scanpath_data{flag}/"
    RADIUS_OBJ_GAZE_DVA = 0.5

    data_paths = sorted(glob.glob(os.path.join(GROUND_TRUTH_PATH, "*_data.csv.gz")))
    eye_paths = sorted(glob.glob(os.path.join(GROUND_TRUTH_PATH, "*_eye.csv.gz")))
    subj_ids = [f.split("_")[-2] for f in data_paths]

    # df_all_gaze = pd.DataFrame()
    df_all_fovs = pd.DataFrame()

    for s_id, subject_id in enumerate(subj_ids):
        print(f"Subject {subject_id}")
        df_data = pd.read_csv(
            data_paths[s_id], compression="gzip", encoding="iso-8859-1"
        )
        df_eye = pd.read_csv(eye_paths[s_id], compression="gzip", encoding="iso-8859-1")
        # combine FIX, SP and PSO events to FOV
        df_eye["final_labels"].replace(
            ["FIX", "SP", "PSO"], ["FOV", "FOV", "FOV"], inplace=True
        )
        # replace all nan with "BLINK"
        df_eye["final_labels"].fillna("BLINK", inplace=True)
        # remove all rows before and after the video actually plays
        df_eye.dropna(subset=["frame"], inplace=True)
        df_eye = df_eye[df_eye["frame"] > 0]
        # get dominant eye
        subj_dominant_eye = df_data.subj_dominant_eye.iloc[0]
        if subj_dominant_eye == 0:
            dom = "l"
        elif subj_dominant_eye == 1:
            dom = "r"
        else:
            raise ValueError("Dominant eye not 0 or 1")
        df_eye = df_eye.rename(columns={f"x{dom}": "x", f"y{dom}": "y"})

        # go through all trials, i.e. videos
        for i in range(len(df_data)):
            trial_id = df_data["ID"].iloc[i]
            videoname = df_data["filename"].iloc[i].split(".")[0]
            # only use videos where masks are available
            if videoname in os.listdir(VIDEO_PATH):
                print(f"Video {videoname} found, lets go...")
            else:
                continue
            # this should not happen, but check and investigate if it does
            if trial_id not in df_eye["ID"].unique():
                print(f"Trial {trial_id} not found in eye data.")
                continue
            df_trial = df_eye[df_eye["ID"] == trial_id]

            vid_pos_x = df_data["movie_pos_x"].iloc[i]
            vid_s_x = df_data["movie_size_x"].iloc[i]
            vid_res_x = df_data["movie_res_x"].iloc[i]
            vid_pos_y = df_data["movie_pos_y"].iloc[i]
            vid_s_y = df_data["movie_size_y"].iloc[i]
            vid_res_y = df_data["movie_res_y"].iloc[i]

            px2dva = su.calc_px2dva((vid_s_y, vid_s_x))

            # eval df_gaze analogous to simulated: one pos per frame!
            # .agg automatically excludes NaNs!
            df_gaze = (
                df_trial.groupby("frame")
                .agg(
                    {
                        "x": "mean",
                        "y": "mean",
                    }  # , "BLSTM EV": lambda x: Counter(x).most_common(1)[0][0]}
                )
                .reset_index()
            )
            # transform gaze data to pixel coordinates
            df_gaze["x"] = df_gaze["x"].apply(
                lambda x: transform_gaze2px(
                    x, vid_pos_x, vid_pos_x + vid_s_x, 0, vid_res_x
                )
            )
            df_gaze["y"] = df_gaze["y"].apply(
                lambda y: transform_gaze2px(
                    y, vid_pos_y, vid_pos_y + vid_s_y, 0, vid_res_y
                )
            )
            # create the equivalent of gaze_data in the simulated data
            # apparently, x & y are confused in the simulations... TODO check angles!
            gaze_data = np.array([df_gaze.y, df_gaze.x]).T.astype(int)
            # df_gaze["frame"] = df_gaze["frame"].astype(int)
            # df_gaze["trial_id"] = trial_id
            # df_gaze["video"] = videoname
            # df_gaze["subject"] = int(subject_id)
            # # not all masks available here, therefore
            # # df_gaze["object"] = df_gaze.apply(...)
            # df_all_gaze = pd.concat([df_all_gaze, df_gaze])

            # group the data to foveation events
            group = df_trial.groupby(
                df_trial["final_labels"].ne(df_trial["final_labels"].shift()).cumsum()
            )
            df_fov = group.apply(
                lambda entry: pd.DataFrame(
                    {
                        "duration_ms": [int(entry["t"].max() - entry["t"].min())],
                        "event": [entry["final_labels"].iloc[0]],
                        "x_start": [
                            transform_gaze2px(
                                entry["x"].iloc[0],
                                vid_pos_x,
                                vid_pos_x + vid_s_x,
                                0,
                                vid_res_x,
                            )
                        ],
                        "x_end": [
                            transform_gaze2px(
                                entry["x"].iloc[-1],
                                vid_pos_x,
                                vid_pos_x + vid_s_x,
                                0,
                                vid_res_x,
                            )
                        ],
                        "y_start": [
                            transform_gaze2px(
                                entry["y"].iloc[0],
                                vid_pos_y,
                                vid_pos_y + vid_s_y,
                                0,
                                vid_res_y,
                            )
                        ],
                        "y_end": [
                            transform_gaze2px(
                                entry["y"].iloc[-1],
                                vid_pos_y,
                                vid_pos_y + vid_s_y,
                                0,
                                vid_res_y,
                            )
                        ],
                        "frame_start": [int(entry["frame"].iloc[0] - 1)],
                        "frame_end": [int(entry["frame"].iloc[-1] - 1)],
                    }
                )
            ).reset_index(drop=True)
            fov_ends = df_fov["duration_ms"].cumsum()
            df_fov["fov_start"] = fov_ends - df_fov["duration_ms"]
            df_fov["fov_end"] = fov_ends
            # foveation must be longer than 30ms (would still be less than duration of a single frame!)
            df_fov = df_fov[(df_fov.event == "FOV") & (df_fov.duration_ms > 30)]
            df_fov.drop(columns=["event"], inplace=True)
            df_fov.reset_index(inplace=True)
            # new - old as in gaze shift of simulated data
            gaze_shifts_x = np.array(np.array(df_fov["x_start"].iloc[1:]) - np.array(df_fov["x_end"].iloc[:-1]))
            gaze_shifts_y = np.array(np.array(df_fov["y_start"].iloc[1:]) - np.array(df_fov["y_end"].iloc[:-1]))
            # old: diff_y = np.array(df_fov['y_end'][:-1]) - np.array(df_fov['y_start'][1:])
            df_fov["sac_amp_dva"] = list(
                np.sqrt(gaze_shifts_x**2 + gaze_shifts_y**2) * px2dva
            ) + [np.nan]
            # df_fov["sac_angle_h"] = list(
            #     -np.arctan2(gaze_shifts_x, gaze_shifts_y) / np.pi * 180
            # ) + [np.nan]
            # df_fov["sac_angle_p"] = [np.nan] + [
            #     su.angle_limits(
            #         df_fov["sac_angle_h"].iloc[i + 1] - df_fov["sac_angle_h"].iloc[i]
            #     )
            #     for i in range(len(df_fov) - 1)
            # ]
            # df_fov.insert(0, "subject", int(subject_id))
            # df_fov.insert(1, "video", videoname)
            df_fov.insert(0, "trial_id", trial_id)

            segmentation_masks = su.load_objectmasks(videoname, VIDEO_PATH)
            assert len(segmentation_masks) > 0, "no object masks have been found!"
            px_obj_radius = RADIUS_OBJ_GAZE_DVA / px2dva 
            # su.calc_px2dva(segmentation_masks[0].shape)

            # run the same analysis as for simulated data
            df_fov = su.calculate_fov_df_info(
                df_fov,
                gaze_data,
                px_obj_radius=px_obj_radius,
                runname=subject_id,
                segmentation_masks=segmentation_masks,
                videoname=videoname,
            )

            df_all_fovs = pd.concat([df_all_fovs, df_fov])

    df_all_fovs.to_csv(os.path.join(GT_STORE_PATH, f'df_res_gt_fov{flag}.csv'))
    # PLOTTING
    sp.plot_fov_dur_sac_amp_hists(df_all_fovs, savedir=GT_STORE_PATH, name_flag=flag, custom_name=run_name)
    sp.plot_sac_ang_hists(df_all_fovs, savedir=GT_STORE_PATH, name_flag=flag)
    sp.plot_ior_stats(df_all_fovs, savedir=GT_STORE_PATH, name_flag=flag, custom_name=run_name)
    sp.plot_object_eval(df_all_fovs, savedir=GT_STORE_PATH, name_flag=flag)
