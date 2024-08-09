import glob
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from collections import Counter

def angle_limits(angle):
    """
    Makes sure that a given angle is within the range of -180<angle<=180

    :param angle: Angle to be tested / converted
    :type angle: float
    :return: angle with -180 < angle <= 180
    :rtype: float
    """
    if -180 < angle <= 180:
        return angle
    elif angle > 180:
        return angle - 360
    else:
        return angle + 360


def calc_px2dva(image_size, display_size=(1080, 1920), display_w_dva=47.7, max_scaling=0.8):
    px2dva_unscaled = display_w_dva / display_size[1]
    if display_size[1] / display_size[0] > image_size[1] / image_size[0]:
        # display is wider than image -> limit by height
        movie_scale_factor = display_size[0] * max_scaling / image_size[0]
    else:
        # display is taller than image -> limit by width
        movie_scale_factor = display_size[1] * max_scaling / image_size[1]
    return px2dva_unscaled * movie_scale_factor


def load_objectmasks(videoname, data_path="/media/vito/TOSHIBA EXT/scanpath_data"):
    """
    Function that loads the object masks from a given path and returns them as a
    list of numpy arrays.

    :param videoname: Name of the Video
    :type videoname: str
    :param path: Path to the object masks
    :type path: str
    :return: List of object masks
    :rtype: list

    """
    objectmasks = []
    for mask in sorted(glob.glob(os.path.join(data_path, videoname, "mask", "*.png"))):
        # read in the mask with cv2 as grayscale and append it to the list
        objectmasks.append(cv2.imread(mask, cv2.IMREAD_GRAYSCALE))
    return objectmasks

def object_at_position(segmentationmap, xpos, ypos, radius=0):
    """
    Function that returns the currently gazed object with a tolerance (radius)
    around the gaze point. If the gaze point is on the background but there are
    objects within the radius, it is not considered to be background.

    :param segmentationmap: Object segmentation of the current frame
    :type segmentationmap: np.array
    :param xpos: Gaze position in x direction
    :type xpos: int
    :param ypos: Gaze position in y direction
    :type ypos: int
    :param radius: Tolerance radius in px, objects within that distance of the gaze point
        are considered to be foveated, defaults to 0
    :type radius: float, optional
    :return: Name of the object(s) at the given position / within the radius
    :rtype: str
    """
    (h, w) = segmentationmap.shape
    if radius == 0:
        objid = segmentationmap[ypos, xpos]
        if objid == 0:
            objname = "Ground"
        else:
            objname = f"Object {objid}"
        return objname
    # more interesting case: check in radius!
    else:
        center_objid = segmentationmap[ypos, xpos]
        if center_objid > 0:
            return f"Object {center_objid}"
        # check if all in rectangle is ground, then no need to draw a circle
        elif (
                np.sum(
                    segmentationmap[
                    max(0, int(ypos - radius)) : min(h - 1, int(ypos + radius)),
                    max(0, int(xpos - radius)) : min(w - 1, int(xpos + radius)),
                    ]
                )
                == 0
        ):
            return "Ground"
        # Do computationally more demanding check for a radius
        # store all objects other than `Ground` that lie within the radius
        else:
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - xpos) ** 2 + (Y - ypos) ** 2)
            mask = dist_from_center <= radius
            objects = np.unique(mask * segmentationmap)
            if len(objects) == 1 and 0 in objects:
                return "Ground"
            else:
                return ", ".join([f"Object {obj}" for obj in objects if (obj > 0)])


def evaluate_model_trial(res_file_path, trial_id=None, segmentation_masks=None, RADIUS_OBJ_GAZE_DVA=0.5, swap_xy=False, data_path="/media/vito/scanpath_backup/scanpath_data_all"):
    """
    Function that evaluates the results of one trial and returns a dataframe
    with the results.

    :param res_file_path: Path to the result file
    :type res_file_path: str
    :return: Dataframe with the results
    :rtype: pd.DataFrame
    """
    # first do all trials of one video (ideally use preloaded masks)
    videoname = res_file_path.split("/")[-2]
    if segmentation_masks is None:
        segmentation_masks = load_objectmasks(videoname, data_path=data_path)
    if trial_id is None:
        runname = res_file_path.split("/")[-3].split("_")[-1]
    else:
        runname = trial_id
    # get the pixel radius for the object tolerance
    # if RADIUS_OBJ_GAZE_DVA is not None:
    px_obj_radius = RADIUS_OBJ_GAZE_DVA / calc_px2dva(segmentation_masks[0].shape)

    # load the result file
    with open(res_file_path, "rb") as f:
        raw_results = pickle.load(f)
        df = raw_results[1]
        gaze_data = np.array(raw_results[0], int)

    # swap_xy is FALSE for new runs (bug is fixed in scanpath_producer)
    if swap_xy:
        df.rename(columns={"x_start":"y_start", "y_start":"x_start", "x_end":"y_end", "y_end":"x_end"}, inplace=True)
    # df = pd.read_csv(res_file[1], index_col=0)
    df = calculate_fov_df_info(df, gaze_data, px_obj_radius, runname, segmentation_masks, videoname)
    return df


def calculate_fov_df_info(df, gaze_data, px_obj_radius, runname, segmentation_masks, videoname):
    N_fov = len(df)
    gt_object_list = [
        Counter(
            ", ".join(
                [
                    # get all foveated objects in this foveation
                    # with tolerance of px_obj_radius (as for the human eye tracking data)
                    object_at_position(
                        segmentation_masks[f_i],
                        gaze_data[f_i][1],
                        gaze_data[f_i][0],
                        radius=px_obj_radius,
                    )
                    if (gaze_data[f_i] > 0).all() else ''  # (f_i < len(segmentation_masks)) and
                    for f_i in range(
                        df["frame_start"].iloc[n],
                        min(len(segmentation_masks), df["frame_end"].iloc[n] + 1),
                    )
                ]
            ).split(", ")
        ).most_common(1)[0][0]
        for n in range(N_fov)
    ]
    gt_object_list = gt_object_list + ["" for _ in range(N_fov - len(gt_object_list))]
    df["gt_object"] = gt_object_list
    # calculate a number of saccade properties based on the gaze shift
    # depending on the end of the current fov and beginning of next one
    if not {"sac_ang_h", "sac_ang_p"}.issubset(df.columns):
        diff = np.array(
            [
                np.array((df["x_start"][i + 1], df["y_start"][i + 1])) - np.array((df["x_end"][i], df["y_end"][i]))
                for i in range(N_fov - 1)
            ]
        )
        # avoid error if no saccades are made
        if diff.size:
            df["sac_ang_h"] = list(
                - np.arctan2(diff[:, 1], diff[:, 0]) / np.pi * 180
            ) + [np.nan]
            # second entry of angle_p will also be nan since first angle_h is nan
            df["sac_ang_p"] = [np.nan] + [
                angle_limits(df["sac_ang_h"][i + 1] - df["sac_ang_h"][i])
                for i in range(N_fov - 1)
            ]
        else:
            df["sac_ang_h"] = [np.nan]
            df["sac_ang_p"] = [np.nan]
    # add start and end time of each foveation
    if not {"fov_start", "fov_end"}.issubset(df.columns):
        df["fov_end"] = (df["duration_ms"] + df["sac_dur"]).cumsum() - df["sac_dur"]
        df["fov_start"] = df["fov_end"] - df["duration_ms"] - df["sac_dur"]
    # calculate the foveation categories (Background, Detection, Inspection, Revisit)
    fov_categories = []
    ret_times = np.zeros(N_fov) * np.nan
    for n in range(N_fov):
        obj = df["gt_object"].iloc[n]
        if obj == "":
            fov_categories.append("-")
        elif obj == "Ground":
            fov_categories.append("B")
        elif (n > 0) and (df["gt_object"].iloc[n - 1] == obj):
            fov_categories.append("I")
        else:
            prev_obj = df["gt_object"].iloc[:n]
            if obj not in prev_obj.values:
                fov_categories.append("D")
            else:
                fov_categories.append("R")
                return_prev_t = df["fov_end"][
                    prev_obj.where(prev_obj == obj).last_valid_index()
                ]
                # store time difference [in milliseconds] in array!
                ret_times[n] = (
                    (df["fov_start"].iloc[n] - return_prev_t)
                )
    df["fov_category"] = fov_categories
    df["ret_times"] = ret_times
    df.insert(1, "video", videoname)
    df.insert(2, "subject", runname)
    return df

