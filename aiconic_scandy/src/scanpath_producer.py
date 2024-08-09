from ObjectSegmentationFilter import ObjectSegmentationFilter, ObjectSegmentationState
from TaskRelevantFilter import TaskRelevantState, TaskRelevantFilter
from GazeFilter import GazeFilter, GazeState
import torch
import cv2 as cv
import numpy as np
import matplotlib
import pandas as pd
import scanpath_utils as su

# matplotlib.use("tkagg")
import flow_vis
import colorsys
import random

use_torch_compile = False  # actually makes it overall a bit slower right now, only interesting if we do some backward passes
torch_compile_backend = "inductor"

N = 100
HSV_tuples = [(x * 1.0 / N, 0.7, 0.6) for x in range(N)]
rgb_out = []
for rgb in HSV_tuples:
    rgb_out.append(colorsys.hsv_to_rgb(*rgb))
random.shuffle(rgb_out)
rgb_colors = np.array([(0.0, 0.0, 0.0)] + rgb_out)
custom_cmap = matplotlib.colors.ListedColormap(rgb_colors, N=N)


class GlobalState:
    def __init__(self, save_particles: bool = False):
        self.gaze = GazeState()
        self.object_segmentation = ObjectSegmentationState(save_particles=save_particles)
        self.task_importance = TaskRelevantState()

        # additions to visualization
        self.task_imp_contrib_vis = None
        self.sensitivity_contrib_vis = None
        self.uncertainty_contrib_vis = None

        # used only for old visualization?!:
        # self.fig = plt.figure()
        # self.axs = self.fig.subplots(3, 2)

    def init_state(self, gaze_loc, object_seg, init_particles, task_importance_map, sensitivity_map):
        self.gaze.init_state(
            gaze_loc, torch.eye(2), sensitivity_map
        )
        self.object_segmentation.init_state(object_seg, init_particles)
        self.task_importance.init_state(task_importance_map)

    @torch.compile(disable=not use_torch_compile, backend=torch_compile_backend)
    def visualize(self, image_size, gaze_history, flow_meas):
        G, sensitivity = self.gaze.create_visualization(image_size)
        (
            obj_seg,
            obj_cert,
            particle_view,
            entropy,
            particles_matched
        ) = self.object_segmentation.create_visualization(image_size)
        # task_imp = self.task_importance.create_visualization(image_size)
        return (
            G,
            self.sensitivity_contrib_vis,  # sensitivity,
            gaze_history,
            self.task_imp_contrib_vis.cpu().numpy(),  # task_imp,
            obj_seg,
            flow_vis.flow_to_color(flow_meas),
            self.uncertainty_contrib_vis.cpu().numpy(),  # obj_cert,
            entropy,
            particles_matched
        )


class ScanPathProducer:
    def __init__(self, config, starting_pos):
        self.gaze_filter = GazeFilter(config)
        self.obj_seg_filter = ObjectSegmentationFilter(
            config.object_segmentation_params
        )
        self.task_filter = TaskRelevantFilter(config)
        self.evidence_dict = {}
        self.inhibition_dict = {}
        self.obj_inhibition = config.obj_inhibition
        self.inhibition_r = config.inhibition_r
        self.decision_threshold = config.decision_threshold
        self.decision_noise = config.decision_noise
        self.drift_noise_dva = config.drift_noise_dva
        self.entropy_added_number = config.entropy_added_number
        self.presaccadic_threshold = config.presaccadic_threshold
        self.task_importance_added_number = config.task_importance_added_number
        self.make_sac_in_obj = config.make_sac_in_obj
        self.use_uncertainty_in_gaze_evidences = config.use_uncertainty_in_gaze_evidences
        self.use_IOR_in_gaze_evidences = config.use_IOR_in_gaze_evidences
        self.use_center_bias = config.center_bias
        self.use_ground_truth_objects = config.use_ground_truth_objects
        # self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=3.0)

        # will be set in preaction (run_experiment.py) or updated during the simulation
        self.center_bias = None
        self.px2dva = 0.0
        self.frames2ms = 1000.0 / 30.0
        self.ongoing_sacdur = (
            0.0  # tracks how long evidence accumulation should be suppressed
        )
        self.already_waited_frames = (
            0.0  # tracks #frames waited during the current saccade
        )
        self.foveation_info = (
            []
        )  # this will be returned as a df at the end of the trial
        self.nfov = 0
        self.fov_start_t = 0
        self.fov_start_loc = starting_pos.cpu().numpy()
        # self.prev_sacdur = 0.0
        # self.prev_sacamp = np.nan

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

    @torch.compile(disable=not use_torch_compile, backend=torch_compile_backend)
    def post_action_inference(
            self, action, state, gaze_meas, obj_meas, of_meas, task_meas, timestamp
    ):
        presac_obj = obj_meas[4]
        if len(self.foveation_info) > 0:
            last_foveation_info = self.foveation_info[-1]
        else:
            last_foveation_info = [np.nan for i in range(11)]
        if self.use_ground_truth_objects:
            segmentation = state.object_segmentation.ground_truth_objects
        else:
            segmentation = state.object_segmentation.object_segmentation
        self.gaze_filter.correct(
            state.gaze,
            action,
            gaze_meas,
            timestamp,
            segmentation,
            presac_obj,
            last_foveation_info
        )
        # PRESAC Hot fix, TODO make nicer! effectively this takes the prompted object for sensitivity but not for the particle filter
        modified_obj_meas = obj_meas
        modified_obj_meas[4] = torch.zeros_like(presac_obj)
        self.obj_seg_filter.correct(
            state.object_segmentation, state.gaze, modified_obj_meas, of_meas, timestamp
        )
        self.task_filter.correct(
            state.task_importance, state.gaze, task_meas, timestamp
        )

    def _calc_sac_dur(self, dist_dva):
        """
        Calculate the saccade duration. We use literature values from
        Collewijn, H., Erkelens, C. J., & Steinman, R. M. (1988). Binocular co-ordination of human horizontal saccadic eye movements.
        :param dist_dva: saccade amplitude
        :type dist_dva: float
        """
        sacdur_ms = 2.7 * dist_dva + 23.0
        return (
                sacdur_ms / self.frames2ms
        )  # convert ms to frames (30fps, as in all UVO videos)

    def _update_inhibition_dict(self, state, obj_id):
        obj_id_str = str(obj_id.item())
        if obj_id_str in self.inhibition_dict:
            if obj_id_str == str(state.gaze.gaze_object):
                # special treatment of the background? ==> xi = 0
                self.inhibition_dict[obj_id_str] = self.obj_inhibition
            else:
                # reduce inhibition by inhibition_r but make sure it's not negative
                self.inhibition_dict[obj_id_str] -= self.inhibition_r
                self.inhibition_dict[obj_id_str] = max(
                    self.inhibition_dict[obj_id_str], 0
                )
        else:
            # if it is not in the dictionary, add it
            self.inhibition_dict[obj_id_str] = 0  # gets added to current gaze position

    def _update_evidence_dict(self, state, obj_id, task_imp_contrib, sensitivity_contrib, uncertainty_contrib,
                              cur_fov_frac):
        obj_id_str = str(obj_id.item())
        if self.use_ground_truth_objects:
            obj_mask = state.object_segmentation.ground_truth_objects == obj_id
        else:
            obj_mask = state.object_segmentation.object_segmentation == obj_id
        kernel = np.ones((3, 3), np.uint8)
        obj_mask = (
            torch.from_numpy(
                cv.morphologyEx(
                    obj_mask.cpu().numpy().astype(np.uint8), cv.MORPH_DILATE, kernel
                )
            )
            .to(self.device)
            .type(torch.get_default_dtype())
        )
        obj_mask = obj_mask.type(torch.get_default_dtype())
        obj_mask_task_sens = (
                obj_mask
                * task_imp_contrib  # saliency
                * sensitivity_contrib
                * uncertainty_contrib
        )
        # uncertain objects are interesting (prioritizes background!)
        # if self.use_uncertainty_in_gaze_evidences: --> then uncertainty_contrib is set to 1*entropy added number

        # normalize obj_mask_task by the log of the size of the mask
        mask_size_px = torch.sum(obj_mask)
        log_mask_size_dva = torch.maximum(torch.log2(mask_size_px * self.px2dva ** 2), torch.tensor(1.0))
        evidence = (
                torch.sum(obj_mask_task_sens)
                / mask_size_px
                * log_mask_size_dva
        )

        if self.use_IOR_in_gaze_evidences:
            evidence *= (1 - self.inhibition_dict[obj_id_str])

        # if X% of the frame are spent on saccading, only accumulate 100-X% of the evidence
        evidence_update = (evidence.item() + np.random.normal(0, self.decision_noise)) * cur_fov_frac
        # check if the object is in the evidence dictionary
        if obj_id_str in self.evidence_dict:
            # if it is, add the new evidence to the old evidence + noise
            self.evidence_dict[obj_id_str] += evidence_update
        else:
            # if it is not, add the new evidence to the dictionary
            self.evidence_dict[obj_id_str] = evidence_update
        pass

    def determine_target_loc(self, state, max_obj_id):
        if self.use_ground_truth_objects:
            obj_mask = state.object_segmentation.ground_truth_objects == int(max_obj_id)
        else:
            obj_mask = state.object_segmentation.object_segmentation == int(max_obj_id)
        probmap_unormliazed = (
                obj_mask.double()
                * state.task_importance.importance_map
                * state.gaze.sensitivity_map
        )
        normlizer = torch.sum(probmap_unormliazed)
        if normlizer == 0.0:
            probmap = torch.ones_like(probmap_unormliazed) / torch.sum(
                torch.ones_like(probmap_unormliazed)
            )
        else:
            probmap = probmap_unormliazed / normlizer
            # convert to torch: new_gaze_loc = np.array(np.unravel_index(np.random.choice(len(gaze_gaussian.ravel()), p=probmap.ravel()),gaze_gaussian.shape))
        new_gaze_loc = np.array(
            np.unravel_index(
                np.random.choice(
                    len(state.gaze.sensitivity_map.cpu().numpy().ravel()),
                    p=probmap.cpu().numpy().ravel(),
                ),
                state.gaze.sensitivity_map.shape,
            )
        )
        return new_gaze_loc

    @torch.compile(disable=not use_torch_compile, backend=torch_compile_backend)
    def determine_action(self, state: GlobalState, optic_flow):
        """Update the gaze position based on the current location, object segmentation,
        task importance and the history of the gaze positions.

        :param state: Global state of the system
        :type state: GlobalState
        :return: Gaze shift
        :rtype: np.ndarray
        """

        # get all object ids of  the current frame from the combined particle filter result
        if self.use_ground_truth_objects:
            obj_ids = torch.unique(state.object_segmentation.ground_truth_objects)
        else:
            obj_ids = torch.unique(state.object_segmentation.object_segmentation)

        entropy = self.obj_seg_filter.get_entropy_from_particles(
            state.object_segmentation.particle_set
        )   # maximally 1 since border/noborder is like a coinflip

        # do not accumulate evidence for the time the saccade was ongoing
        cur_fov_frac = np.clip(
            1.0 + self.already_waited_frames - self.ongoing_sacdur, 0.0, 1.0
        )
        # if saccade takes up the whole frame, do not update decision variables
        if cur_fov_frac == 0.0:
            self.already_waited_frames += 1
        else:
            self.already_waited_frames = 0
            self.ongoing_sacdur = 0.0

        # store previous decision vals to calculate the update -> waiting time
        prev_evidence_dict = self.evidence_dict.copy()

        # get all the object-independent contributions
        task_imp_contrib = (self.task_importance_added_number + (
                state.task_importance.importance_map / state.task_importance.importance_map.max()).type(
            torch.get_default_dtype())) / (1.0 + self.task_importance_added_number)
        if self.use_center_bias:
            task_imp_contrib *= self.center_bias
        sensitivity_contrib = state.gaze.sensitivity_map.type(torch.get_default_dtype())
        if self.use_uncertainty_in_gaze_evidences:
            uncertainty_contrib = (self.entropy_added_number + entropy.type(torch.get_default_dtype())) / (
                        1.0 + self.entropy_added_number)
            # uncertainty_contrib = self.gaussian_blur(uncertainty_contrib)
            # TODO stronger smoothing? was 3, set to 5
            uncertainty_contrib = torch.from_numpy(cv.GaussianBlur(uncertainty_contrib.cpu().numpy(), (15, 15), 15.0)).to(
                self.device).type(torch.get_default_dtype())
        else:
            min_uncert = self.entropy_added_number / (self.entropy_added_number + 1.0)
            uncertainty_contrib = torch.ones_like(sensitivity_contrib) * min_uncert

        #  plt.imshow(uncertainty_contrib.cpu().numpy() * sensitivity_contrib.cpu().numpy() * task_imp_contrib.cpu().numpy())#, vmin=0, vmax=1)
        state.task_imp_contrib_vis = task_imp_contrib
        state.sensitivity_contrib_vis = sensitivity_contrib
        state.uncertainty_contrib_vis = uncertainty_contrib

        # iterate through all object ids
        for obj_id in obj_ids:
            if self.use_IOR_in_gaze_evidences:
                # add new obj to ior dict or update inhibition based on foveation
                self._update_inhibition_dict(state, obj_id)
            # calculate the evidence based on sensitivity, saliency, history and uncertainty
            # and add it to the evidence dictionary
            self._update_evidence_dict(state, obj_id, task_imp_contrib, sensitivity_contrib, uncertainty_contrib,
                                       cur_fov_frac)
        # print("evidence: ", self.evidence_dict)
        # print("inhibition: ", self.inhibition_dict)

        # after all are updated, make decision if a saccade should be made
        max_dv = max(self.evidence_dict.values())
        if max_dv > self.decision_threshold and len(prev_evidence_dict.keys()) != 0:
            # print(self.evidence_dict)
            # find the object with the highest evidence
            target_obj = max(self.evidence_dict, key=self.evidence_dict.get)
            target_obj_id = int(float(target_obj))
            # Conceptual change compared to ScanDy: no saccades within (our current segmentation) objects!
            # This should account for the fact that humans sometimes have very long fixations.
            # Since our segs are more finegrained than the ground truth, this still allows for within-object saccades
            if self.make_sac_in_obj or (target_obj_id != state.gaze.gaze_object):  # lazy eval!
                # when was the threshold crossed (assuming linearity)
                frac_at_sac_start = (max_dv - self.decision_threshold) / (
                        max_dv - prev_evidence_dict.get(target_obj, 0.0)
                )
                self.already_waited_frames = np.clip(frac_at_sac_start, 0, 1)

                # select a location within that object with probability based on state
                new_gaze_loc = self.determine_target_loc(state, target_obj_id)
                gaze_shift = (
                        torch.tensor(new_gaze_loc, dtype=torch.get_default_dtype())
                        - state.gaze.mu  # center of previous gaze prob
                )
                sac_amp = np.linalg.norm(gaze_shift.cpu().numpy()) * self.px2dva
                self.ongoing_sacdur = self._calc_sac_dur(sac_amp)
                # self.prev_sacdur = self.ongoing_sacdur.copy()
                sac_ang_h = - torch.atan2(gaze_shift[0], gaze_shift[1]).cpu().numpy() / np.pi * 180
                if len(self.foveation_info) > 0:
                    sac_ang_p = su.angle_limits(sac_ang_h - self.foveation_info[-1][10])
                else:
                    sac_ang_p = np.nan

                fov_dur = (
                        state.gaze.current_time - self.fov_start_t - self.already_waited_frames
                )
                self.foveation_info.append(
                    [
                        self.nfov,
                        int(self.fov_start_t),
                        state.gaze.current_time
                        - 1,  # not 100% sure why, but now results make sense
                        fov_dur,
                        self.fov_start_loc[1],  # x_start, corrected Sep 26th
                        self.fov_start_loc[0],  # y_start
                        int(state.gaze.mu[1].cpu().numpy()),  # x_end
                        int(state.gaze.mu[0].cpu().numpy()),  # y_end
                        sac_amp,
                        self.ongoing_sacdur,
                        sac_ang_h,
                        sac_ang_p,
                    ]
                )
                print(
                    f'SACCADE! nfov: {self.nfov}, max_dv: {max_dv}, f_start: {int(self.fov_start_t)}, f_end: '
                    f'{state.gaze.current_time - 1}, fov_dur: {fov_dur}, sac_ang_h: {sac_ang_h}, sac_ang_p: {sac_ang_p}'
                )

                # update variables for the next step
                self.fov_start_loc = new_gaze_loc
                self.fov_start_t = self.fov_start_t + fov_dur + self.ongoing_sacdur
                self.nfov += 1

            else:
                loc = torch.round(state.gaze.mu).type(torch.long)
                object_shift = optic_flow[loc[0], loc[1]]
                gaze_shift = object_shift + self.drift_noise_dva / self.px2dva * (
                        2 * torch.rand(2, dtype=torch.get_default_dtype()) - 1)
            # update dicts: reset evidence dictionary
            for k in self.evidence_dict.keys():
                self.evidence_dict[k] = 0.0
            if self.use_IOR_in_gaze_evidences:
                # update dicts: set inhibition of previous object to 1
                self.inhibition_dict[state.gaze.gaze_object] = 1.0
        else:
            # gaze_shift = torch.zeros(2, dtype=torch.get_default_dtype())
            # if not, smooth pursuit and drift
            loc = torch.round(state.gaze.mu).type(torch.long)
            object_shift = torch.flip(optic_flow[loc[0], loc[1]], dims=(0,))
            shift_norm = torch.norm(object_shift)
            if shift_norm * self.px2dva > 1:
                object_shift = object_shift / shift_norm * 1.0 / self.px2dva
            gaze_shift = object_shift + self.drift_noise_dva / self.px2dva * (
                    2 * torch.rand(2, dtype=torch.get_default_dtype()) - 1)

        return gaze_shift

    def create_final_fov_df(self, state):
        """The last foveation is not ended by a saccade.
        Therefore, we first append the current (i.e. last) foveation to the
        foveation info list, convert it to a dataframe, and return it to
        enable an easy evaluation of the simulated scanpath.

        :return: Foveation info of simulated scanpath
        :rtype: pd.DataFrame
        """
        self.foveation_info.append(
            [
                self.nfov,
                int(self.fov_start_t),
                state.gaze.current_time,  # not 100% sure why, but now results make sense
                state.gaze.current_time
                - self.fov_start_t
                + 1,  # not 100% sure about the +1
                self.fov_start_loc[1],  # corrected, Sep 26th
                self.fov_start_loc[0],
                state.gaze.mu[1].cpu().numpy(),
                state.gaze.mu[0].cpu().numpy(),
                np.nan,  # self.prev_sacamp,
                np.nan,  # self.prev_sacdur,
                np.nan,  # sac_ang_h
                np.nan,  # sac_ang_p
            ]
        )

        foveation_df = pd.DataFrame(
            self.foveation_info,
            columns=[
                "nfov",
                "frame_start",
                "frame_end",
                "duration_ms",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "sac_amp_dva",
                "sac_dur",
                "sac_ang_h",
                "sac_ang_p"
            ],
        )

        # foveation_df["sac_amp_dva"] = foveation_df["sac_amp_dva"] * self.px2dva
        foveation_df["duration_ms"] = foveation_df["duration_ms"] * self.frames2ms
        foveation_df["sac_dur"] = foveation_df["sac_dur"] * self.frames2ms

        # print(f"Video: {}")
        print(
            f"#Fovs: {len(foveation_df)}, mean sac amp: {foveation_df.sac_amp_dva.mean()}, median sac amp: {foveation_df.sac_amp_dva.median()}, mean fov dur: {foveation_df.duration_ms.mean()}, median fov dur: {foveation_df.duration_ms.median()}")

        return foveation_df
