import argparse
import torch
from GazeFilter import gaussian_2d_torch
from experiment_util import BaseExperiment
from scanpath_producer import GlobalState, ScanPathProducer
from ObjectSegmentationFilter import ObjectSegmentationParams
from segmentation_particle_filter.segmenters import get_segmenter
from segmentation_particle_filter.utils import shape_convert_np_cv2
from dataclasses import dataclass, field
import cv2 as cv
import os
import glob
import numpy as np
import pickle
import random
import threading
import tqdm
import shutil
import tyro


def anisotropic_centerbias(xmax, ymax, sigx2=0.22, v=0.45, mean_to_1=False):
    """
    Function returns a 2D anisotropic Gaussian center bias across the whole frame.
    Inspired by Clark & Tatler 2014 / also used in Nuthmann2017
    Image dimensions are normalized to [-1,1] in x and [-a,a] in y, with aspect ratio a.
    The default values have been taken from Clark & Tatler (2014).
    Influence may be smaller (sigx2 bigger) in dynamic scenes (Cristino&Baddeley2009; tHart2009).
    Update: Now in torch!

    :param xmax: Size of the frame in x-direction
    :type xmax: int
    :param ymax: Size of the frame in y-direction
    :type ymax: int
    :param sigx2: Normalized variance of the Gaussian in x direction, defaults to 0.22
    :type sigx2: float, optional
    :param v: Anisotropy, defaults to 0.45
    :type v: float, optional
    :param mean_to_1: Normalize such that the mean of the Gauss is one (instead of the max), defaults to False
    :type mean_to_1: bool, optional
    :return: Center bias in the form of a 2D Gaussian with dimensions of a frame
    :rtype: np.array
    """
    X, Y = torch.meshgrid(
        torch.linspace(-1, 1, xmax), torch.linspace(-ymax / xmax, ymax / xmax, ymax)
    )
    G = torch.exp(-(X ** 2 / (2 * sigx2) + Y ** 2 / (2 * sigx2 * v)))
    if mean_to_1:
        G = G / torch.mean(G)
    return G


TMP_CACHING_DIR = "/tmp/scanpath_data_cache"


@dataclass
class ScanpathExperimentConfig:
    object_segmentation_params: ObjectSegmentationParams = field(default_factory=lambda: ObjectSegmentationParams())
    video_directory: str = None
    save_imgs: bool = False
    save_particles: bool = False
    seed: int = 123
    assumed_gaze_motion_noise: float = 0.1
    assumed_gaze_meas_noise: float = 0.1
    obj_inhibition: float = 0.0  # 0.9 UNUSED! config.obj_inhibition, set in [0,1]
    inhibition_r: float = 1 / 100  # UNUSED! config.inhibition_r, set in [0,1]
    decision_threshold: float = 4.0  # config.decision_threshold, set in [0,inf]
    presaccadic_threshold: float = 0.8 * decision_threshold
    decision_noise: float = 0.3  # decision_threshold * 0.075 but does not adapt when thres is changed, set in [0,small]
    sensitivity_dva_sigma: float = 6.0  # set in [DVA] (~7-10)
    drift_noise_dva: float = 0.0  # 0.125  # config.drift_noise, set in [0,inf]  # TODO seems a bit high to me but dunno
    entropy_added_number: float = 0.5  # influences how much impact entropy has impact on saccade decision, set in ]0,1]
    task_importance_added_number: float = 0.1  # added towards task importance map, set in ]0,1]  # was 1/3
    use_uncertainty_in_gaze_evidences: bool = True
    use_IOR_in_gaze_evidences: bool = False
    use_ground_truth_objects: bool = False
    stop_after_3sec: bool = False
    output_directory: str = None
    prompted_sam: bool = True  # if set to false, do not use the prompted sam!
    presaccadic_prompting: bool = False  # if set to false, do not use the presaccadic prompting
    presaccadic_sensitivity: float = 0.8
    use_motion_seg: bool = True
    use_app_seg: bool = True
    use_semantic_seg: bool = True
    saccadic_momentum: bool = False  # if set to True, include an angle preference in the (otherwise Gaussian) sensitivity
    sac_momentum_min: float = 0.0  # momentum map is scaled between this and 1, was 0.5
    sac_momentum_max: float = 2.0  # momentum map is scaled between this and 1, was 0.5
    sac_momentum_restricted_angle: int = 180  # 180 means whole range, smaller leads to cone
    sac_momentum_on_obj: bool = False
    make_sac_in_obj: bool = True  # if set to false, this leads to long fix_dur, makes sense if prompt gives higher granularity
    center_bias: bool = True  # most recent but already used in scandy, might help with more horizontal sacs
    use_low_level_prompt : bool = False


class ScanpathExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self.gaze_meas_noise_distribution = None
        self.gaze_action_noise_distribution = None
        self.global_state = None
        self.scanpath_agent = None
        self.gaze_locations = None
        self.experiment_steps = None
        self.true_pos = None
        self.init_frame_num = None
        self.save_imgs = False
        self.save_particles = False

    def calc_px2dva(self, image_size, display_size=(1080, 1920), display_w_dva=47.7, max_scaling=0.8):
        px2dva_unscaled = display_w_dva / display_size[1]
        if display_size[1] / display_size[0] > image_size[1] / image_size[0]:
            # display is wider than image -> limit by height
            movie_scale_factor = display_size[0] * max_scaling / image_size[0]
        else:
            # display is taller than image -> limit by width
            movie_scale_factor = display_size[1] * max_scaling / image_size[1]
        return px2dva_unscaled * movie_scale_factor

    def preaction(self, config):
        if os.path.isdir(TMP_CACHING_DIR):
            shutil.rmtree(TMP_CACHING_DIR)
        os.makedirs(TMP_CACHING_DIR)
        self.save_imgs = config.save_imgs
        self.save_particles = config.save_particles
        torch.random.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        cv.setRNGSeed(config.seed)

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

        # self.init_frame_num = int(init_object_seg_path.split("/")[-1].split(".")[0])
        self.init_frame_num = 0
        init_frame_path = os.path.join(
            config.video_directory, "images", f"{(self.init_frame_num):04d}.png"
        )
        init_img = torch.from_numpy(cv.imread(init_frame_path)).to(self.device)

        self.true_pos = torch.concatenate([torch.randint(low=0, high=init_img.shape[0], size=(1,)),
                                           torch.randint(low=0, high=init_img.shape[1], size=(1,))]).type(
            torch.get_default_dtype())
        shape = shape_convert_np_cv2(init_img.shape)[:2]
        self.img_shape = shape
        px2dva = self.calc_px2dva(shape, display_size=(1080, 1920), display_w_dva=47.7, max_scaling=0.8)

        segmenter_sam = get_segmenter("SAM", shape, shape)
        init_object_seg = torch.from_numpy(segmenter_sam.get_labeled_img(init_img.cpu().numpy())).to(
            self.device
        )
        # self.saliency = cv.saliency.StaticSaliencySpectralResidual_create()
        # _, init_sal_map = self.saliency.computeSaliency(init_img.cpu().numpy())
        # init_task_importance_map = torch.from_numpy(init_sal_map).to(torch.cuda.current_device())
        init_sal_path = os.path.join(
            config.video_directory, "saliencies", "unisal_DHF1K", f"{(self.init_frame_num):04d}.png"
        )
        tmp_saliency = cv.resize(cv.cvtColor(cv.imread(init_sal_path), cv.COLOR_BGR2GRAY), shape)
        init_task_importance_map = torch.from_numpy(tmp_saliency).to(self.device)
        if config.stop_after_3sec:
            self.experiment_steps = 90  # 30 fps -1 ?
        else:
            self.experiment_steps = len(glob.glob(os.path.join(config.video_directory, "images", "*.png"))) - 1
        self.gaze_locations = torch.zeros(self.experiment_steps, 2)
        # define global state and agent
        self.scanpath_agent = ScanPathProducer(config, self.true_pos)
        self.scanpath_agent.px2dva = px2dva
        self.scanpath_agent.gaze_filter.px2dva = px2dva
        self.scanpath_agent.gaze_filter.saccadic_momentum = config.saccadic_momentum
        if config.center_bias:
            self.scanpath_agent.center_bias = anisotropic_centerbias(shape[1], shape[0])

        sensitivity_map = gaussian_2d_torch(
            int(self.true_pos[0]), int(self.true_pos[1]), init_task_importance_map.shape[0],
            init_task_importance_map.shape[1], config.sensitivity_dva_sigma / px2dva
        )
        self.global_state = GlobalState(save_particles=config.save_particles)
        segmenter_graph = get_segmenter("graph", (np.array(shape) * 0.35).astype(int),
                                        (np.array(shape) * 0.35).astype(int))
        grey = cv.cvtColor(init_img.cpu().numpy(), cv.COLOR_BGR2GRAY)
        init_object_seg2 = torch.from_numpy(segmenter_graph.get_labeled_img(grey)).to(
            self.device
        )
        init_particles = self.scanpath_agent.obj_seg_filter.get_init_particle_set(
            [init_object_seg.cpu().numpy(), init_object_seg2.cpu().numpy()]
        )
        self.global_state.init_state(
            self.true_pos, init_object_seg, init_particles, init_task_importance_map, sensitivity_map,
        )
        self.gaze_action_noise_distribution = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(2), torch.eye(2) * config.assumed_gaze_motion_noise
            )
        )
        self.gaze_meas_noise_distribution = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(2), torch.eye(2) * config.assumed_gaze_meas_noise
            )
        )

        self.scanpath_agent.px2dva = px2dva
        self.segmenter_sam = get_segmenter("SAM", (np.array(shape) * 0.5).astype(int),
                                           (np.array(shape) * 0.5).astype(int))
        if not config.use_low_level_prompt:
            self.segmenter_sam_prompt = get_segmenter("PromptFASTSAM", shape, shape)
        else:
            assert config.prompted_sam, 'Prompted graph overwrites prompted sam, so it has to be active!'
            self.segmenter_sam_prompt = get_segmenter("graph_prompt", shape, shape)

        self.segmenter_graph = get_segmenter("graph", (np.array(shape) * 0.35).astype(int),
                                             (np.array(shape) * 0.35).astype(int))
        self.segmenter_motion = get_segmenter("graph", (np.array(shape) * 0.35).astype(int),
                                              (np.array(shape) * 0.35).astype(int))

        self.frame_path = os.path.join(
            config.video_directory, "images", f"{self.init_frame_num :04d}.png"
        )
        self.path_app_segs = os.path.join(
            config.video_directory, "app_seg"
        )
        os.makedirs(self.path_app_segs, exist_ok=True)
        self.path_motion_segs = os.path.join(
            config.video_directory, "motion_seg_simple"
        )
        os.makedirs(self.path_motion_segs, exist_ok=True)
        self.path_semantic_segs = os.path.join(
            config.video_directory, "semantic_seg"
        )
        os.makedirs(self.path_semantic_segs, exist_ok=True)

        if config.use_ground_truth_objects:
            init_ground_truth = torch.from_numpy(
                cv.imread(os.path.join(config.video_directory, "mask", f"{(self.init_frame_num):04d}.png"),
                          cv.IMREAD_GRAYSCALE)).to(
                self.device)
            self.global_state.object_segmentation.ground_truth_objects = init_ground_truth

        # img = torch.from_numpy(cv.imread(self.frame_path)).to(torch.cuda.current_device())
        # self.img_shape = shape_convert_np_cv2(img.shape)[:2]
        # self.preloaded_saliency = np.load(os.path.join(config.video_directory, "molin_sal.npy"))

    def experiment_main(self, config, pre_results):
        outputpath = TMP_CACHING_DIR

        # avoid instant saccades in the first frame
        self.scanpath_agent.evidence_dict = {}

        for i in tqdm.tqdm(range(self.experiment_steps)):
            if i == 0:
                print('First frame of new vid...')
            # start_time = time.time()  # for printing only!
            flow_path = os.path.join(
                config.video_directory, "flows", "VideoFlow", f"flow_{(self.init_frame_num + i + 1):03d}.npy"
                # config.video_directory, "flows", f"{(self.init_frame_num + i):04d}.flo"
            )
            # optic_flow = torch.from_numpy(read_flo(flow_path)).to(torch.cuda.current_device())
            optic_flow = torch.from_numpy(np.load(flow_path)).to(self.device).type(torch.get_default_dtype())

            frame_path = os.path.join(
                config.video_directory, "images", f"{(self.init_frame_num + i):04d}.png"
            )
            img = torch.from_numpy(cv.imread(frame_path)).to(self.device)
            if config.use_ground_truth_objects:
                ground_truth_objects = torch.from_numpy(
                    cv.imread(os.path.join(config.video_directory, "mask", f"{(self.init_frame_num + i):04d}.png"),
                              cv.IMREAD_GRAYSCALE)).to(
                    self.device)
                self.global_state.object_segmentation.ground_truth_objects = ground_truth_objects

            action = self.scanpath_agent.determine_action(self.global_state, optic_flow)
            old_true = self.true_pos
            self.true_pos = self.true_pos + action
            self.true_pos[0] = torch.clip(self.true_pos[0], 0, img.shape[0] - 1)
            self.true_pos[1] = torch.clip(self.true_pos[1], 0, img.shape[1] - 1)
            action_delta = old_true + action - self.true_pos
            self.true_pos = self.true_pos + self.gaze_action_noise_distribution.sample()
            self.true_pos[0] = torch.clip(self.true_pos[0], 0, img.shape[0] - 1)
            self.true_pos[1] = torch.clip(self.true_pos[1], 0, img.shape[1] - 1)
            action = action - action_delta
            # self.saliency = cv.saliency.StaticSaliencySpectralResidual_create()
            # _, saliencyMap = self.saliency.computeSaliency(img.cpu().numpy())
            sal_path = os.path.join(
                config.video_directory, "saliencies", "unisal_DHF1K", f"{(self.init_frame_num + i):04d}.png"
            )
            tmp_saliency = cv.resize(
                cv.cvtColor(cv.imread(sal_path), cv.COLOR_BGR2GRAY), self.img_shape
            )
            task_importance_map = torch.from_numpy(tmp_saliency).to(self.device)
            # mask_path = os.path.join(
            #     config.video_directory, "mask", str(self.init_frame_num + i) + ".png"
            # )

            object_segs = self.obtain_object_segmentations(config, i, img, optic_flow)
            self.scanpath_agent.post_action_inference(
                action,
                self.global_state,
                self.true_pos + self.gaze_meas_noise_distribution.sample(),
                object_segs,
                optic_flow,
                task_importance_map,
                float(i),
            )
            self.gaze_locations[i] = self.true_pos
            self.process_and_save(self.init_frame_num + i, img, object_segs, optic_flow, outputpath,
                                  self.gaze_locations[:i].cpu().numpy())
            # print("Time for step: "+str(time.time() - start_time))

        foveation_df = self.scanpath_agent.create_final_fov_df(self.global_state)

        return self.gaze_locations.detach().cpu(), foveation_df

    def obtain_object_segmentations(self, config, i, img, optic_flow):
        if config.use_app_seg:
            curr_app_seg_path = os.path.join(self.path_app_segs, f"{(self.init_frame_num + i):04d}.png")
            if os.path.exists(curr_app_seg_path):
                object_seg_graph = torch.from_numpy(cv.imread(curr_app_seg_path).astype(np.float32)).to(
                    self.device)[:, :, 0]
            else:
                grey = cv.cvtColor(img.cpu().numpy(), cv.COLOR_BGR2GRAY)
                object_seg_graph = torch.from_numpy(self.segmenter_graph.get_labeled_img(grey)).to(
                    self.device)
                cv.imwrite(curr_app_seg_path, object_seg_graph.cpu().numpy())
        else:
            object_seg_graph = torch.zeros(shape_convert_np_cv2(self.img_shape))
        curr_motion_seg_path = os.path.join(self.path_motion_segs, f"{(self.init_frame_num + i):04d}.png")
        if config.use_motion_seg:
            if os.path.exists(curr_motion_seg_path):
                object_seg_motion = torch.from_numpy(cv.imread(curr_motion_seg_path).astype(np.float32)).to(
                    self.device)[:, :, 0]
            else:
                object_seg_motion = torch.from_numpy(
                    self.segmenter_motion.get_labeled_img(optic_flow.cpu().numpy() * 10)
                ).to(self.device)
                if torch.all(object_seg_motion == object_seg_motion[0, 0]):  # basically no segments
                    object_seg_motion = torch.zeros_like(object_seg_motion)
                cv.imwrite(curr_motion_seg_path, object_seg_motion.cpu().numpy())

        else:
            object_seg_motion = torch.zeros(shape_convert_np_cv2(self.img_shape))
        if config.use_semantic_seg:
            curr_semantic_seg_path = os.path.join(self.path_semantic_segs, f"{(self.init_frame_num + i):04d}.png")
            if os.path.exists(curr_semantic_seg_path):
                object_seg_sam = torch.from_numpy(cv.imread(curr_semantic_seg_path).astype(np.float32)).to(
                    self.device)[:, :, 0]
            else:
                object_seg_sam = torch.from_numpy(self.segmenter_sam.get_labeled_img(img.cpu().numpy())).to(
                    self.device
                )
                cv.imwrite(curr_semantic_seg_path, object_seg_sam.cpu().numpy())
        else:
            object_seg_sam = torch.zeros(shape_convert_np_cv2(self.img_shape))
        self.segmenter_sam_prompt.give_prompt_point(self.true_pos.cpu().numpy()[::-1])
        if config.prompted_sam:
            object_seg_sam_prompt = torch.from_numpy(self.segmenter_sam_prompt.get_labeled_img(img.cpu().numpy())).to(
                self.device)
        else:
            object_seg_sam_prompt = torch.zeros(shape_convert_np_cv2(self.img_shape))
        # presaccadic
        if config.presaccadic_prompting and (
                max(self.scanpath_agent.evidence_dict.values()) > self.scanpath_agent.presaccadic_threshold):
            # TODO that means onlz one object?!
            target_obj_id = int(
                float(max(self.scanpath_agent.evidence_dict, key=self.scanpath_agent.evidence_dict.get))
            )
            # randomly select a location within that object
            new_prompt_loc = self.scanpath_agent.determine_target_loc(self.global_state, target_obj_id)
            self.segmenter_sam_prompt.give_prompt_point(new_prompt_loc[::-1])
            object_seg_sam_prompt_presaccadic = torch.from_numpy(
                self.segmenter_sam_prompt.get_labeled_img(img.cpu().numpy())
            ).to(self.device)
        else:
            object_seg_sam_prompt_presaccadic = torch.zeros_like(object_seg_sam_prompt)
        object_segs = [object_seg_graph, object_seg_motion, object_seg_sam, object_seg_sam_prompt,
                       object_seg_sam_prompt_presaccadic]
        return object_segs

    def process_and_save(self, frame_num, img, object_segs, optic_flow, outputpath, gaze_locs):
        if not self.save_imgs:
            return

        G, sensitivity_contrib, gaze_history, task_contrib, obj_seg, flow_visulaized, uncert_contrib, entropy, particles_matched = self.global_state.visualize(
            img.shape,
            gaze_locs,
            optic_flow.cpu().numpy(),
        )

        if not self.save_particles:
            particles_matched = []
        t = threading.Thread(target=self.save_data, args=(
            G, flow_visulaized, frame_num, uncert_contrib, entropy, obj_seg, object_segs, task_contrib,
            sensitivity_contrib, self.scanpath_agent.evidence_dict, outputpath, particles_matched), name="IOforFrame" + str(frame_num))
        t.start()
        self.io_threads.append(t)

    @staticmethod
    def save_data(G, flow_visulaized, frame_num, uncert_contrib, entropy, obj_seg, object_segs, task_contrib,
                  sensitivity_contrib, evidence_dict, outputpath, particles_matched):
        data = {"gaze_img": G,
                "task_imp_img": task_contrib,
                "obj_seg_img": obj_seg,
                "flow_visualized": flow_visulaized,
                "obj_cert_img": uncert_contrib,
                "particle_entropy": entropy,
                "sensitivity_img": sensitivity_contrib,
                "obj_seg_measurements": object_segs,
                "evidence_dict": evidence_dict,
                "particle_segmentations": particles_matched}
        with open(os.path.join(outputpath, f"{(frame_num):04d}.pickle"), "wb") as f:
            pickle.dump(data, f)

    def postaction(self, config, results, pre_results):
        for t in self.io_threads:
            t.join()
        del self.io_threads
        del self.pre_results
        del self.results
        del self.gaze_meas_noise_distribution
        del self.gaze_action_noise_distribution
        del self.global_state
        del self.scanpath_agent
        del self.gaze_locations
        del self.experiment_steps
        del self.true_pos
        del self.init_frame_num
        torch.cuda.empty_cache()
        shutil.copytree(TMP_CACHING_DIR, config.output_directory, dirs_exist_ok=True)
        shutil.rmtree(TMP_CACHING_DIR)

    def extract_directory(self, config):
        return config.output_directory


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Using CUDA")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("NOT using CUDA")
        torch.set_default_tensor_type(torch.FloatTensor)
    parser = argparse.ArgumentParser(
        prog='ScanpathExperiment',
        description='Runs one Experiment, meaning ONE video Sequence with ONE set of parameters, BOTH specified with '
                    'the config file')
    parser.add_argument('config_file_path', type=str, help="Path to the config file for the experiment")
    parser.add_argument('local_cache_dir', type=str, help="Path to temporally cache intermediate results")
    args, _ = parser.parse_known_args()
    with open(args.config_file_path, "rb") as f:
        config = tyro.from_yaml(ScanpathExperimentConfig, f)
    TMP_CACHING_DIR = args.local_cache_dir
    exp = ScanpathExperiment()
    exp.run(config)
