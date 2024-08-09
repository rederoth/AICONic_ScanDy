import torch
import torchvision
from segmentation_particle_filter.particle_filter import (
    ParticleFilterWrapper,
    ParticleFilterParams,
)
from skimage.morphology import skeletonize
from segmentation_particle_filter.utils import shape_convert_np_cv2
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import min_weight_full_bipartite_matching


@dataclass
class ObjectSegmentationParams:
    particle_filter_params: ParticleFilterParams = field( default_factory=lambda: ParticleFilterParams(
        segmentation_algo="fake",
        particle_num=50,
        velocity_apply_number=20,  # 20 -- faster with lower numbers
        gradient_descent_steps=0,
        dist_map_thresh=5,
        img_scale_factor=0.5,
        # max_spawn_prob=0.4,
        segmenter_scale_factor = 1.0,
        # half_max_prob_normalized_size= 0.00000001,
        # overspawn_prob=0.8,

        noise_smoothing_sigma = 100,           # 100 -> higher means less noise
        # FOR ABLATION:
        # SET the measurement Weight and the spawning weight of that segmentation to 0
        # leads to that measurement to be fully skipped
        # measurement order: Appearance, Motion, Semantic Global, Prompt based on Loc, Prompt for Presaccadic
        measurement_weights =[0.4, 0.05, 1.0, 0.6, 0.5],
        measurement_spawning_weights = [0.025, 0.025, 0.1, 0.8, 0.5], # 0.05, 0.05, 0.2, 0.7, 0.5 last two are prompted
        )
    )
    consistent_seg_kernel_size : int = 5
    consistent_seg_sigma : float = 0.5
    consistent_seg_lower_bound_likelihood : float = 0.125   # 0.2


def get_contours_and_weights(particle_set):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    contour_imgs = torch.cat(
        [
            torch.tensor(p.get_boundary_img(), dtype=torch.get_default_dtype()).unsqueeze(0)
            for p in particle_set
        ],
        dim=0,
    )
    w_array = np.array([p.curr_weight for p in particle_set], dtype=np.float32)
    weights = torch.from_numpy(w_array)
    weights = weights.type(torch.get_default_dtype())
    weights = weights.to(device)
    return contour_imgs, weights


def match_old_segmentation_ids(all_old_segmentations, new_segmentation):
    old_id_max = np.max(all_old_segmentations)
    max_id_new = int(np.max(new_segmentation)) + 1
    reward_matrix = np.zeros((max_id_new, int(old_id_max + max_id_new + 1)))
    expanded_mean_img = np.expand_dims(new_segmentation.astype(np.uint32) * 100000, 0)
    all_comp_seg = all_old_segmentations + expanded_mean_img
    for j in range(all_old_segmentations.shape[0]):
        old_segmentation = all_old_segmentations[j]
        comp_seg = all_comp_seg[j]
        ids_old, counts_old = np.unique(old_segmentation[old_segmentation != 0], return_counts=True)
        ids_new, counts_new = np.unique(new_segmentation[new_segmentation != 0], return_counts=True)
        ids_matching, counts_matching = np.unique(
            comp_seg[np.logical_or(old_segmentation != 0, new_segmentation != 0)],
            return_counts=True)
        for i in range(len(ids_matching)):
            matched_id, matched_count = ids_matching[i], counts_matching[i]
            new_id = int(np.floor(matched_id / 100000))
            old_id = int(matched_id % 100000)
            if new_id == 0 or old_id == 0:
                continue
            new_idx = np.argwhere(ids_new == new_id)[0]
            old_idx = np.argwhere(ids_old == old_id)[0]
            time_prefac = (0.2 ** j)
            reward_matrix[new_id, old_id] += matched_count / (
                    counts_new[new_idx] + counts_old[old_idx] - matched_count) * time_prefac
    for i in range(old_id_max + 1, old_id_max + max_id_new + 1):
        reward_matrix[:, i] = 0.0000000001 * 10 / float(i + 1 - old_id_max)
    reward_matrix_csr = csr_matrix(reward_matrix)
    new_ids, corresponding_old_ids = min_weight_full_bipartite_matching(reward_matrix_csr, maximize=True)
    new_seg = np.zeros_like(new_segmentation, dtype=np.int32)
    for i in range(1, len(new_ids)):
        new_seg[new_segmentation == new_ids[i]] = corresponding_old_ids[i]
    return new_seg

class ObjectSegmentationState:
    def __init__(self, save_particles: bool = False):
        self.object_segmentation = torch.zeros(200, 200)
        self.particle_set = []
        self.current_time = 0
        self.ground_truth_objects = None
        self.save_particles = save_particles

    def init_state(self, object_segmentation, particle_set):
        self.object_segmentation = object_segmentation
        self.particle_set = particle_set
        self.current_time = 0

    def update_state(self, object_segmentation, particle_set):
        self.object_segmentation = object_segmentation
        self.particle_set = particle_set
        self.current_time = self.current_time + 1

    def create_visualization(self, image_size):
        np_seg = self.object_segmentation.cpu().numpy().astype(np.uint32)
        particles_labeled = [p.get_labeled_img() for p in self.particle_set[:15]]
        if self.save_particles:
            smaller_segmentation = cv.resize(np_seg.astype(float),
                                         np.flip(particles_labeled[0].shape).astype(int),
                                         interpolation=cv.INTER_NEAREST).astype(np.uint32)
            smaller_segmentation = np.expand_dims(smaller_segmentation, 0)
            particles_matched = [match_old_segmentation_ids(smaller_segmentation, p) for p in particles_labeled]
        else:
            particles_matched = []
        contour_imgs, weights = get_contours_and_weights(self.particle_set)
        border_likelihood = torch.einsum("pij,p->ij", contour_imgs, weights)
        no_border_likelihood = 1.0 - border_likelihood
        border_likelihood[border_likelihood == 0] = 0.00000001  # 0 is edge case
        no_border_likelihood[no_border_likelihood == 0] = 0.00000001  # 0 is edge case
        log_likelihood_border = torch.log2(border_likelihood)
        log_likelihood_no_border = torch.log2(no_border_likelihood)
        entropy = (
            -border_likelihood * log_likelihood_border
            - no_border_likelihood * log_likelihood_no_border
        )
        np_cert = entropy.cpu().numpy()
        return np_seg, np_cert, border_likelihood.cpu().numpy(), entropy.cpu().numpy(), particles_matched  # TODO only np_seg, np_cert


class ObjectSegmentationFilter:
    def __init__(self, config):
        self.spf = ParticleFilterWrapper(config.particle_filter_params)
        self.consistent_seg_kernel_size = config.consistent_seg_kernel_size
        self.consistent_seg_sigma = config.consistent_seg_sigma
        self.consistent_seg_lower_bound_likelihood = config.consistent_seg_lower_bound_likelihood
        self.old_segmentations = None
        self.recdeing_time_horizon = 5
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

    def get_init_particle_set(self, img):
        self.spf.update_orig_img_shape(img[0].shape)
        return self.spf.create_particle_set(img)

    def get_segmentation_from_particles(self, particle_set, old_segmentation):
        # build new segmentation
        contour_imgs, weights = get_contours_and_weights(particle_set)
        border_likelihood = torch.einsum("pij,p->ij", contour_imgs, weights)
        blur = torchvision.transforms.functional.gaussian_blur(border_likelihood.unsqueeze(0),
                                                               self.consistent_seg_kernel_size,
                                                               self.consistent_seg_sigma).squeeze()
        mean = blur > self.consistent_seg_lower_bound_likelihood
        mean_img = mean.squeeze().cpu().numpy().astype(np.uint8)
        # thinning thick uncertain boundaries (needs a white border)
        extended_mean_img = np.ones((mean_img.shape[0]+2, mean_img.shape[1]+2), dtype=np.uint8)
        extended_mean_img[1:-1, 1:-1] = mean_img
        mean_img = skeletonize(extended_mean_img)
        kernel = np.ones((2, 2), np.uint8)
        mean_img = cv.morphologyEx(mean_img[1:-1, 1:-1].astype(np.uint8), cv.MORPH_DILATE, kernel)
        mean_img = np.logical_not(mean_img).astype(np.uint8)
        contours, _ = cv.findContours(mean_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea)
        contours.reverse()
        mean_img = np.zeros(mean_img.shape, np.uint8)
        for i in range(len(contours)):
            tmp_img = np.zeros_like(mean_img)
            cv.drawContours(tmp_img, contours=contours, contourIdx=i, color=1, thickness=-1)
            mask = tmp_img > 0
            if np.sum(tmp_img) > 100:
                mean_img[mask] = i+1

        # get last segmentation(s) for ID matching
        old_segmentation = old_segmentation.cpu().numpy().astype(float)
        old_segmentation = cv.resize(old_segmentation,
                                     np.flip(mean_img.shape).astype(int),
                                     interpolation=cv.INTER_NEAREST).astype(np.uint32)
        if self.old_segmentations is None:

            self.old_segmentations = np.expand_dims(old_segmentation, 0)
        else:
            self.old_segmentations = np.concatenate([np.expand_dims(old_segmentation, 0), self.old_segmentations[:(
                        self.recdeing_time_horizon - 1)]], axis=0)

        all_old_segmentations = self.old_segmentations
        labeled_img = self.match_segmentation_ids(all_old_segmentations, mean_img)
        return labeled_img

    def match_segmentation_ids(self, all_old_segmentations, new_segmentation):
        new_seg = match_old_segmentation_ids(all_old_segmentations, new_segmentation)
        # next_possible_id = np.max(corresponding_old_ids) + 1
        # for i in range(max_id_new):
        #     if np.sum(new_ids == i) == 0:
        #         new_seg[mean_img == i] = next_possible_id
        #         next_possible_id += 1
        # print("Available old IDS: ", ids_old)
        # print("IDs after op ", corresponding_old_ids)
        labeled_img = cv.resize(new_seg, np.flip(1 / self.spf.img_scale_factor * np.array(new_seg.shape)).astype(int),
                                interpolation=cv.INTER_NEAREST)
        labeled_img = torch.from_numpy(labeled_img).type(torch.get_default_dtype()).to(self.device)
        return labeled_img

    def get_entropy_from_particles(self, particle_set):
        contour_imgs, weights = get_contours_and_weights(particle_set)
        border_likelihood = torch.einsum("pij,p->ij", contour_imgs, weights)
        no_border_likelihood = 1.0 - border_likelihood
        border_likelihood[border_likelihood <= 0] = 0.00000001  # 0 is edge case
        no_border_likelihood[no_border_likelihood <= 0] = 0.00000001  # 0 is edge case
        log_likelihood_border = torch.log2(border_likelihood)
        log_likelihood_no_border = torch.log2(no_border_likelihood)
        entropy = (
            -border_likelihood * log_likelihood_border
            - no_border_likelihood * log_likelihood_no_border
        )
        if torch.any(torch.isnan(border_likelihood)):
            print("NAAAAAN")
        if torch.any(torch.isnan(log_likelihood_border)):
            print("NAAAAAN")
        if torch.any(torch.isnan(no_border_likelihood)):
            print("NAAAAAN")
        if torch.any(torch.isnan(log_likelihood_no_border)):
            print("NAAAAAN")
        entropy = entropy.cpu().numpy()
        entropy = cv.resize(entropy, np.flip((1 / self.spf.img_scale_factor * np.array(entropy.shape)).astype(int)), interpolation=cv.INTER_NEAREST)
        return torch.from_numpy(entropy).type(torch.get_default_dtype()).to(self.device)

    def correct(self, state, gaze_state, measurements, of_measurement, meas_time):
        # from particles
        modified_measurements = [cv.resize(m.cpu().numpy(), shape_convert_np_cv2(self.spf.filter.segmenter.goal_shape), interpolation=cv.INTER_NEAREST) for m in measurements]
        modified_of_meas = cv.resize(of_measurement.cpu().numpy(), shape_convert_np_cv2(self.spf.filter.segmenter.goal_shape), interpolation=cv.INTER_NEAREST)
        modified_of_meas = modified_of_meas * self.spf.img_scale_factor
        particle_set = self.spf.one_iteration_multiple_measurements(state.particle_set, modified_measurements, modified_of_meas)
        object_segmentation = self.get_segmentation_from_particles(particle_set, state.object_segmentation)
        particle_based_entropy = self.get_entropy_from_particles(
            particle_set
        )  # lies between 0 and 1
        state.update_state(object_segmentation, particle_set)
