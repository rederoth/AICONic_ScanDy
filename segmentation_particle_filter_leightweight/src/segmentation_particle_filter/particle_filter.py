import random

from segmentation_particle_filter.measurement_model import MeasurementModel
from segmentation_particle_filter.forward_model import ForwardModel
from segmentation_particle_filter.particle import Particle
from segmentation_particle_filter.utils import convert_label_to_boundary, get_highest_weighted_particle
from segmentation_particle_filter.segmenters import get_segmenter
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ParticleFilterParams:
    """
    :param particle_num: number of particles to use
    :param img_scale_factor: scalefactor for the internally used image shape of the segmentations
    :param segmentation_algo: string identifying the segmentation algorithm to be used
    :param velocity_apply_number: amount of times that the velocities should be applied
    :param noise_smoothing_sigma: standard deviation used for the gaussian blurring of the random sampled image used to apply noise to each particles segmentation
    :param gradient_descent_steps: amount of times that the gradient descent steps should be performed
    :param half_max_prob_normalized_size: normalized size for which the spawning probability should be half of the maximum
    :param max_spawn_prob: maximum possible probability with which to spawn a segment
    :param dist_map_thresh: threshold after which distance (in pixels) the gradient descent should not be applied anymore
    :param overspawn_prob: probability that in case of spawning the boundaries in that area are kept or "overspawned"
    """
    particle_num : int = 50
    img_scale_factor: float = 0.4
    segmenter_scale_factor : float = 1.0
    segmentation_algo : str = "hfs"
    velocity_apply_number : int = 1

    noise_smoothing_sigma : int = 100
    gradient_descent_steps : int = 20
    half_max_prob_normalized_size : float = 0.000001

    max_spawn_prob : float = 0.7
    dist_map_thresh : int = 30
    overspawn_prob : float = 0.9

    measurement_weights: list = field(default_factory=lambda: [1.0, 1.0, 3.0, 4.0])
    measurement_spawning_weights: list = field(default_factory=lambda: [0.01, 0.01, 10.0, 30.0])


class ParticleFilterWrapper:
    """Wrapper for the particle filter that enables parallel computation for the particles"""

    def __init__(self, config):
        """
        Initializes filter models, segmenter and threaded workers for parallelization
        """

        self.worker_pool = None  # concurrent.futures.ThreadPoolExecutor(max_workers=int(os.cpu_count() * 7.0/8.0))
        self.segmetation_algo = config.segmentation_algo
        self.img_scale_factor = config.img_scale_factor
        self.segmenter_scale_factor = config.segmenter_scale_factor
        self.filter = SegmentationParticleFilter(particle_num=config.particle_num,
                                                 segmenter=None,
                                                 velocity_apply_number=config.velocity_apply_number,
                                                 noise_smoothing_sigma=config.noise_smoothing_sigma,
                                                 gradient_descent_steps=config.gradient_descent_steps,
                                                 half_max_prob_normalized_size=config.half_max_prob_normalized_size,
                                                 max_spawn_prob=config.max_spawn_prob, dist_map_thresh=config.dist_map_thresh,
                                                 overspawn_prob=config.overspawn_prob, img_shape=None,
                                                 measurement_weights=config.measurement_weights,
                                                 measurement_spawning_weights=config.measurement_spawning_weights)

    def create_particle_set(self, img):
        """Creates the particle set for the filter, initializes particles based on the given source image

        Parameters
        ----------
        img :
            camera view image of the scene

        Returns
        -------
        type
            list of the created particles (set is also kept track of in the filter itself)

        """
        return self.filter.create_particle_set(img)

    def one_iteration(self, particle_set, img, optical_flow):
        """Performs one iteration of the filter based on the given camera view image

        Parameters
        ----------
        img :
            camera view image of the scene
        shape_masks :
            images containing the masks of currently tracked bodies
        feature_positions :
            current positions of features

        Returns
        -------
        type
            the output of the segmentation algorithm (so it can be sent to the visualizer)

        """
        return self.filter.one_iteration(img=img, particle_set=particle_set, worker_pool=self.worker_pool, optical_flow=optical_flow)

    def one_iteration_multiple_measurements(self, particle_set, measuremnts, optical_flow):
        return self.filter.one_iteration_multiple_measurements(particle_set, measuremnts, optical_flow, worker_pool=self.worker_pool)

    def update_orig_img_shape(self, orig_img_shape):
        goal_shape = (int(np.round(orig_img_shape[0]*self.img_scale_factor)), int(np.round(orig_img_shape[1]*self.img_scale_factor)))
        # print(str(orig_img_shape) + " " + str(goal_shape))
        self.filter.update_image_shape(goal_shape)
        segmenter = get_segmenter(self.segmetation_algo, goal_shape, orig_img_shape, self.segmenter_scale_factor)
        self.filter.set_segmenter(segmenter)

    def __del__(self):
        """
        Destructor
        """
        del self.filter
        # self.worker_pool.shutdown()
        del self.worker_pool


class SegmentationParticleFilter:
    """The segmentation particle filter class"""

    def __init__(self, segmenter, img_shape, particle_num=100, velocity_apply_number=1,
                 noise_smoothing_sigma=5, gradient_descent_steps=3, half_max_prob_normalized_size=0.01,
                 max_spawn_prob=0.01, dist_map_thresh=15, overspawn_prob=0.65, measurement_weights=None, measurement_spawning_weights=None):
        """
        Initializes filter models

        :param segmenter: segmenter object, which can perform image segmentation
        :param particle_num: number of particles to use
        :param velocity_apply_number: amount of times that the velocities should be applied
        :param noise_smoothing_sigma: standard deviation used for the gaussian blurring of the random sampled image used to apply noise to each particles segmentation
        :param gradient_descent_steps: amount of times that the gradient descent steps should be performed
        :param half_max_prob_normalized_size: normalized size for which the spawning probability should be half of the maximum
        :param max_spawn_prob: maximum possible probability with which to spawn a segment
        :param dist_map_thresh: threshold after which distance (in pixels) the gradient descent should not be applied anymore
        :param overspawn_prob: probability that in case of spawning the boundaries in that area are kept or "overspawned"
        :param img_shape: image shape in which the particles keep the segmentation
        """
        self.particle_num = particle_num
        self.segmenter = segmenter
        self.measurement_model = MeasurementModel(particle_num=particle_num,
                                                  gradient_descent_steps=gradient_descent_steps,
                                                  half_max_prob_normalized_size=half_max_prob_normalized_size,
                                                  max_spawn_prob=max_spawn_prob, dist_map_thresh=dist_map_thresh,
                                                  overspawn_prob=overspawn_prob)
        self.forward_model = ForwardModel(velocity_apply_number=velocity_apply_number,
                                          noise_smoothing_sigma=noise_smoothing_sigma, img_shape=img_shape)
        if measurement_weights is not None:
            self.measurement_weights = measurement_weights
        else:
            self.measurement_weights = [1.0]
        if measurement_spawning_weights is not None:
            self.measurement_spawning_weight = measurement_spawning_weights
        else:
            self.measurement_spawning_weight = [1.0]


    def create_particle_set(self, img):
        """Creates the particle set based on the given image

        Parameters
        ----------
        img :
            camera view image of the scene

        Returns
        -------
        type
            list of the created particles (set is also kept track of in the filter itself)

        """
        particle_set = []
        diff_graphs = []
        for i in img:
            diff_graphs.append(Particle(self.segmenter.get_labeled_img(i).astype(np.float64)))
        # in principle multiple different segmenters could be used here
        i = 0
        while i < self.particle_num and i < len(diff_graphs):
            particle_set.append(diff_graphs[i])
            i += 1
        while i < self.particle_num:
            particle_set.append(diff_graphs[i % len(diff_graphs)].__copy__())
            i += 1
        self.normalize_particle_set(particle_set)

        return particle_set

    def one_iteration(self, particle_set, img, optical_flow, worker_pool=None):
        """Performs one iteration of the particle filter

        Parameters
        ----------
        img :
            camera view image of the scene
        shape_masks :
            images containing the masks of currently tracked bodies
        feature_positions :
            current positions of features
        worker_pool :
            threaded worker pool to perform particle computations parallelized (Default value = None)

        Returns
        -------
        type
            the output of the segmentation algorithm (so it can be sent to the visualizer)

        """
        # resmaple
        particle_set = self.measurement_model.resample_particles(particle_set)
        # updating all measurement info
        labeled_img = self.segmenter.get_labeled_img(img)
        contour_img = convert_label_to_boundary(labeled_img)
        self.forward_model.set_optical_flow_values(optical_flow[:, :, 0], optical_flow[:, :, 1])
        self.measurement_model.update_measurment_information(contour_img=contour_img, labeled_img=labeled_img)
        # computing the changes in particle state and the weights
        if worker_pool is None:
            particle_set = [self._compute_singular_particle_iteration(p) for p in particle_set]
        else:
            particle_set = list(worker_pool.map(self._compute_singular_particle_iteration, particle_set))
        # normalize
        self.normalize_particle_set(particle_set)
        return particle_set

    def one_iteration_multiple_measurements(self, particle_set, measurements, optical_flow, worker_pool=None):
        # resmaple
        particle_set = self.measurement_model.resample_particles(particle_set)
        self.forward_model.set_optical_flow_values(optical_flow[:, :, 0], optical_flow[:, :, 1])
        # computing the changes in particle state and the weights
        if worker_pool is None:
            particle_set = [self._compute_singular_particle_forward_only(p) for p in particle_set]
        else:
            particle_set = list(worker_pool.map(self._compute_singular_particle_forward_only, particle_set))
        order = list(range(len(measurements)))
        # random.shuffle(order)
        for i in order:
            m = measurements[i]
            if np.sum(m) == 0 or (self.measurement_weights[i]==0 and self.measurement_spawning_weight[i]==0):
                # print("Measurement "+str(i) + " is zero!")
                continue
            contour_img = convert_label_to_boundary(m)
            self.measurement_model.update_measurment_information(contour_img=contour_img, labeled_img=m, measurement_weight=self.measurement_weights[i], measurement_spawning_weight=self.measurement_spawning_weight[i])
            if worker_pool is None:
                particle_set = [self._compute_singular_particle_meas_update_only(p) for p in particle_set]
            else:
                particle_set = list(worker_pool.map(self._compute_singular_particle_meas_update_only, particle_set))
        # normalize
        self.normalize_particle_set(particle_set)

        return particle_set

    def _compute_singular_particle_iteration(self, particle):
        """Computes a singular iteration except resampling for a singular particle

        Parameters
        ----------
        particle :
            the particle

        Returns
        -------
        type
            the particle, but change also happens in place

        """
        self.forward_model.predict_step_single_particle(particle=particle)
        self.measurement_model.informed_update_and_scoring_for_singular_particle(particle=particle)
        return particle

    def _compute_singular_particle_meas_update_only(self, particle):
        self.measurement_model.informed_update_and_scoring_for_singular_particle(particle=particle)
        return particle

    def _compute_singular_particle_forward_only(self, particle):
        self.forward_model.predict_step_single_particle(particle=particle)
        return particle

    def normalize_particle_set(self, particle_set):
        sum = 0.0
        for p in particle_set:
            sum += p.curr_weight
        for p in particle_set:
            p.curr_weight = p.curr_weight / sum

    def get_highest_weight_particle(self, particle_set):
        """Returns the currently highest weighted particle
        
        :return: the highest weighted particle

        Parameters
        ----------

        Returns
        -------

        """
        return get_highest_weighted_particle(particle_set)

    def set_segmenter(self, segmenter):
        self.segmenter = segmenter

    def update_image_shape(self, shape):
        self.forward_model.set_img_shape(shape=shape)

    def get_smallest_intersected_segmentation(self, particle_set):
        combined = np.zeros(particle_set[0].get_labeled_img().shape, np.float64)
        for p in particle_set:
            curr_highest = np.max(combined)
            labeled = p.get_labeled_img().copy()
            labeled = labeled + curr_highest - np.min(labeled)
            combined += labeled
            combined -= np.min(combined)
        return combined
