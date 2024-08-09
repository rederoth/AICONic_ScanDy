import numpy as np
import random
import segmentation_particle_filter.utils as utils


class MeasurementModel:
    """The measurement model of the particle filter, which performs informed update, weighting and resampling"""

    def __init__(self, particle_num, gradient_descent_steps=3, half_max_prob_normalized_size=0.01, max_spawn_prob=0.1,
                 overspawn_prob=0.4, dist_map_thresh=10):
        """
        Initializes the Measurement model

        Parameters
        ----------
        particle_num : int
            number of particles used
        gradient_descent_steps : int
            amount of times that the gradient descent steps should be performed
        half_max_prob_normalized_size : float
            normalized size for which the spawning probability should be half of the maximum
        max_spawn_prob : float
            maximum possible probability with which to spawn a segment
        dist_map_thresh : float
            threshold after which distance (in pixels) the gradient descent should not be applied anymore
        overspawn_prob : float
            probability that in case of spawning the boundaries in that area are kept or "overspawned"
        """
        self.particle_num = particle_num
        self.half_max_prob_normalized_size = half_max_prob_normalized_size
        self.max_spawn_prob = max_spawn_prob
        self.steepness = 0.5 * max_spawn_prob / half_max_prob_normalized_size  # continous and differentiable in x0
        self.gradient_descent_steps = gradient_descent_steps
        self.dist_map_thresh = dist_map_thresh
        self.overspawn_prob = overspawn_prob
        self.measurement_weight = 1.0
        self.measurement_spawning_weight = 1.0
        # maps and masks for informed updates and score computation
        self.measured_border_img = None
        self.dist_map = None
        self.spawn_masks = []
        self.spawn_probabilities = []
        self.gradient_masks = []

    def correction_step(self, particle_set, contour_img, labeled_img):
        """Performs the complete correction step for the particle set using the observed segmentation

        Parameters
        ----------
        particle_set : list[Particle]
            list containing the particles
        contour_img : ndarray
            binary boundary image for the observed segmentation
        labeled_img : ndarray
            labelled image for the observed segmentation

        Returns
        -------
        list[Particle]
            list containing the particles after the correction step

        """
        self.update_measurment_information(contour_img, labeled_img)
        particle_set = self.perform_informed_update_and_compute_weights(particle_set)
        final_set = self.resample_particles(particle_set)
        return final_set

    def perform_informed_update_and_compute_weights(self, particle_set):
        """Performs informed update and weighting for all particles in the set

        Parameters
        ----------
        particle_set : list[Particle]
            list containing the particles

        Returns
        -------
        list[ndarray]
            list of the particles after the step

        """
        for particle in particle_set:
            self.informed_update_and_scoring_for_singular_particle(particle)
        return particle_set

    def informed_update_and_scoring_for_singular_particle(self, particle):
        """Performs informed update and weighting for one particle

        In place change in the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None
        """
        self.spawn_random_segments_for_particle(particle)
        for _ in range(self.gradient_descent_steps):
           self.perform_singular_informed_update(particle)
        self.compute_unnormalized_weight(particle)

    def update_measurment_information(self, contour_img, labeled_img, measurement_weight=1.0, measurement_spawning_weight=1.0):
        """Updates the information of the model about the last observation (has to be done each step)

        Parameters
        ----------
        contour_img : ndarray
            binary boundary image for the observed segmentation
        labeled_img : ndarray
            labelled image for the observed segmentation

        Returns
        -------
        None
        """
        self.measurement_weight = measurement_weight
        self.measurement_spawning_weight = measurement_spawning_weight
        self.update_spawn_masks(labeled_img=labeled_img)
        self.update_distance_map(border_img=contour_img)

    def update_spawn_masks(self, labeled_img):
        """Updates the spawn masks and spawn probabilities according to the last observation

        Parameters
        ----------
        labeled_img : ndarray
            labelled image for the observed segmentation

        Returns
        -------
        None
        """
        measured_img = labeled_img
        max_label = int(np.max(measured_img)) + 1
        self.spawn_masks = []
        self.spawn_probabilities = []
        for i in range(1, max_label):
            mask = utils.get_label_mask(measured_img, i)
            self.spawn_masks.append(mask)
            normalized_size = np.sum(mask) / labeled_img.shape[0] / labeled_img.shape[1]
            probability = self.compute_spawn_probability(normalized_size)
            self.spawn_probabilities.append(probability)

    def compute_spawn_probability(self, normalized_size):
        """Computes the spawn probability for one segment using its normalized size

        Parameters
        ----------
        normalized_size : float
            the normalized size (percentage of image area) of the segment

        Returns
        -------
        float
            the probability

        """
        # self.steepness = 0.5 * self.measurement_spawning_weight / self.half_max_prob_normalized_size
        # if normalized_size < self.half_max_prob_normalized_size:
        #     # linear in [0, x0[
        #     probability = self.steepness * normalized_size
        # else:
        #     # logistic in [x0, inf[
        #     probability = self.measurement_spawning_weight/ (
        #             1 + np.exp(- self.steepness * (normalized_size - self.half_max_prob_normalized_size)))
        return self.measurement_spawning_weight

    def update_distance_map(self, border_img):
        """Updates the distance map based on the last observation

        Parameters
        ----------
        border_img : ndarray
            binary measured boundary image of the observation

        Returns
        -------
        None
        """
        self.measured_border_img = border_img
        self.dist_map, self.gradient_masks = utils.generate_dist_map_and_movement_masks(border_img, self.dist_map_thresh)

    def spawn_random_segments_for_particle(self, particle):
        """Spawns segments in the particle based on their probabilities

        In place change in the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None
        """
        for i in range(len(self.spawn_probabilities)):
            if np.random.ranf() < self.spawn_probabilities[i]:
                # if np.random.ranf() < self.overspawn_prob:
                #     particle.split_segments(self.spawn_masks[i])
                #     particle.merge_segments(self.spawn_masks[i])
                # else:
                #     particle.merge_segments(self.spawn_masks[i])
                particle.spawn_segment(self.spawn_masks[i])

    def perform_singular_informed_update(self, particle):
        """Performs informed update of the segmentation (moving the boundaries closer to the observed boundaries)

        In place change in the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None
        """
        masks = list(self.gradient_masks)
        random.shuffle(masks)
        particle.apply_movement_masks_without_velocity_generation(masks)

    def compute_unnormalized_weight(self, particle):
        """Computes the unnormalized weight for the particle based on its distance to the observed segmentation

        In place change in the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None
        """
        border_img = particle.get_boundary_img()
        distances_p_m = np.multiply(border_img, self.dist_map)  # distances in a map
        dist_map_particle = utils.get_dist_map_for_border_img(border_img)
        distances_m_p = np.multiply(self.measured_border_img, dist_map_particle)  # distances in a map
        particle.curr_weight = np.power(1 / (np.sum(distances_p_m) + 1) / (np.sum(distances_m_p) + 1), self.measurement_weight) * particle.curr_weight

    def resample_particles(self, particle_set):
        """Resamples the particles based on their unnormalized weights

        Parameters
        ----------
        particle_set : list[Particle]
            list containing the particles

        Returns
        -------
        list[Particle]
            the resampled set

        """
        weights = [p.curr_weight for p in particle_set]
        total_score = np.sum(weights)
        final_set = []
        limit = np.random.ranf() / self.particle_num * total_score
        curr_weight = weights[0]
        i = 0
        nr = 0
        for _ in range(self.particle_num):
            while limit > curr_weight:
                i += 1
                nr = 0
                curr_weight += weights[i]
                particle_set[i].curr_weight /= total_score
            if nr == 0:
                final_set.append(particle_set[i])
                final_set[-1].curr_weight = 1.0
            else:
                final_set.append(particle_set[i].__copy__())
                final_set[-1].curr_weight = 1.0
            nr += 1
            limit += total_score / self.particle_num
        return final_set
