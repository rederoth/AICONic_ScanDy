import random
import numpy as np
import cv2 as cv
import segmentation_particle_filter.utils as utils


class ForwardModel:
    """Forward Model used by the particle filter, applies velocities and noise for each particle"""

    def __init__(self, img_shape, velocity_apply_number=1, noise_smoothing_sigma=5.0, dist_map_thresh=10):
        """
        Initializes the forward model with its parameters

        Parameters
        ----------
        img_shape : tuple[int]
            image shape in which the particles keep the segmentation
        velocity_apply_number : int
            amount of times that the velocities should be applied
        noise_smoothing_sigma : float
            standard deviation used for the gaussian blurring of the random sampled image
            used to apply noise to each particles segmentation
        dist_map_thresh : int
            threshold after which distance (in pixels) the gradient descent should not be applied anymore
        """

        self.velocity_apply_number = velocity_apply_number
        self.noise_smoothing_sigma = noise_smoothing_sigma
        self.radius = 10
        self.img_shape= img_shape  # np compatible not cv
        self.split_based_on_mask_prob = 0.3
        self.merge_based_on_mask_prob = 0.5
        self.dist_map_thresh = dist_map_thresh
        self.masks = {}

    def predict_step(self, particle_set):
        """Performs the predict step for all particles in the set

        Parameters
        ----------
        particle_set : list[Particle]
            a list of particles

        Returns
        -------
        list[Particle]
            list of the changed particles

        """
        for particle in particle_set:
            self.predict_step_single_particle(particle)
        return particle_set

    def predict_step_single_particle(self, particle):
        """Performs predict step for singular particle

        In place change of the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None

        """
        self.apply_velocities(particle)
        # self.apply_process_noise(particle)

    def apply_process_noise(self, particle):
        """Applies noise to the singular particle by sampling a random image and generating velocities based on its
        gradient

        In place change of the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None

        """
        particle_shape = particle.labeled_img.shape
        noise_sampling_shape = (int(np.round(particle_shape[0]*0.15)), int(np.round(particle_shape[1]*0.15)))
        # TODO make dependent on optical flow velocities
        random_image = np.random.randint(255, size=noise_sampling_shape, dtype=np.uint8)
        random_image = cv.resize(random_image, utils.shape_convert_np_cv2(particle_shape), interpolation=cv.INTER_NEAREST)
        smoothed_img = cv.GaussianBlur(random_image, ksize=(11, 11), sigmaX=self.noise_smoothing_sigma)
        masks = utils.generate_movement_masks_for_dist_img(smoothed_img)
        random.shuffle(masks)
        particle.apply_movement_masks_without_velocity_generation(masks)

    def apply_velocities(self, particle):
        """Applies the velocities of the particle to its segmentation

        In place change of the particle

        Parameters
        ----------
        particle : Particle
            the particle

        Returns
        -------
        None

        """
        for i in range(len(self.masks.keys())):
            masks = list(self.masks[i])
            random.shuffle(masks)
            particle.apply_movement_masks_without_velocity_generation(masks)

    def set_img_shape(self, shape):
        self.img_shape = shape

    def set_optical_flow_values(self, optical_flow_velocitiy_x, optical_flow_velocitiy_y):
        self.masks = {}
        for i in range(self.velocity_apply_number):
            _, x_p = cv.threshold(optical_flow_velocitiy_x, 0.1, 1, cv.THRESH_BINARY)
            _, x_n = cv.threshold(optical_flow_velocitiy_x, -0.1, 1, cv.THRESH_BINARY_INV)
            _, y_p = cv.threshold(optical_flow_velocitiy_y, 0.1, 1, cv.THRESH_BINARY)
            _, y_n = cv.threshold(optical_flow_velocitiy_y, -0.1, 1, cv.THRESH_BINARY_INV)
            if np.all(np.logical_and(np.logical_and(x_n == 0, x_p == 0), np.logical_and(y_n == 0, y_p == 0))):
                break
            self.masks[i] = utils.generate_4_directions_masks(x_n, x_p, y_n, y_p)

            optical_flow_velocitiy_x_new = np.zeros_like(optical_flow_velocitiy_x)
            optical_flow_velocitiy_y_new = np.zeros_like(optical_flow_velocitiy_y)

            M = np.float32([[1, 0, 1], [0, 1, 0]])
            img_part = np.multiply(optical_flow_velocitiy_x - 1, optical_flow_velocitiy_x > 1)
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            optical_flow_velocitiy_x_new += moved_img_part

            M = np.float32([[1, 0, -1], [0, 1, 0]])
            img_part = np.multiply(optical_flow_velocitiy_x + 1, optical_flow_velocitiy_x < -1)
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            optical_flow_velocitiy_x_new += moved_img_part

            M = np.float32([[1, 0, 0], [0, 1, 1]])
            img_part = np.multiply(optical_flow_velocitiy_y - 1, optical_flow_velocitiy_y > 1)
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            optical_flow_velocitiy_y_new += moved_img_part

            M = np.float32([[1, 0, 0], [0, 1, -1]])
            img_part = np.multiply(optical_flow_velocitiy_y + 1, optical_flow_velocitiy_y < -1)
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            optical_flow_velocitiy_y_new += moved_img_part

            optical_flow_velocitiy_x = optical_flow_velocitiy_x_new
            optical_flow_velocitiy_y = optical_flow_velocitiy_y_new

