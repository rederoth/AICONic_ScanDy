import cv2 as cv
import numpy as np
import segmentation_particle_filter.utils as utils
import sklearn.cluster as skcluster
import matplotlib.pyplot as plt
import rospy, seaborn


################################################################################################
# TODO
# TODO      This is kept as a reference for now, definitely not working right now though
# TODO
################################################################################################

class Visualizer:
    """Visualization class for the SPF"""

    def __init__(self, src_shape, entropy_epsilon=0.0001, clustering_epsilon=1.0,
                 clustering_min_samples=4, variance_sigma=3, action_radius=25, action_timesteps=None,
                 action_centerpoints=None):
        """
        Initializes parameters, clusterer and videofiles

        Parameters
        ----------
        src_shape : tuple[int]
            shape of the indivdual images (goal shape of the SPF, not original)
        entropy_epsilon : float
            epsilon used in the entropy to prevent log(0)
        variance_sigma : float
            std dev used for the blurring in the variance image
        action_radius : float
            radius used for action uncertainty compuation
        action_timesteps : list[int]
            list of timesteps where actions take place
        action_centerpoints : list[tuple[int]]
            list of center points of localized actions

        clustering_epsilon: epsilon used by the pairwise clustering algorithm
        clustering_min_samples: minimum samples to be one cluster
        """


        self.color_LUT = np.expand_dims(
            np.asarray(seaborn.diverging_palette(10.9, 10.9, s=96.1, l=41.7, n=int(256 * 2)))[256:, :3],
            0)  # r : 10,9, 96.1, 41.7; b: 229.3, 94.6, 40.1; g: 0, 0, 72.6
        self.color_LUT[:, :, [0, 2]] = self.color_LUT[:, :, [2, 0]]  # bgr

        # params
        self.src_shape = src_shape
        self.entropy_epsilon = entropy_epsilon
        self.variance_sigma = variance_sigma

        # to record uncertainty over time
        self.action_radius = action_radius
        self.action_timesteps = action_timesteps if action_timesteps is not None else []
        self.action_center_points = action_centerpoints if action_centerpoints is not None else []
        self.action_masks = [np.ones((src_shape[1], src_shape[0]), dtype=np.float32)]
        for point in self.action_center_points:
            mask_img = np.zeros((src_shape[1], src_shape[0]), dtype=np.float32)
            cv.circle(mask_img, center=point, radius=action_radius, color=1, thickness=-1)
            self.action_masks.append(mask_img)
        self.tracked_uncertainties = [[] for _ in range(len(self.action_masks))]
        self.last_uncertainty_img = None

        # internal timestep counter
        self.internal_timestep = 0

        # deprecated clustering stuff
        self.clusterer = skcluster.DBSCAN(metric="precomputed", eps=clustering_epsilon,
                                          min_samples=clustering_min_samples)
        # other off the shelf pairwise clustering options, were not investigated further
        # Spectral Clustering
        # self.clustering = skcluster.SpectralClustering(affinity="precomputed", n_clusters=8)
        # Affinity Propagation
        # self.clustering = skcluster.AffinityPropagation(affinity="precomputed", damping=0.95)

        # to record clustering info
        self.t = []
        self.number_clusters = []
        self.number_noise = []
        self.cluster_sizes = []

    def update_taken_actions(self, action_timesteps, action_centerpoints):
        """Updates taken actions (timestep and centerpoint) after initialization

        Parameters
        ----------
        action_timesteps : list[int]
            list of timesteps where actions take place
        action_centerpoints : list[tuple[int]]
            list of center points of localized actions

        """
        self.action_timesteps = action_timesteps if action_timesteps is not None else []
        self.action_center_points = action_centerpoints if action_centerpoints is not None else []
        self.action_masks = [np.ones((self.src_shape[1], self.src_shape[0]), dtype=np.float32)]
        for point in self.action_center_points:
            mask_img = np.zeros((self.src_shape[1], self.src_shape[0]), dtype=np.float32)
            cv.circle(mask_img, center=point, radius=self.action_radius, color=1, thickness=-1)
            self.action_masks.append(mask_img)
        self.tracked_uncertainties = [[] for _ in range(len(self.action_masks))]

    def create_visualization(self, contour_imgs, orig_contour, orig_src, weights, step, biased_labeled, grouping_beliefs, labeled_imgs, rb_labels):
        """Creates all possible visualizations (no clustering)

        Parameters
        ----------
        orig_src : ndarray
            original source image (but goal shape resolution)
        orig_contour : ndarray
            original boundary image (in goal shape)
        contour_imgs : list[ndarray]
            list of boundary images (of the particle set)
        weights : list[float]
            list of weights (of the particle set)
            currently only used for highest weighted particle as the set is visualized after resampling

        Returns
        -------
        list[ndarray]
            all generated images: orig_float_mean, mean, (src, mean), (src, boundary, mean, entropy),
            (source, mean, entropy, uncertainty), (src, boundary, mean, highest weighted particle),
            (src, boundary, mean, uncertainty), entropy, overlayed highest + src, variance

        """
        biased_contour = utils.generate_boundary_img(biased_labeled)
        mean_img = self.create_mean_img(contour_imgs, weights)
        orig_float_mean = mean_img.astype(np.float32)
        entropy_img = self.entropy_img_of_mean(mean_img)
        variance_array = self.get_uncertainty_image(contour_imgs, weights)
        max_weight_contour_img, overlayed_max_weight_contour_img = self.get_highest_weighted_imgs(contour_imgs,
                                                                                                  weights, orig_src)
        rb_segmentation, rb_association = self.create_rb_segmentation_img(grouping_beliefs, labeled_imgs, weights, rb_labels)

        # make into right formats
        mean_img = self.float_gray_to_colormapped(mean_img)
        entropy_img = self.float_gray_to_colormapped(entropy_img)
        variance_img = self.float_gray_to_colormapped(variance_array)
        biased_contour = self.float_gray_to_colormapped(biased_contour)
        # mark maximum
        #self.mark_max_positions(variance_array, variance_img, orig_src)
        #self.note_timestep(orig_src, step)
        combined2 = self.combine_2_to_1([orig_src, mean_img])
        combined2_biased = self.combine_2_to_1([orig_src, biased_contour])
        combined4_h = self.combine_4_to_1([orig_src, mean_img, orig_contour, max_weight_contour_img])
        combined4_e = self.combine_4_to_1([orig_src, mean_img, orig_contour, entropy_img])
        combined4_v = self.combine_4_to_1([orig_src, mean_img, orig_contour, variance_img])
        combined4_ev = self.combine_4_to_1([orig_src, mean_img, entropy_img, variance_img])
        particle_view = self.create_particle_view(contour_imgs, weights, orig_src, orig_contour)
        return orig_float_mean, mean_img, combined2, combined4_e, combined4_ev, combined4_h, combined4_v, entropy_img,\
               overlayed_max_weight_contour_img, variance_img, particle_view, biased_contour, combined2_biased, rb_segmentation, rb_association

    def transform_mean_into_single_segmentation(self, mean_img):
        # mean_img = cv.GaussianBlur(mean_img, (7,7), 0.25)
        _, mean_img = cv.threshold(mean_img, 0.5, 255, cv.THRESH_BINARY)
        mean_img = mean_img.astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        mean_img = cv.morphologyEx(mean_img, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(mean_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        mean_img = np.zeros(mean_img.shape, np.uint8)
        cv.drawContours(mean_img, contours, -1, 255, 2)
        mean_img = cv.morphologyEx(mean_img, cv.MORPH_CLOSE, kernel)
        mean_img = cv.morphologyEx(mean_img, cv.MORPH_ERODE, kernel)
        return mean_img.astype(np.float32) / 255.0

    def create_rb_segmentation_img(self, grouping_beliefs, labeled_imgs, weights, rb_labels):
        rb_grouping_img = self.create_rb_groupings_segmentation(grouping_beliefs, labeled_imgs, weights)
        # rb_segmentation = np.take(rb_labels, np.argmax(rb_grouping_img, axis=2))
        color_lut = np.random.randint(0, 255, (rb_labels.shape[0], 3), dtype=np.uint8)
        colored_segmentation = np.take(color_lut, np.argmax(rb_grouping_img, axis=2), axis=0)
        legend_strip = np.zeros((colored_segmentation.shape[0], int(colored_segmentation.shape[1] * 0.25), 3), dtype=np.uint8)
        one_number_size = int(float(colored_segmentation.shape[0]) / float((rb_labels.shape[0] + 1)))
        cv.putText(legend_strip, "RB", (0,one_number_size), cv.FONT_HERSHEY_SIMPLEX, 0.2, color=(255, 255, 255))
        for i, rb_id in enumerate(rb_labels):
            legend_strip[int(one_number_size*(i+1)):int(one_number_size*(i+2)), int(colored_segmentation.shape[1] * 0.1):] = color_lut[i,:]
            cv.putText(legend_strip, str(rb_id), (0, int(one_number_size*(i+2))), cv.FONT_HERSHEY_SIMPLEX, 0.2, color=(255, 255, 255))
        colored_segmentation = np.hstack((colored_segmentation, legend_strip))
        return colored_segmentation, (rb_labels, rb_grouping_img)

    def create_rb_groupings_segmentation(self, grouping_beliefs, labeled_imgs, weights):
        over_all_rb_likelihoods = np.zeros((labeled_imgs[0].shape[0], labeled_imgs[0].shape[1], grouping_beliefs[0].shape[1]), dtype=np.float64)
        for i, belief in enumerate(grouping_beliefs):
            rb_likelihoods = np.take(belief, labeled_imgs[i].astype(np.int64), axis=0)
            over_all_rb_likelihoods += rb_likelihoods * weights[i]
        return over_all_rb_likelihoods

    def make_uncertainty_plots(self):
        """Makes global cumulative uncertainty plot and local ones for all action centerpoints and saves them"""
        # TODO fix dis with new way of saving and publishing
        # for i, mask in enumerate(self.action_masks):
        #     cum_uncertainty_normalized = np.sum(mask * self.last_uncertainty_img) / np.sum(mask)
        #     self.tracked_uncertainties[i].append(cum_uncertainty_normalized)
        #     plt.plot(range(len(self.tracked_uncertainties[i])), self.tracked_uncertainties[i], color="#C50e1F")
        #     for t in self.action_timesteps:
        #         plt.axvline(t, color='#146683', linestyle='--')
        #     plt.xlabel("Timestep")
        #     if i == 0:
        #         plt.ylabel("Global average uncertainty per pixel")
        #         plt.savefig(self.folder_str + "imgs/uncertainty_development_global.png")
        #     else:
        #         plt.ylabel("Average uncertainty per pixel in action window of action " + str(i))
        #         plt.savefig(self.folder_str + "imgs/uncertainty_development_action_" + str(i) + ".png")
        #     plt.close()

    def create_particle_view(self, contour_imgs, weights, orig_src, meas_contour):
        i = np.argmax(weights)
        max_weight_contour_img = contour_imgs[i]
        max_weight_contour_img = self.float_gray_to_char_bgr(max_weight_contour_img)
        i = np.argmin(weights)
        min_weight_contour_img = contour_imgs[i]
        min_weight_contour_img = self.float_gray_to_char_bgr(min_weight_contour_img)
        i = weights.index(np.percentile(weights, 66, interpolation='nearest'))
        per66_weight_contour_img = contour_imgs[i]
        per66_weight_contour_img = self.float_gray_to_char_bgr(per66_weight_contour_img)
        i = weights.index(np.percentile(weights, 33, interpolation='nearest'))
        per33_weight_contour_img = contour_imgs[i]
        per33_weight_contour_img = self.float_gray_to_char_bgr(per33_weight_contour_img)
        img = self.combine_6_to_1([orig_src, max_weight_contour_img, per66_weight_contour_img, meas_contour, per33_weight_contour_img, min_weight_contour_img])
        shape = contour_imgs[0].shape
        cv.putText(img, "Source", (0, 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv.putText(img, "Heighest Weight", (shape[1] +1, 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv.putText(img, "66 percentile Weight", (shape[1] *2 +2, 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv.putText(img, "Measured Contour", (0, shape[0]+1 + 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv.putText(img, "33 percentile Weight", (shape[1] +1, shape[0]+1+ 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv.putText(img, "Lowest Weight", (shape[1] *2 +2, shape[0]+1+ 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        return img

    def entropy_img_of_mean(self, mean):
        """Constructs entropy image based on the mean image

        Parameters
        ----------
        mean : ndarray
            mean image

        Returns
        -------
        ndarray
            entropy image

        """
        border_likelihood = mean + self.entropy_epsilon
        no_border_likelihood = 1 - mean + self.entropy_epsilon
        log_likelihood_border = np.log2(border_likelihood)
        log_likelihood_no_border = np.log2(no_border_likelihood)
        entropy = - border_likelihood * log_likelihood_border - no_border_likelihood * log_likelihood_no_border
        return entropy

    def get_highest_weighted_imgs(self, contour_imgs, weights, orig_src):
        """Gets highest weighted particle in img and overlayed image

        Parameters
        ----------
        contour_imgs : list[ndarray]
            list of boundary images of the particle set
        weights : list[float]
            list of weights of the set
        orig_src : ndarray
            original source image (goal shape)

        Returns
        -------
        ndarray, ndarray
            highest weighted boundary image, weighted boundary image overlayed on source

        """
        i = np.argmax(weights)
        max_weight_contour_img = contour_imgs[i]
        max_weight_contour_img = self.float_gray_to_char_bgr(max_weight_contour_img)
        overlayed_img = self.overlay_borders_on_img(orig_src, max_weight_contour_img)
        return max_weight_contour_img, overlayed_img

    def note_timestep(self, img, step):
        """Puts timestep  on the source image and increases internal timestep counter

        Parameters
        ----------
        img : ndarray
            source image (manipulated in place)
        step : int
            timestep of this frame

        """
        cv.putText(img, str(step), org=(int(self.src_shape[0] * 0.1), int(self.src_shape[1] * 0.1)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 255))

    def get_uncertainty_image(self, contour_imgs, weights):
        """Creates uncertainty image based on variance of smoothed boundary images of the set

        Parameters
        ----------
        contour_imgs : list[ndarray]
            list of boundary images (of the particle set)
        weights : list[ndarray]
            list of weights (of the particle set)

        Returns
        -------
        ndarray
            uncertainty image

        """
        img = np.zeros(contour_imgs[0].shape, np.float32)
        smoothed_contour_imgs = [cv.GaussianBlur(img, ksize=(11, 11), sigmaX=self.variance_sigma) for img in
                                 contour_imgs]
        alpha_img = self.create_mean_img(smoothed_contour_imgs, weights)
        total_alphas = 0.0
        for i in range(len(weights)):
            tmp = np.square(smoothed_contour_imgs[i] - alpha_img)
            img = cv.addWeighted(tmp, weights[i], img, 1, 0)
            total_alphas += weights[i]
        img = img / total_alphas
        max = np.max(img)
        self.last_uncertainty_img = img.copy()
        rospy.loginfo("Max current uncertainty " + str(max))
        img = img / (0.15 if max < 0.15 else max)
        return img

    def mark_max_positions(self, uncertainty_array, uncertainty_img, src):
        """ Marks Position of the maximum uncertainty in source image and uncertainty image with a circle (in place)

        Parameters
        ----------
        uncertainty_array : ndarray
            raw array version of the uncertainty
        uncertainty_img : ndarray
            colormapped uncertainty image
        src : ndarray
            original source image (in goal shape)

        """
        idx = np.argmax(uncertainty_array)
        idx_ax1 = int(idx % uncertainty_array.shape[1])
        idx_ax0 = int(idx / uncertainty_array.shape[1])
        cv.circle(uncertainty_img, (idx_ax1, idx_ax0), 5, (131,102,20), 1)
        cv.circle(src, (idx_ax1, idx_ax0), 5, (131,102,20), 1)

    @staticmethod
    def create_mean_img(contour_imgs, weights):
        """Creates mean image of the resampled particle set based on the boundary images

        Parameters
        ----------
        contour_imgs : list[ndarray]
            list of boundary images (of the resampled particle set)

        Returns
        -------
        ndarray
            mean img

        """
        img = np.zeros(contour_imgs[0].shape, np.float64)
        total_alphas = 0.0
        for i in range(len(contour_imgs)):
            img = cv.addWeighted(contour_imgs[i].astype(np.float64), weights[i], img, 1, 0)
            total_alphas += weights[i]
        img = img / total_alphas
        return img.astype(np.float32)

    @staticmethod
    def get_particle_weights_and_contours(particle_set):
        """Generates a separated boundary image list and a weight list for a list of particles

        Parameters
        ----------
        particle_set : list[Particle]
            list of particles

        Returns
        -------
        list[ndarray], list[float]
            list of boundary imgs, list of weights

        """
        return [p.get_boundary_img().astype(np.float32) for p in particle_set], [p.curr_weight for p in particle_set]

    @staticmethod
    def overlay_borders_on_img(src_img, boundary_img):
        """Overlays a boundary image on another image

        Parameters
        ----------
        src_img : ndarray
            the imae to overlay boundaries on
        boundary_img : ndarray
            the boundary image

        Returns
        -------
        ndarray
            the overlayed image

        """
        _, mask = cv.threshold(boundary_img, 0, 1, cv.THRESH_BINARY_INV)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        new_img = cv.add(boundary_img, src_img, dst=boundary_img, mask=mask)
        return new_img

    @staticmethod
    def float_gray_to_char_bgr(img):
        """Converts single channel float (0-1) gray image to bgr image

        Parameters
        ----------
        img : ndarray
            grey image

        Returns
        -------
        ndarray
            bgr img

        """
        new_img = img * 255
        new_img = cv.cvtColor(new_img.astype(np.uint8), cv.COLOR_GRAY2BGR)
        return new_img

    def float_gray_to_colormapped(self, img):
        """Converts single channel float (0-1) gray image to bgr image and applies the color map to it

        Parameters
        ----------
        img : ndarray
            grey image

        Returns
        -------
        ndarray
            color mapped bgr img

        """
        new_img = self.float_gray_to_char_bgr(img)
        new_img = cv.LUT(new_img, self.color_LUT) * 255
        return new_img.astype(np.uint8)

    @staticmethod
    def combine_4_to_1(imgs):
        """Combines 4 same size multichannel images into one (top left, top right, bottom left, bottom right)

        Parameters
        ----------
        imgs : list[ndarray]
            list of 4 images

        Returns
        -------
        ndarray
            combined img

        """
        if len(imgs[0].shape) == 3:  # multichannel
            new_img = np.ones((imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] * 2 + 1, imgs[0].shape[2]),
                              imgs[0].dtype) * 255
        else:
            new_img = np.ones((imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] * 2 + 1), imgs[0].dtype) * 255

        new_img[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
        new_img[:imgs[0].shape[0], imgs[0].shape[1] + 1:imgs[0].shape[1] * 2 + 1] = imgs[1]
        new_img[imgs[0].shape[0] + 1:imgs[0].shape[0] * 2 + 1, :imgs[0].shape[1]] = imgs[2]
        new_img[imgs[0].shape[0] + 1:imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] + 1:imgs[0].shape[1] * 2 + 1] = imgs[3]
        return new_img

    @staticmethod
    def combine_6_to_1(imgs):
        """Combines 6 same size multichannel images into one (top left, top right, bottom left, bottom right)

        Parameters
        ----------
        imgs : list[ndarray]
            list of 6 images

        Returns
        -------
        ndarray
            combined img

        """
        if len(imgs[0].shape) == 3:  # multichannel
            new_img = np.ones((imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] * 3 + 2, imgs[0].shape[2]),
                              imgs[0].dtype) * 255
        else:
            new_img = np.ones((imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] * 3 + 2), imgs[0].dtype) * 255

        new_img[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
        new_img[:imgs[0].shape[0], imgs[0].shape[1] + 1:imgs[0].shape[1] * 2 + 1] = imgs[1]
        new_img[:imgs[0].shape[0], imgs[0].shape[1] * 2 + 2:imgs[0].shape[1] * 3 + 2] = imgs[2]
        new_img[imgs[0].shape[0] + 1:imgs[0].shape[0] * 2 + 1, :imgs[0].shape[1]] = imgs[3]
        new_img[imgs[0].shape[0] + 1:imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] + 1:imgs[0].shape[1] * 2 + 1] = imgs[4]
        new_img[imgs[0].shape[0] + 1:imgs[0].shape[0] * 2 + 1, imgs[0].shape[1] * 2 + 2:imgs[0].shape[1] * 3 + 2] = imgs[5]
        return new_img

    @staticmethod
    def combine_2_to_1(imgs):
        """Combines 2 same size images into one (left, right)

        Parameters
        ----------
        imgs : list[ndarray]
            list of 2 images

        Returns
        -------
        ndarray
            combined img

        """
        if len(imgs[0].shape) == 3:  # multichannel
            new_img = np.ones((imgs[0].shape[0], imgs[0].shape[1] * 2 + 1, imgs[0].shape[2]), imgs[0].dtype) * 255
        else:
            new_img = np.ones((imgs[0].shape[0], imgs[0].shape[1] * 2 + 1), imgs[0].dtype) * 255

        new_img[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
        new_img[:imgs[0].shape[0], imgs[0].shape[1] + 1:imgs[0].shape[1] * 2 + 1] = imgs[1]
        return new_img

    def __del__(self):
        """
        Destructor
        """
        del self.clusterer

    ####### Deprecated methods based on pairwise distance matrix #################

    def create_and_save_further_particle_set_visualization(self, orig_src, timestep, contour_imgs, weights):
        """DEPRECATED, clustering and furthest pair were not further investigated
        
        Creates and saves furthest pair images of the set and clustering results

        Parameters
        ----------
        orig_src :
            original source img of the step
        timestep :
            current timestep
        contour_imgs :
            list of boundary images (of the particle set)
        weights :
            list of weights (of the particle set)

        Returns
        -------

        """
        # src img writes
        cv.imwrite(self.folder_str + "imgs/furthest_pairs/" + "t" + str(timestep) + ".png", orig_src)
        cv.imwrite(self.folder_str + "imgs/clusters/" + "t" + str(timestep) + ".png", orig_src)
        # create dist matrix and create information
        distance_matrix = utils.build_pairwise_dist_matrix(contour_imgs)
        self.create_and_save_clustering(distance_matrix, timestep, contour_imgs, weights)
        self.create_and_save_furthest_pair(distance_matrix, timestep, contour_imgs)

    def create_and_save_clustering(self, distance_matrix, timestep, contour_imgs, weights):
        """DEPRECATED, clustering was not further investigated
        
        Creates clustering and saves its result

        Parameters
        ----------
        distance_matrix :
            pairwise distance matrix for the given particle set
        timestep :
            current timestep
        contour_imgs :
            list of boundary images (of the particle set)
        weights :
            list of weights (of the particle set)

        Returns
        -------

        """
        self.clusterer.fit(distance_matrix)

        n_clusters = len(set(self.clusterer.labels_)) - (1 if -1 in self.clusterer.labels_ else 0)
        n_noise = list(self.clusterer.labels_).count(-1)

        cluster_offset = (1 if -1 in self.clusterer.labels_ else 0)
        clusters = [([], []) for _ in range(n_clusters + cluster_offset)]
        for idx in range(len(contour_imgs)):
            clusters[self.clusterer.labels_[idx] + cluster_offset][0].append(contour_imgs[idx])
            clusters[self.clusterer.labels_[idx] + cluster_offset][1].append(weights[idx])
        clusters = clusters[cluster_offset:]

        for i, c in enumerate(clusters):
            contour_imgs_cluster, weights_cluster = c
            alpha_img = self.create_mean_img(contour_imgs_cluster)
            entropy_img = self.entropy_img_of_mean(alpha_img)
            alpha_img = self.float_gray_to_char_bgr(alpha_img)
            entropy_img = self.float_gray_to_char_bgr(entropy_img)
            combined = self.combine_2_to_1([alpha_img, entropy_img])
            cv.imwrite(self.folder_str + "imgs/clusters/" + "t" + str(timestep) + "_c" + str(i) + "_size_" + str(
                len(c)) + ".png", combined)

        cluster_sizes = [len(c[0]) for c in clusters]
        self.t.append(timestep)
        self.number_clusters.append(n_clusters)
        self.number_noise.append(n_noise)
        self.cluster_sizes.append(cluster_sizes)
        print("Clusters ", n_clusters, "\tNoise ", n_noise, "\tCluster sizes ", cluster_sizes)
        self.protocol_file.write("timestep " + str(timestep) + "\tClusters " + str(n_clusters) + "\tNoise " + str(
            n_noise) + "\tCluster sizes " + str(cluster_sizes) + "\n")

    def create_and_save_furthest_pair(self, distance_matrix, timestep, contour_imgs):
        """DEPRECATED, furthest pair was not further investigated
        
        Creates and saves furthest pair in the set

        Parameters
        ----------
        distance_matrix :
            pairwise distance matrix for the given particle set
        timestep :
            current timestep
        contour_imgs :
            list of boundary images (of the particle set)

        Returns
        -------

        """
        furthest_pair = self.get_furthest_pair_idxs(distance_matrix)
        contour_img1 = self.float_gray_to_char_bgr(contour_imgs[furthest_pair[0]])
        contour_img2 = self.float_gray_to_char_bgr(contour_imgs[furthest_pair[1]])
        combined = self.combine_2_to_1([contour_img1, contour_img2])
        cv.imwrite(self.folder_str + "imgs/furthest_pairs/" + "t" + str(timestep) + "_pair.png", combined)

    @staticmethod
    def get_furthest_pair_idxs(distance_matrix):
        """DEPRECATED, furthest pair was not further investigated
        
        Finds the indices of the pair which has the highest pairwise distance

        Parameters
        ----------
        distance_matrix :
            pairwise distance matrix

        Returns
        -------
        type
            index1, index2)

        """
        i = np.argmax(distance_matrix)
        p1 = int(i % len(distance_matrix[0]))
        p2 = int((i - p1) / len(distance_matrix[0]))
        return (p1, p2)

    def plot_number_clusters_and_noise(self):
        """DEPRECATED, clustering was not further investigated
        
        Plots number of clusters and noisy particles over the timesteps

        Parameters
        ----------

        Returns
        -------

        """
        fig, ax1 = plt.subplots()
        plt.title("Number of Clusters and Noise over time", fontweight='bold')  # , pad=20)

        color1 = 'tab:red'
        ax1.set_xlabel('timestep')
        ax1.set_ylabel("Number of Clusters", color=color1)
        ax1.grid(True)

        color2 = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Noise", color=color2)
        ax1.plot(self.t, self.number_clusters, color=color1)
        ax2.plot(self.t, self.number_noise, color=color2)
        plt.savefig(self.folder_str + "imgs/clusters/" + "development.png")
        plt.close()
