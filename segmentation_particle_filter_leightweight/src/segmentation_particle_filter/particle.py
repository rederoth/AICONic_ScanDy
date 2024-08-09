import numpy as np
import cv2 as cv
from segmentation_particle_filter.utils import convert_label_to_boundary


class Particle:
    """The singular particle containing of a labeled segmentation image, last steps velocities and a weight"""

    def __init__(self, labeled_img, curr_weight=1):
        """Initializes the particle with segmentation, weight and zero velocities

        Parameters
        ----------
        labeled_img: ndarray
            labeled segmentation image
        curr_weight: float
            weight of the particle
        """
        self.labeled_img = labeled_img
        self.velocity_x = np.zeros(self.labeled_img.shape, np.float32)
        self.velocity_y = np.zeros(self.labeled_img.shape, np.float32)
        self.curr_weight = curr_weight

    def get_boundary_img(self):
        """Gets the current segmentation as boundary image

        Returns
        -------
        ndarray
            boundary image of the segmentation
        """
        return convert_label_to_boundary(self.labeled_img.astype(np.float32))

    def get_labeled_img(self):
        return self.labeled_img.astype(np.float32)

    def apply_movement_masks_with_velocity_generation(self, masks):
        """Applies the given masks as movements to the segmentation and records the movement for the new velocities

        Changes labeled_img, velocity_x, velocity_y

        Parameters
        ----------
        masks : list[(ndarray, (int, int)]
            movement masks to be applied
            - list of touples containg a mask, and a (x,y) direction

        Returns
        -------
        None
        """
        new_img = np.zeros(self.labeled_img.shape, np.float64)
        history_img_x = np.zeros(self.labeled_img.shape, np.float32)
        history_img_y = np.zeros(self.labeled_img.shape, np.float32)
        for mask, direction in masks:
            img_part = np.multiply(self.labeled_img, mask)
            M = np.float32([[1, 0, direction[0]], [0, 1, direction[1]]])
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            moved_mask = cv.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            _, add_mask = cv.threshold(new_img, 0.000001, 1, cv.THRESH_BINARY_INV)
            add_mask = add_mask.astype(np.uint8)
            new_img = cv.add(new_img, moved_img_part, dst=new_img, mask=add_mask)
            history_img_x = cv.add(history_img_x, moved_mask*direction[0], dst=history_img_x, mask=add_mask)
            history_img_y = cv.add(history_img_y, moved_mask * direction[1], dst=history_img_y, mask=add_mask)

        _, add_mask = cv.threshold(new_img, 0.1, 1, cv.THRESH_BINARY_INV)
        new_img = cv.add(new_img, self.labeled_img, dst=new_img, mask=add_mask.astype(np.uint8))
        self.velocity_x = history_img_x
        self.velocity_y = history_img_y
        self.labeled_img = new_img

    def apply_movement_masks_without_velocity_generation(self, masks):
        """Applies the given masks as movements to the segmentation

        Changes labeled_img

        Parameters
        ----------
        masks : list[(ndarray, (int, int)]
            movement masks to be applied
            - list of touples containg a mask, and a (x,y) direction

        Returns
        -------
        None
        """
        new_img = np.zeros(self.labeled_img.shape, np.float64)
        for mask, direction in masks:
            img_part = np.multiply(self.labeled_img, mask)
            M = np.float32([[1, 0, direction[0]], [0, 1, direction[1]]])
            moved_img_part = cv.warpAffine(img_part, M, (img_part.shape[1], img_part.shape[0]))
            _, add_mask = cv.threshold(new_img, 0.00000001, 1, cv.THRESH_BINARY_INV)
            new_img = cv.add(new_img, moved_img_part, dst=new_img, mask=add_mask.astype(np.uint8))

        _, add_mask = cv.threshold(new_img, 0.00000001, 1, cv.THRESH_BINARY_INV)
        new_img = cv.add(new_img, self.labeled_img, dst=new_img, mask=add_mask.astype(np.uint8))
        self.labeled_img = new_img

    def replace_labels_with_integers(self):
        """Replaces current label interval and space with integer counting up labels"""

        unique_values = np.unique(self.labeled_img)
        empty_labels = []
        ideal_max = len(unique_values)
        for i in range(1, ideal_max + 1):
            if i not in unique_values:
                empty_labels.append(i)
        empty_label_idx = 0
        for i in unique_values:
            if i > ideal_max:
                np.place(self.labeled_img, self.labeled_img==i, empty_labels[empty_label_idx])
                empty_label_idx += 1

    def get_segmentation_graph(self):
        nodes = set()
        edges = set()
        boundary_img = self.get_boundary_img()
        for x in range(boundary_img.shape[1]):
            for y in range(boundary_img.shape[0]):
                if boundary_img[y, x] == 1:
                    label = self.labeled_img[y, x]
                    nodes.add(label)
                    for i in range(-1, 1):
                        x_n = x + i
                        if 0 <= x_n < boundary_img.shape[1]:
                            for j in range(-1, 1):
                                y_n = y + i
                                if 0 <= y_n < boundary_img.shape[0]:
                                    label_n = self.labeled_img[y_n, x_n]
                                    if label < label_n:
                                        edges.add((label, label_n))
                                    elif label_n < label:
                                        edges.add((label_n, label))

        return nodes, edges

    def merge_segments(self, mask):
        bool_mask = mask.astype(bool)
        masked_part = self.labeled_img[bool_mask]
        ids, counts = np.unique(masked_part, return_counts=True)
        ids = ids.astype(np.int32)
        ids_to_merge = list()
        merged_segment = np.full(self.labeled_img.shape, False, bool)
        full_segment_sizes = list()
        for i, id in enumerate(ids):
            full_segment = np.equal(self.labeled_img, id)
            covered_part = counts[i] / float(np.sum(full_segment))
            if covered_part >= 0.3:
                ids_to_merge.append(id)
                full_segment_sizes.append(float(np.sum(full_segment)))
                merged_segment = np.logical_or(merged_segment, full_segment)
        if len(ids_to_merge) > 0:
            new_id = np.min(np.array(ids_to_merge))
            add_mask = merged_segment.astype(np.float64) * new_id
            mask_inv = np.logical_not(merged_segment)
            self.labeled_img = cv.add(add_mask, self.labeled_img, dst=add_mask, mask=mask_inv.astype(np.uint8))

    def split_segments(self, mask):
        min_pixel_thresh = self.labeled_img.shape[0] * self.labeled_img.shape[1] * 0.00025
        bool_mask = mask.astype(bool)
        masked_part = self.labeled_img[bool_mask]
        ids, counts = np.unique(masked_part, return_counts=True)
        ids = ids.astype(np.int32)
        next_id = np.max(self.labeled_img) + 1
        for i, id in enumerate(ids):
            full_segment = np.equal(self.labeled_img, id)
            covered_part = counts[i] / float(np.sum(full_segment))
            full_img_covered_part = counts[i] / (self.labeled_img.shape[0] * self.labeled_img.shape[1])
            if (0.9 >= covered_part >= 0.1 or  0.98 >= full_img_covered_part >= 0.02) and counts[i] > min_pixel_thresh and np.sum(full_segment) - counts[i] > min_pixel_thresh:
                new_segment = np.logical_and(bool_mask, full_segment)
                add_mask = new_segment.astype(np.float64) * next_id
                mask_inv = np.logical_not(new_segment)
                self.labeled_img = cv.add(add_mask, self.labeled_img, dst=add_mask, mask=mask_inv.astype(np.uint8))
                next_id += 1

    def spawn_segment(self, mask):
        bool_mask = mask.astype(bool)
        self.labeled_img[bool_mask] = np.max(self.labeled_img) + 1

    def __copy__(self):
        """
        Creates a copy of the particle

        :return: copy of this particle
        """
        p = Particle(np.copy(self.labeled_img), curr_weight=self.curr_weight)
        p.velocity_x = np.copy(self.velocity_x)
        p.velocity_y = np.copy(self.velocity_y)
        return p

    def __repr__(self):
        """
        String representation of the particle

        :return: String representation of the particle
        """
        return "P w/ " + str(self.curr_weight)

