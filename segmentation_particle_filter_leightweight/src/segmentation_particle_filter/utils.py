"""
Different utility functions used in various components
"""
import numpy as np
import cv2 as cv


def shape_convert_np_cv2(shape):
    """Converts numpy shape to cv2 shape and the other way around

    Parameters
    ----------
    shape : tuple[int]
        shape tuple

    Returns
    -------
    tuple[int]
        shape tuple in the other convention

    """
    return (shape[1], shape[0], *shape[2:])


def generate_dist_map_and_movement_masks(boundary_img, distance_threshold=-1):
    """Generates a distance map and the corresponding movement masks

    Parameters
    ----------
    boundary_img : ndarray
        boundary image (1 if boundary, 0 else)
    distance_threshold: float, optional
        A threshold of distance beyond which no movement should be displayed in the movement masks

    Returns
    -------
    (ndarray, list[(ndarray, (int, int)])
        corresponding distance map and movement masks with direction
        (list of movement masks (x minus, x plus, y minus, y plus))
    """

    dist_map = get_dist_map_for_border_img(boundary_img)
    if distance_threshold > 0:
        _, dist_map_threshed = cv.threshold(dist_map, distance_threshold, 255, cv.THRESH_TRUNC)
    else:
        dist_map_threshed = dist_map
    gradient_masks = generate_movement_masks_for_dist_img(dist_map_threshed)
    return dist_map, gradient_masks


def generate_movement_masks_for_dist_img(dist_image):
    """Generates movement masks based on the gradients of the distance image

    Parameters
    ----------
    dist_image : ndarray
        distance transform of a boundary image

    Returns
    -------
    list[ndarray]
        list of movement masks (x minus, x plus, y minus, y plus)

    """
    kernel = np.array([[-1, 0, 1]], np.float32)
    x_grad = cv.filter2D(dist_image, cv.CV_32F, kernel=kernel)
    kernel = np.array([[-1], [0], [1]], np.float32)
    y_grad = cv.filter2D(dist_image, cv.CV_32F, kernel=kernel)
    _, x_grad_minus = cv.threshold(x_grad, 1, 1, cv.THRESH_BINARY)
    _, x_grad_plus = cv.threshold(x_grad, -1, 1, cv.THRESH_BINARY_INV)
    _, y_grad_minus = cv.threshold(y_grad, 1, 1, cv.THRESH_BINARY)
    _, y_grad_plus = cv.threshold(y_grad, -1, 1, cv.THRESH_BINARY_INV)
    movement_masks = generate_4_directions_masks(x_grad_minus, x_grad_plus, y_grad_minus, y_grad_plus)
    return movement_masks


def generate_8_directions_masks(x_grad_minus, x_grad_plus, y_grad_minus, y_grad_plus):
    """Generates additional diagonal movement masks for the 4-direction masks
    Currently not used (takes longer, but does almost not improve the result)

    Parameters
    ----------
    x_grad_minus : ndarray
        movement mask x negative direction
    x_grad_plus : ndarray
        movement mask x positive direction
    y_grad_minus : ndarray
        movement mask y negative direction
    y_grad_plus : ndarray
        movement mask y positive direction

    Returns
    -------
    list[(ndarray, (int, int)]
        list of 8-direction gradient masks each as touple (mask, direction (x,y))

    """
    _, xpyp = cv.threshold(x_grad_plus + y_grad_plus, 1, 1, cv.THRESH_BINARY)
    _, xpyn = cv.threshold(x_grad_plus + y_grad_minus, 1, 1, cv.THRESH_BINARY)
    _, xnyp = cv.threshold(x_grad_minus + y_grad_plus, 1, 1, cv.THRESH_BINARY)
    _, xnyn = cv.threshold(x_grad_minus + y_grad_minus, 1, 1, cv.THRESH_BINARY)
    gradient_masks = [(x_grad_plus, (1, 0)), (x_grad_minus, (-1, 0)),
                      (y_grad_plus, (0, 1)), (y_grad_minus, (0, -1)),
                      (xpyp, (1, 1)), (xpyn, (1, -1)),
                      (xnyp, (-1, 1)), (xnyn, (-1, -1))]
    return gradient_masks


def generate_4_directions_masks(x_grad_minus, x_grad_plus, y_grad_minus, y_grad_plus):
    """Generates 4-direction masks

    Parameters
    ----------
    x_grad_minus : ndarray
        movement mask x negative direction
    x_grad_plus : ndarray
        movement mask x positive direction
    y_grad_minus : ndarray
        movement mask y negative direction
    y_grad_plus : ndarray
        movement mask y positive direction

    Returns
    -------
    list[(ndarray, (int, int)]
        list of 4-direction gradient masks each as touple (mask, direction (x,y))

    """
    gradient_masks = [(x_grad_plus, (1, 0)), (x_grad_minus, (-1, 0)),
                      (y_grad_plus, (0, 1)), (y_grad_minus, (0, -1))]
    return gradient_masks


def get_highest_weighted_particle(particle_set):
    """Finds and returns the highest weighted particle in a weighted particle set

    Parameters
    ----------
    particle_set : List[Particle]
        list of particles

    Returns
    -------
    Particle
        highest weighted particle

    """
    max_score = -1
    max_scored_particle = None
    for p in particle_set:
        if p.curr_weight > max_score:
            max_score = p.curr_weight
            max_scored_particle = p
    return max_scored_particle


def convert_label_to_boundary(labeled_img):
    """Converts a labeled image to a boundary image where every pixel which is direct neighbor to an other segment is part
    of the boundary

    Parameters
    ----------
    labeled_img : ndarray
        labeled segmentation image

    Returns
    -------
    ndarray
        the corresponding boundary image

    """
    border_img_merged = generate_boundary_img(labeled_img)
    return border_img_merged


def generate_boundary_img(img):
    """Generates a boundary image for a given image

    A boundary image is in this context an image, where every pixel with at least one different value neighboring pixel
    is one, all else are zero

    Parameters
    ----------
    img : ndarray
        image with regions of some sort

    Returns
    -------
    ndarray
        The corresponding boundary image
    """
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
    border_img = cv.filter2D(img.astype(np.float32), cv.CV_32F, kernel=kernel)  # borders where labels change
    _, border_img1 = cv.threshold(border_img, 0.0001, 1, cv.THRESH_BINARY)  # binary
    _, border_img2 = cv.threshold(border_img, -0.0001, 1, cv.THRESH_BINARY_INV)  # binary (negatives)
    border_img_merged = cv.add(border_img1, border_img2)
    _, border_img_merged = cv.threshold(border_img_merged, 0.1, 1, cv.THRESH_BINARY)
    return border_img_merged.astype(np.uint8)


def get_dist_map_for_border_img(boundary_image):
    """Creates the distance transform for the given boundary image

    Parameters
    ----------
    boundary_image : ndarray
        boundary image, where pixel is 1 if on boundary and 0 else

    Returns
    -------
    ndarray
        distance map for the boundary image

    """
    _, img = cv.threshold(boundary_image, 0, 1, cv.THRESH_BINARY_INV)  # invert so border is 0
    return cv.distanceTransform(img, cv.DIST_L2, maskSize=5)  # distance to border_pixels


def build_pairwise_dist_matrix(boundary_images):
    """Builds a pairwise distance matrix for a list of boundary images

    Parameters
    ----------
    boundary_images : list[ndarray]
        list of boundary images

    Returns
    -------
    ndarray
        pairwise distance matrix for the list

    """
    dist_mat = np.zeros((len(boundary_images), len(boundary_images)), np.float32)
    uint_imgs = [img.astype(np.uint8) for img in boundary_images]
    for i, p1 in enumerate(uint_imgs):
        for j, p2 in enumerate(uint_imgs[i:]):
            dist_mat[j + i][i] = determine_segmentation_pair_dist(p1, p2)
            dist_mat[i][j + i] = dist_mat[j + i][i]
    return dist_mat


def determine_segmentation_pair_dist(boundary1, boundary2):
    """Determines the distance between two boundary segmentation images

    Parameters
    ----------
    boundary1 : ndarray
        boundary image
    boundary2 : ndarray
        boundary image

    Returns
    -------
    float
        distance between the boundary images

    """
    dist_map1 = get_dist_map_for_border_img(boundary1)
    dist_map2 = get_dist_map_for_border_img(boundary2)
    pixel_count1 = np.sum(boundary1)
    pixel_count2 = np.sum(boundary2)
    score1 = np.sum(np.multiply(boundary1, dist_map2))
    score2 = np.sum(np.multiply(boundary2, dist_map1))
    if pixel_count1 == 0 or pixel_count2 == 0:
        # if one of them is empty the dist is 0
        return 0
    total_score = score1 / pixel_count1 + score2 / pixel_count2
    return total_score / 2


def get_label_mask(label_img, label_id):
    """Gets region with a specific label as masked image (1 where this label is in input, 0 elsewhere)

    Parameters
    ----------
    label_img : ndarray
        labeled segmentation image
    label_id : float
        id of the label to have the mask extracted

    Returns
    -------
    ndarray
        mask image (1 where this label id is in input, 0 elsewhere)

    """
    _, mask = cv.threshold(label_img, label_id, 0, cv.THRESH_TOZERO_INV)    # only id and smaller + 0s
    _, mask = cv.threshold(mask, label_id - 1, 1, cv.THRESH_BINARY)         # only id pixels are 1
    return mask


def show_segmentation_graph(label_image, graph):
    nodes, edges = graph
    print("Graph with ",len(nodes)," ", len(edges))
    boundary_img = convert_label_to_boundary(label_image)
    bounded_boundaries = cv.rectangle(boundary_img, (0,0), (boundary_img.shape[1]-1, boundary_img.shape[0]-1), 1, thickness=1)
    contours, _ = cv.findContours(boundary_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    img = cv.cvtColor(boundary_img * 255, cv.COLOR_GRAY2BGR)
    positions = dict()
    for c in contours:
        m = cv.moments(c)
        if m['m00'] != 0:  # else contour is disregarded as it has no area
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            label = label_image[cy, cx]
            if label in nodes:
                cv.circle(img, (cx, cy), 4, (0,0,255), thickness=1)
                positions[label] = (cx, cy)

    for e in edges:
        try:
            p1 = positions[e[0]]
            p2 = positions[e[1]]
            cv.line(img, p1, p2, (255, 0,0), 1)
        except Exception:
            pass
    return img


def normalize_seg2rb_belief(belief):
    """Normalizes a seg2RB belief

    Parameters
    ----------
    belief : ndarray
        a belief or likelihood of segment2rb association (#segments x #rbs)

    Returns
    -------
    ndarray
        the normalized belief

    """
    return belief / np.sum(belief, axis=1)[:, None]


def change_certainty_of_seg2rb_belief(belief, certainty_factor):
    """Changes the certainty of a seg2RB belief according to a certainty factor

    Parameters
    ----------
    belief : ndarray
        a belief or likelihood of segment2rb association (#segments x #rbs)
    certainty_factor : float
        factor describing how the certainty should be changed

    Returns
    -------
    ndarray
        the same belief or likelihood of segment2rb association (#segments x #rbs), but with different certainty

    """
    new_belief = np.power(belief, certainty_factor)
    return normalize_seg2rb_belief(new_belief)

