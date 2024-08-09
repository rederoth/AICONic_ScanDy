import socket

import cv2 as cv
import numpy as np
import glob
import torch
import os
from abc import abstractmethod, ABCMeta
from segmentation_particle_filter.utils import shape_convert_np_cv2
# from Models.CoherenceNets.MethodeB import MethodeB
from fastsam import FastSAM, FastSAMPrompt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


CLUSTER = socket.gethostname() != "scioi-1712"
if CLUSTER:
    SAM_PATH = "/scratch/vito/scanpathes/SAM_model_weights/"
    FASTSAM_PATH = "/scratch/vito/scanpathes/code/domip_scanpathes/semantic_segmentation/FastSAM/weights/FastSAM-x.pt"
else:
    SAM_PATH = "/media/vito/scanpath_backup/SAMcheckpoint/"
    FASTSAM_PATH = "/home/vito/ws_domip/src/scanpath_code/domip_scanpathes/semantic_segmentation/FastSAM/weights/FastSAM-x.pt"


def get_segmenter(segmentation_algo, goal_shape, orig_shape=None, segmenter_scale_factor=1.0):
    """Constructs the corresponding segmenter and returns it

    Parameters
    ----------
    segmentation_algo : {"graph", "lsc", "slic", "hfs",
        "cyclic_GSL", "cyclic_GH"}
        string identifying the wanted segmenter type (graph, lsc, slic, hfs,
        cyclic_GSL=cyclic with Graph SLIC and LSC
        , cyclic_GH=cyclic with Graph and HFS)
    goal_shape : (int, int)
        shape in which the segmentation should be returned by the segmenter
    orig_shape : (int, int)
        original input shape (Default value = None)

    Returns
    -------
    Segmenter
        the segmenter object

    """
    segmenter_shape = (int(np.round(orig_shape[0]*segmenter_scale_factor)), int(np.round(orig_shape[1]*segmenter_scale_factor)))
    if segmentation_algo == "graph":
        segmenter = GraphSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "graph_prompt":
        segmenter = PromptableGraphSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "lsc":
        segmenter = LSCSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "slic":
        segmenter = SLICSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "hfs":
        segmenter = HFSSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "cyclic_GSL":
        segmenter = CyclicSegmenter([GraphSegmenter(goal_shape, segmenter_shape),
                                     LSCSegmenter(goal_shape, segmenter_shape),
                                     SLICSegmenter(goal_shape, segmenter_shape)])
    elif segmentation_algo == "cyclic_GH":
        segmenter = CyclicSegmenter([GraphSegmenter(goal_shape, segmenter_shape),
                                     HFSSegmenter(goal_shape, segmenter_shape)])
    elif segmentation_algo == "fake":
        segmenter = FakeSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "SAM":
        segmenter = SAMSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "PromptSAM":
        segmenter = PromptableSAMSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "FASTSAM":
        segmenter = FASTSAMSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "PromptFASTSAM":
        segmenter = PromptableFASTSAMSegmenter(goal_shape, segmenter_shape)
    elif segmentation_algo == "motion":
        segmenter = MotionSegmenterEMFlow(goal_shape, segmenter_shape)
    else:
        raise ValueError("segmenation algorithm unkown")
    return segmenter


class Segmenter(metaclass=ABCMeta):
    """Abstract Segmenter class"""

    def __init__(self, goal_shape, segmenter_shape):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter

        """
        self.goal_shape = goal_shape
        self.segmenter_shape = segmenter_shape
        self.shape_masks = []
        self.curr_belief = []
        self.weights = []
        # self.curr_frame_stamp = rospy.Time()
        self.curr_frame_stamp = 0.0

    @abstractmethod
    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : array_like
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        pass

    def set_shape_masks(self, masks):
        self.shape_masks = masks

    def set_frame_stamp(self, stamp):
        self.curr_frame_stamp = stamp

    def set_curr_belief(self, labeled_imgs, weights):
        self.curr_belief = labeled_imgs
        self.weights = weights


class GraphSegmenter(Segmenter):
    """Graph based segmentation
    
    Based on:
    P. F. Felzenszwalb and D. P. Huttenlocher, “Efficient graph-based
    image segmentation,” International journal of computer vision, vol. 59,
    no. 2, pp. 167–181, 2004

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, goal_shape, segmenter_shape, init_algorithm=True):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter

        """
        super(GraphSegmenter, self).__init__(goal_shape=goal_shape, segmenter_shape=segmenter_shape)
        self.k = 350
        self.sigma = 0.5
        self.min_size = self.segmenter_shape[0] * self.segmenter_shape[1] * 1.0 / 3072.0 * 9
        if init_algorithm:
            self.segmentation_algo = cv.ximgproc.segmentation.createGraphSegmentation(self.sigma, self.k, int(np.round(self.min_size)))

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = self.segmentation_algo.processImage(scaled_img).astype(np.float32)
        labeled_img = cv.resize(labeled_img, self.goal_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = np.add(labeled_img, 1)  # to forbid a segment with id 0
        return labeled_img


class PromptableGraphSegmenter(GraphSegmenter):

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)
        self.point = None

    def get_labeled_img(self, img):
        labeled = super().get_labeled_img(img)
        id = labeled[int(self.point[1]), int(self.point[0])]
        mask = labeled == id
        new_img = np.zeros_like(labeled)
        new_img[mask] = 1.0
        return new_img

    def give_prompt_point(self, point):
        self.point = point


class SLICSegmenter(Segmenter):
    """SLIC segmentation
    
    Based on:
    W. F. Noh and P. Woodward, “Slic (simple line interface calculation),”
    in Proceedings of the fifth international conference on numerical
    methods in fluid dynamics June 28–July 2, 1976 Twente University,
    Enschede. Springer, 1976, pp. 330–340.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, goal_shape, segmenter_shape, iterations=2, region_size=125):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter
        iterations : int
            number of iterations to be performed by the segmenter
        region_size : int
            region size parameter of the SLIC algorithm

        """
        self.region_size = region_size
        self.iterations = iterations
        super(SLICSegmenter, self).__init__(goal_shape=goal_shape, segmenter_shape=segmenter_shape)

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        segmenter = cv.ximgproc.createSuperpixelSLIC(scaled_img, region_size=self.region_size, algorithm=cv.ximgproc.MSLIC)
        segmenter.iterate(self.iterations)
        labeled_img = segmenter.getLabels().astype(np.float32)
        labeled_img = cv.resize(labeled_img, self.goal_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = np.add(labeled_img, 1)    # to forbid a segment with id 0
        return labeled_img


class LSCSegmenter(Segmenter):
    """LSC segmentation
    
    Based on:
    Z. Li and J. Chen, “Superpixel segmentation using linear spectral
    clustering,” in Proceedings of the IEEE Conference on Computer
    Vision and Pattern Recognition, 2015, pp. 1356–1363.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, goal_shape, segmenter_shape, iterations=2, region_size=125):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter
        iterations : int
            number of iterations to be performed by the segmenter
        region_size : int
            region size parameter of the LSC algorithm

        """
        self.region_size = region_size
        self.iterations = iterations
        super(LSCSegmenter, self).__init__(goal_shape=goal_shape, segmenter_shape=segmenter_shape)

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        segmenter = cv.ximgproc.createSuperpixelLSC(scaled_img, region_size=self.region_size)
        segmenter.iterate(self.iterations)
        labeled_img = segmenter.getLabels().astype(np.float32)
        labeled_img = cv.resize(labeled_img, self.goal_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = np.add(labeled_img, 1)  # to forbid a segment with id 0
        return labeled_img


class HFSSegmenter(Segmenter):
    """Hierarchical feature selection segmentation
    
    Based on:
    M.-M. Cheng, Y. Liu, Q. Hou, J. Bian, P. Torr, S.-M. Hu, and Z. Tu,
    “Hfs: Hierarchical feature selection for efficient image segmentation,”
    in European conference on computer vision. Springer, 2016, pp.867–882.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, goal_shape, segmenter_shape, iterations=5, region_size=125):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter
        orig_shape : (int, int)
            shape of the input image
        iterations : int
            number of iterations to be performed by the segmenter
        region_size : int
            region size parameter of the HFS algorithm

        """
        self.hfssegmenter = cv.hfs.HfsSegment_create(segmenter_shape[1], segmenter_shape[0], minRegionSizeI=region_size, numSlicIter=iterations)
        super(HFSSegmenter, self).__init__(goal_shape, segmenter_shape=segmenter_shape)

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = self.hfssegmenter.performSegmentGpu(scaled_img).astype(np.float32)[:, :, 0] # only take one channel (all are the same)
        labeled_img = cv.resize(labeled_img, self.goal_shape, interpolation=cv.INTER_NEAREST)
        labeled_img = np.add(labeled_img, 1)  # to forbid a segment with id 0
        return labeled_img


class CyclicSegmenter(Segmenter):
    """Segmnenter that continously iterates through the given list of segmenters"""

    def __init__(self, segmenters):
        """Initializes the segmenter

        Parameters
        ----------
        segmenters: list[Segmenter]
            list with  segmters in the order they should be used

        """
        self.segmenters = segmenters
        self._internal_counter = 0
        super(CyclicSegmenter, self).__init__(goal_shape=segmenters[0].goal_shape, segmenter_shape=segmenters[0].segmenter_shape)

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        labeled_img = self.segmenters[self._internal_counter].get_labeled_img(img=img)
        self._internal_counter += 1
        if self._internal_counter >= len(self.segmenters):
            self._internal_counter = 0
        return labeled_img


class FakeSegmenter(Segmenter):

    def get_labeled_img(self, img):
        return cv.resize(img, shape_convert_np_cv2(self.goal_shape), interpolation=cv.INTER_NEAREST)


class RVOSSegmenter(Segmenter):
    """Segmnenter that uses precomputed RVOS segmentations"""

    def __init__(self, goal_shape):
        """Initializes the segmenter

        Parameters
        ----------
        goal_shape : (int, int)
            shape in which the segmentation should be returned by the segmenter
        """
        self.interaction_path = None
        self.curr_value_dict = dict()
        self.next_id = 1.0
        self.last_orig_segs = []
        super(RVOSSegmenter, self).__init__(goal_shape, None)

    def get_labeled_img(self, img):
        """Computes the segmentation image of the given image and returns it

        Parameters
        ----------
        img : np.Array
            Camera view input image

        Returns
        -------
        np.Array
            labeled image representation of the segmented image

        """
        if self.interaction_path is None:
            # rospy.logerr("RVOS Segmenter has no specified path!")
            return np.zeros(shape_convert_np_cv2(self.goal_shape), dtype=np.float32)
        curr_path = os.path.join(self.interaction_path, str(self.curr_frame_stamp)+"_instance_*.png")
        files = glob.glob(curr_path)
        if len(files) == 0:
            # rospy.logerr("RVOS Segmenter -- No found segmentations! Path: "+curr_path)
            return np.zeros(shape_convert_np_cv2(self.goal_shape), dtype=np.float32)
        # get their common segmentation by overlapping all bondaries
        complete_seg = np.zeros(shape_convert_np_cv2(self.goal_shape), dtype=np.int64)
        counter = 1
        assert len(files) < 30
        self.last_orig_segs = []
        for f in files:
            seg = cv.imread(f)
            seg = cv.cvtColor(seg, cv.COLOR_BGR2GRAY).astype(bool)
            self.last_orig_segs.append(seg.astype(np.float32))
            complete_seg += counter * seg.astype(np.int64)
            counter *= 10   # unsafe for many files
        # some value shifting for stable and small ids
        values, inverse = np.unique(complete_seg, return_inverse=True)
        ids = np.zeros(values.shape, np.float32)
        for i, v in enumerate(values):
            if v in self.curr_value_dict.keys():
                ids[i] = self.curr_value_dict[v]
            else:
                ids[i] = self.next_id
                self.curr_value_dict[v] = self.next_id
                self.next_id += 1.0
        labeled_img = np.reshape(ids[inverse], complete_seg.shape)  # reconstruct with new ids
        labeled_img = cv.resize(labeled_img, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img

    def set_interaction_path(self, path):
        self.interaction_path = path

    def get_last_orig_segs(self):
        return self.last_orig_segs


class SAMSegmenter(Segmenter):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    sam = None

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)
        if SAMSegmenter.sam is None:
            # SAMSegmenter.sam = sam_model_registry["vit_h"](checkpoint="/media/vito/TOSHIBA EXT/SAMcheckpoint/sam_vit_h_4b8939.pth").to(device=SAMSegmenter.device)
            # SAMSegmenter.sam = sam_model_registry["vit_tiny"](checkpoint="/media/vito/TOSHIBA EXT/SAMcheckpoint/sam_hq_vit_tiny.pth").to(device=SAMSegmenter.device)
            # SAMSegmenter.sam = sam_model_registry["vit_b"](checkpoint="/media/vito/TOSHIBA EXT/SAMcheckpoint/sam_hq_vit_b.pth").to(device=SAMSegmenter.device)
            # SAMSegmenter.sam = sam_model_registry["vit_l"](checkpoint="/media/vito/TOSHIBA EXT/SAMcheckpoint/sam_hq_vit_l.pth").to(device=SAMSegmenter.device)
            if torch.cuda.is_available():
                SAMSegmenter.sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH+"sam_hq_vit_h.pth").to(device=SAMSegmenter.device)
            else:
                SAMSegmenter.sam = sam_model_registry["vit_h"](
                    checkpoint=SAM_PATH+"sam_hq_vit_h.pth")
            print('We use sam_hq_vit_h.pth')

        self.mask_generator = SamAutomaticMaskGenerator(SAMSegmenter.sam)
        self.predictor = SamPredictor(SAMSegmenter.sam)

    def get_labeled_img(self, img):
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        masks = self.mask_generator.generate(scaled_img, multimask_output=True)
        seg = np.zeros(scaled_img.shape[:2])
        for i, m in enumerate(masks):
            basic_mask = m['segmentation']
            seg[basic_mask] = i + 1
        labeled_img = cv.resize(seg, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img


class PromptableSAMSegmenter(SAMSegmenter):

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)
        self.point = None
        self.additonal_point_weight = 1.0
        self.additional_point_dist = 0.0

    def get_labeled_img(self, img):
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        self.predictor.set_image(scaled_img)
        if self.point is None:
            raise AssertionError("No point for prompt provided!")
        point = np.array([self.point])

        points = point
        values = np.array([1.0])

        # points = np.concatenate([point,
        #                          point + np.array([[self.additional_point_dist, 0]]),
        #                          point + np.array([[-self.additional_point_dist, 0]]),
        #                          point + np.array([[0, self.additional_point_dist]]),
        #                          point + np.array([[0, -self.additional_point_dist]]), ])
        # values = np.array([1, self.additonal_point_weight, self.additonal_point_weight, self.additonal_point_weight,
        #                   self.additonal_point_weight])
        masks, scores, logits = self.predictor.predict(
                            point_coords=points,
                            point_labels=values,
                            multimask_output=True,
                            hq_token_only=False
                            )
        seg = np.zeros(scaled_img.shape[:2])
        for i in range(masks.shape[0]):
            basic_mask = masks[i]
            # kernel = np.ones((7, 7), np.uint8)
            # basic_mask = cv.morphologyEx(basic_mask.astype(np.uint8), cv.MORPH_OPEN, kernel)
            # basic_mask = cv.morphologyEx(basic_mask, cv.MORPH_CLOSE, kernel)
            additional_mask = np.logical_and(basic_mask.astype(bool), np.logical_not(seg.astype(bool)))
            seg[additional_mask] = i + 1
        labeled_img = cv.resize(seg, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img

    def give_prompt_point(self, point):
        self.point = point


class FASTSAMSegmenter(Segmenter):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    model = None

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)
        if self.model is None:  # model singleton
            self.model = FastSAM(FASTSAM_PATH)

    def get_labeled_img(self, img):
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        seg = np.zeros(scaled_img.shape[:2])
        out = self.model(scaled_img, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        if out is not None:
            masks = out[0].masks.data
            for i in range(masks.shape[0]):
                basic_mask = masks[i].cpu().numpy().astype(bool)
                seg[basic_mask] = i + 1
        else:
            print("FASTSAMSegmenter did not provide any masks!")
        labeled_img = cv.resize(seg, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img


class PromptableFASTSAMSegmenter(FASTSAMSegmenter):

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)

    def get_labeled_img(self, img):
        scaled_img = cv.resize(img, self.segmenter_shape, interpolation=cv.INTER_NEAREST)
        out = self.model(scaled_img, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        if self.point is None:
            raise AssertionError("No point for prompt provided!")
        ann = FastSAMPrompt(scaled_img, out, device=self.device).point_prompt(points=[np.round(self.point).astype(int).tolist()], pointlabel=[1])

        seg = np.zeros(scaled_img.shape[:2])
        if len(ann) > 0:
            for i in range(ann.shape[0]):
                basic_mask = ann[i]
                # kernel = np.ones((7, 7), np.uint8)
                # basic_mask = cv.morphologyEx(basic_mask.astype(np.uint8), cv.MORPH_OPEN, kernel)
                # basic_mask = cv.morphologyEx(basic_mask, cv.MORPH_CLOSE, kernel)
                additional_mask = np.logical_and(basic_mask.astype(bool), np.logical_not(seg.astype(bool)))
                seg[additional_mask] = i + 1
        else:
            print("PromptableFASTSAMSegmenter did not provide a mask!")
        labeled_img = cv.resize(seg, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img

    def give_prompt_point(self, point):
        self.point = point


class MotionSegmenterEMFlow(Segmenter):

    def __init__(self, goal_shape, segmenter_shape):
        super().__init__(goal_shape, segmenter_shape)
        ckpt = '/media/vito/TOSHIBA EXT/em-driven-segmentation-data-main-2-Masks ( 1tdcjfqp )-checkpoint/2-Masks ( 1tdcjfqp )/checkpoint/epoch=31-epoch=epoch_val_loss=0.53934.ckpt'  # Path of the model
        self.model = MethodeB.load_from_checkpoint(ckpt, strict=False).eval()
        self.minimum_flow = 3.0

    def get_labeled_img(self, img):
        flow = cv.resize(img, (self.model.hparams['img_size'][1], self.model.hparams['img_size'][0]))
        flow = torch.swapaxes(torch.from_numpy(flow).unsqueeze(0), 0, 3).squeeze().unsqueeze(0)
        masks = []
        if torch.sum(torch.sum(torch.abs(flow), axis=1) > self.minimum_flow) == 0:
            with torch.no_grad():
                r = self.model.prediction({'Flow': flow.type(torch.get_default_dtype()).to(torch.cuda.current_device())})
            result_mask = r['Pred'].argmax(1)[0]
            masks.append(result_mask)
        while torch.sum(torch.sum(torch.abs(flow), axis=1) > self.minimum_flow) > 0:
            with torch.no_grad():
                r = self.model.prediction({'Flow': flow.type(torch.get_default_dtype()).to(torch.cuda.current_device())})
            result_mask = r['Pred'].argmax(1)[0]
            masks.append(result_mask)
            flow[:, :, result_mask > 0] = 0
        color = 1
        seg = np.zeros(flow.shape[-2:], np.uint8)
        for _, m in enumerate(masks):
            contours, _ = cv.findContours(m.cpu().numpy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):
                tmp_img = np.zeros_like(seg)
                cv.drawContours(tmp_img, contours=contours, contourIdx=j, color=1, thickness=-1)
                mask = tmp_img > 0
                additional_mask = np.logical_and(mask.astype(bool), np.logical_not(seg.astype(bool)))
                if np.sum(additional_mask) > 10 and np.sum(seg[additional_mask]) == 0:
                    seg[additional_mask] = color
                    color += 1
        labeled_img = cv.resize(seg, self.goal_shape, interpolation=cv.INTER_NEAREST)
        return labeled_img
