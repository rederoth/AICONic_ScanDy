# A Robotics-Inspired Scanpath Model for Dynamic Real-World Scenes

## Interconnected Object Segmentation and Saccadic Decision-Making

Humans visually explore dynamic real-world scenes based on perceived objects, and yet the perception of these objects depends on the visual exploration. Object perception and saccadic decision-making are therefore complex and interdependent processes. We present a model that mechanistically simulates these processes in two interacting components. To model their interaction, we employ an information processing pattern from robotics, reasoning over perceptual uncertainty in their active interconnection. Thereby, we obtain an image-computable model that progressively refines its object segmentation while uncertainty over that segmentation guides the scanpath.

## Software Architecture

The saccadic decision making module is based on the [ScanDy](https://github.com/rederoth/ScanDy/) framework for simulating realistic **Scan**paths in **Dy**namic real-world scenes. The decision-making is implemented in `aiconic_scandy/src/scanpath_producer`.
The object segmentation module is based on a Bayesian particle filter that recursively updates the object segmentation based on multiple object cues, including a high confidence segmentation mask of the currently foveated object. We adapt an existing implementation of the particle filter (`segmentation_particle_filter_lightweight`)from [previous work](https://ieeexplore.ieee.org/abstract/document/10160908) on combining object segmentation for robotics.

## Installation & Reproducibility

All figures from our manuscript (Mengers, Roth et al., 2024) can be reproduced with [this notebook](aiconic_scandy/src/result_figs.ipynb), which is also executable on [Colab](https://colab.research.google.com/github/rederoth/AICONic_ScanDy/blob/main/aiconic_scandy/src/result_figs.ipynb).

If you want to run the model yourself, you first have to add the semantic segmentation submodules (see `aiconic_scandy/semantic_segmentation/README.md`). 
The easiest way to install all dependencies is to create a conda environment from the provided config file:

```bash
conda env create --file=cluster_config.yml
```

In our experience, the combination of `opencv` and the installation of the semantic segmentation modules can lead to some issues.
If you have problems with the installation, please contact us (see below).

## More information

### How to cite

If this project is useful for your research, please cite our pre-print:
> Mengers\*, V., Roth\*, N., Brock\*\*, O., Obermayer\*\*, K., & Rolfs\*\*, M. (2024). A Robotics-Inspired Scanpath Model Reveals the Importance of Uncertainty and Semantic Object Cues for Gaze Guidance in Dynamic Scenes. *arXiv preprint* arXiv:2408.01322.

\* equal contribution; \*\* equal supervision

```bibtex
@article{mengersroth2024robotics,
  title={A Robotics-Inspired Scanpath Model Reveals the Importance of Uncertainty and Semantic Object Cues for Gaze Guidance in Dynamic Scenes},
  author={Mengers*, Vito and Roth*, Nicolas and Brock**, Oliver and Obermayer**, Klaus and Rolfs**, Martin},
  journal={arXiv preprint arXiv:2408.01322},
  year={2024}
}
```

### Contact

If you have feedback, questions, and/or ideas, feel free to send a mail to [Nico](mailto:roth@tu-berlin.de) and/or [Vito](mailto:v.mengers@tu-berlin.de).

Technische Universität Berlin\
Science of Intelligence (SCIoI, MAR 5-2)\
Marchstr. 23, 10587 Berlin, Germany

### Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC 2002/1 "Science of Intelligence" – project number 390523135.
