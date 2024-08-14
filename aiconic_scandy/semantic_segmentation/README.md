# Semantic Segmentation

The code expects to the find semantic segmentation modules `FastSAM` and `sam-hq` in this directory. We did not include them in this repository due to potential licensing issues. Please download the models from the original repositories. The easiest way too do this is via submodules, for this, just run the following commands:

```bash
cd PATH_TO_THIS_DIRECTORY/aiconic_scandy/semantic_segmentation
git submodule add https://github.com/CASIA-IVA-Lab/FastSAM
git submodule add https://github.com/SysCV/sam-hq
```

For more details about submodules, see this [GitHub Blog](https://github.blog/open-source/git/working-with-submodules/).
