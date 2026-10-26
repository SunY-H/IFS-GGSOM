# IFS-GGSOM

MATLAB code for incremental feature selection from the paper:

**Granular growing self-organizing map for novelty detection and incremental feature selection**

Yuanhao Sun, Ping Zhu, Kanglin Qu, and Jiucheng Xu.
*Information Sciences*, 758, 123969 (2027). Available online 1 August 2026.

[Read the paper](https://doi.org/10.1016/j.ins.2026.123969)

IFS-GGSOM uses a granular growing self-organizing map to assign labels to incoming unlabeled data and selects informative features using entropy-based uncertainty measures.

The main model is `GGSOM_OFS.m`, with feature selection implemented in `OFS_FNGBRS111.m`.

If you use this code in your research, please cite the paper above.
