# Knowledge Vector Weakening: Efficient Training-free Unlearning for Large Vision-Language Models
This repository provides the official PyTorch implementation of the following paper:
> [**Knowledge Vector Weakening: Efficient Training-free Unlearning for Large Vision-Language Models**](https://arxiv.org/abs/2601.21794) <br>
> [Yejin Kim](https://sites.google.com/view/yejin-c-kim/home?authuser=0), [Dongjun Hwang](https://dongjunhwang.github.io/), [Sungmin Cha](https://sites.google.com/view/sungmin-cha/)†, [Junsuk Choe](https://sites.google.com/site/junsukchoe/)† <br>
> †Co-corresponding Authors

[![arXiv](https://img.shields.io/badge/arXiv-2601.21794-9acd32.svg)](https://arxiv.org/abs/2601.21794)

## Abstract
> Large Vision-Language Models (LVLMs) are widely adopted for their strong multimodal capabilities, yet they raise serious concerns such as privacy leakage and harmful content generation. Machine unlearning has emerged as a promising solution for removing the influence of specific data from trained models. However, existing approaches largely rely on gradient-based optimization, incurring substantial computational costs for large-scale LVLMs. To address this limitation, we propose Knowledge Vector Weakening (KVW), a training-free unlearning method that directly intervenes in the full model without gradient computation. KVW identifies knowledge vectors that are activated during the model's output generation on the forget set and progressively weakens their contributions, thereby preventing the model from exploiting undesirable knowledge. Experiments on the MLLMU and CLEAR benchmarks demonstrate that KVW achieves a stable forget-retain trade-off while significantly improving computational efficiency over gradient-based and LoRA-based unlearning methods.

## Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/d3735af0-a30a-40a9-bacb-1b49cd45db90" width="800"/>
</p>


## Cite
```
@misc{kim2026knowledgevectorweakeningefficient,
      title={Knowledge Vector Weakening: Efficient Training-free Unlearning for Large Vision-Language Models}, 
      author={Yejin Kim and Dongjun Hwang and Sungmin Cha and Junsuk Choe},
      year={2026},
      eprint={2601.21794},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21794}, 
}
```
