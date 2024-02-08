# Delving into Multi-modal Multi-task Foundation Models for Road Scene Understanding: From Learning Paradigm Perspectives
 **Abstract:** Foundation models have indeed made a profound
impact on various fields, emerging as pivotal components that
significantly shape the capabilities of intelligent systems. In the
context of intelligent vehicles, leveraging the power of foundation
models has proven to be transformative, offering notable advancements in visual understanding. Equipped with multi-modal
and multi-task learning capabilities, multi-modal multi-task visual understanding foundation models (MM-VUFMs) effectively
process and fuse data from diverse modalities and simultaneously
handle various driving-related tasks with powerful adaptability,
contributing to a more holistic understanding of the surrounding
scene. In this survey, we present a systematic analysis of MMVUFMs specifically designed for road scenes. Our objective is not
only to provide a comprehensive overview of common practices,
referring to task-specific models, unified multi-modal models,
unified multi-task models, and foundation model prompting
techniques, but also to highlight their advanced capabilities
in diverse learning paradigms. These paradigms include openworld understanding, efficient transfer for road scenes, continual
learning, interactive and generative capability. Moreover, we
provide insights into key challenges and future trends, such as
closed-loop driving systems, interpretability, embodied driving
agents, and world models.

**Authors:** Sheng Luo, Wei Chen, Wanxin Tian, Rui Liu, Luanxuan Hou, Xiubao Zhang, Haifeng Shen,
Ruiqi Wu, Shuyi Geng, Yi Zhou*, Ling Shao, Yi Yang, Bojun Gao, Qun Li and Guobin Wu

![At a glance of our survey](/assets/at_a_glance.pdf)

## 📖Table of Contents
- [News](#💥news)
- [Roadmap](#roadmap) 
- [Paper Collection](#paper-collection)
- [Acknowledgement & Citation](#acknowledgement--citation)


## 💥News
- [2024.02.05] Our survey is available at [hear](https://arxiv.org/abs/2402.02968).

## Roadmap

![Roadmap](/assets/roadmap.pdf)

## 📚Paper Collection
- [Related Surveys](#papers.md#related-surveys)
- [Task-specific Models](#paper.md#task-specific-models)
  - [From instance-level perception to global-level understanding](#paper.md#from-instance-level-perception-to-global-level-understanding)
  - [From closed-set condition to open-set condition](#paper.md#from-closed-set-condition-to-open-set-condition)
  - [From single modality to multi-modalities](#paper.md#from-single-modality-to-multiple-modalities)
- [Unified Multi-task Models](#paper.md#unified-multi-task-models)
  - [Task-specific outputs](#paper.md#task-specific-outputs)
  - [Unified language outputs](#paper.md#unified-language-outputs)
- [Unified Multi-modal Models](#paper.md#unified-multi-modal-models)
  - [LLM functions as sequence modeling](#paper.md#LLM-functions-as-sequence-modeling)
  - [Cross-modal interaction in VLM](#paper.md#Cross-modal-interaction-in-VLM)
- [Prompting Foundation Models](#paper.md#prompting-foundation-models)
  - [Textual prompt](#paper.md#textual-prompt)
  - [Visual prompt](#paper.md#visual-prompt)
  - [Multi-step prompt](#paper.md#multi-step-prompt)
  - [Task-specific prompt](#paper.md#task-specific-prompt)
  - [Prompt pool](#paper.md#prompt-pool)
- [Towards Open-world Understanding](#paper.md#towards-open-world-understanding)
- [Efficient Transfer for Road Scenes](#paper.md#efficient-transfer-for-road-scenes)
- [Continual Learning](#paper.md#continual-learning)
- [Learn to Interact](#paper.md#learn-to-interact)
- [Generative Foundation Models](#paper.md#generative-foundation-models)
- [Closed-loop Driving Systems](#paper.md#closed-loop-driving-systems)
- [Interpretability](#paper.md#interpretability)
- [Embodied Driving Agent](#paper.md#embodied-driving-agent)
- [World Model](#paper.md#world-model)


## 💗Acknowledgement & Citation
This work was supported by DiDi GAIA Research Cooperation Initiative. If you find this work useful, please consider cite:
```
@misc{luo2024delving,
      title={Delving into Multi-modal Multi-task Foundation Models for Road Scene Understanding: From Learning Paradigm Perspectives}, 
      author={Sheng Luo and Wei Chen and Wanxin Tian and Rui Liu and Luanxuan Hou and Xiubao Zhang and Haifeng Shen and Ruiqi Wu and Shuyi Geng and Yi Zhou and Ling Shao and Yi Yang and Bojun Gao and Qun Li and Guobin Wu},
      year={2024},
      eprint={2402.02968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


