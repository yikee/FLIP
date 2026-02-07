### FLIP: Backward Inference for Generative Reward Modeling in Downscaled Regimes

<div align="center">
  <b>Yike Wang<sup>1</sup>, Faeze Brahman<sup>2</sup>, Shangbin Feng<sup>1</sup>, Teng Xiao<sup>2</sup>, Hannaneh Hajishirzi<sup>1</sup><sup>2</sup>, Yulia Tsvetkov<sup>1</sup></b>
  <br>
  <sup>1</sup>University of Washington, <sup>2</sup>Allen Institute for Artificial Intelligence
  <br><br>
  <!-- <a href="https://www.arxiv.org/abs/?"><img src="https://img.shields.io/badge/Paper-arXiv-orange"></a> -->
</div>

<img width="1680" height="458" alt="overview" src="https://github.com/user-attachments/assets/86910cfe-2516-4ca8-be3b-081dfdaa8e86" />

#### FLIP Implementation
Given a response, we use a LM to infer the instruction that would most plausibly generate the response, and use the F1 score between the inferred and original instructions as the reward.
prompts in `prompts`, than use f1 in `metric.py`

#### RL Training
Our GRPO training runs are built using Open Instruct (https://github.com/allenai/open-instruct). 
specifying your configuration, such as 
DATASETS I use "yikeee/rlvr_general_chat_flip" (https://huggingface.co/datasets/yikeee/rlvr_general_chat_flip), feel free to change to others
--llm_judge_model hosted_vllm/Qwen/Qwen3-4B \ 

```bash
bash open-instruct/scripts/train/grpo_flip.sh
```
```bash
bash open-instruct/scripts/train/grpo_llmjudge.sh
```

## Questions
If you have any questions or comments about our paper, or if you notice any issues in the code, feel free to reach out via email at `yikewang@cs.washington.edu`. We will do our best to respond within one business day.

## Citing
If you found this work helpful, please consider starring this repository and citing our paper as shown below:
