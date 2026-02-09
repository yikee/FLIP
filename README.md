# 🔄 FLIP: Backward Inference for Generative Reward Modeling in Downscaled Regimes

<div align="center">
  <b>Yike Wang<sup>1</sup>, Faeze Brahman<sup>2</sup>, Shangbin Feng<sup>1</sup>, Teng Xiao<sup>2</sup>, Hannaneh Hajishirzi<sup>1</sup><sup>2</sup>, Yulia Tsvetkov<sup>1</sup></b>
  <br>
  <sup>1</sup>University of Washington, <sup>2</sup>Allen Institute for Artificial Intelligence
  <br><br>
  <!-- <a href="https://www.arxiv.org/abs/?"><img src="https://img.shields.io/badge/Paper-arXiv-orange"></a> -->
</div>

<img width="1680" height="458" alt="overview" src="https://github.com/user-attachments/assets/86910cfe-2516-4ca8-be3b-081dfdaa8e86" />


FLIP uses **backward inference** for reward modeling: given a model response, an LLM infers the instruction that would most plausibly have produced it; the reward is the **F1 score** between the inferred instruction and the original instruction.


## Repository structure

| Path | Description |
|------|-------------|
| **`prompts/`** | Prompt templates: FLIP (instruction inference) and LLM-judge baselines (pointwise, pairwise, listwise). |
| **`metrics.py`** | F1 score and normalization utilities for comparing inferred vs. ground-truth instructions. |
| **`open-instruct/`** | [Open Instruct](https://github.com/allenai/open-instruct) fork with FLIP integrated as a GRPO reward (judge type `flip`). |

---


### 1. Using the FLIP reward (standalone)

**Step 1 — Get inferred instruction from a response**

- Use the templates in **`prompts/`**:
  - **`prompts/FLIP_SYSTEM.prompt`**: system message for the instruction-reconstruction task.
  - **`prompts/FLIP_USER.prompt`**: user message template; replace `{response}` with the model’s response.
- Call your LLM with these prompts to obtain an inferred instruction (and optional reasoning). The model should output JSON with keys like `"REASONING"` and `"INFERRED INSTRUCTION"`.

**Step 2 — Compute reward with F1**

```python
from metrics import f1_score

# prediction = inferred instruction from the LLM
# ground_truth = original instruction
result = f1_score(prediction, ground_truth)["f1"]
```

`metrics.py` also provides **`normalize_answer(s)`** for normalizing text before comparison (lowercasing, removing punctuation/articles, fixing whitespace).

---

### 2. RL training with GRPO (FLIP as reward)

Training runs use the **Open Instruct** codebase under **`open-instruct/`**, with FLIP wired in as the judge type **`flip`**.

**Example datasets**

- **FLIP:** [`yikeee/rlvr_general_chat_flip`](https://huggingface.co/datasets/yikeee/rlvr_general_chat_flip)  
- **LLM Judge:** [`yikeee/rlvr_general_chat`](https://huggingface.co/datasets/yikeee/rlvr_general_chat)

You can use these datasets as-is, or adapt your own data to match the same schema.  
Both datasets share the same structure and content, differing only in the judge type specified by the `"dataset"` attribute.

**Running training**

- **FLIP reward (instruction inference + F1):**
  ```bash
  bash open-instruct/scripts/train/grpo_flip.sh
  ```
- **LLM-judge baseline (e.g. pointwise quality score):**
  ```bash
  bash open-instruct/scripts/train/grpo_llmjudge.sh
  ```

**Important script variables (in the `.sh` scripts)**

- **`DATASETS`** — Training mix, e.g. `"yikeee/rlvr_general_chat_flip 1.0"`. 
- **`MODEL_NAME_OR_PATH`** — Policy model (e.g. `allenai/Olmo-3-7B-Think-DPO`).
- **`--llm_judge_model`** — Judge model used for FLIP (infer instruction) or LLM-judge (e.g. `hosted_vllm/Qwen/Qwen3-4B`).

Inside Open Instruct, the FLIP pipeline uses the same idea as the standalone use: the judge model is prompted to infer the instruction from the response, then **`f1_score(inferred_instruction, ground_truth)`** (in `open-instruct/open_instruct/judge_utils.py` and `ground_truth_utils.py`) gives the reward.

---

## Questions

If you have any questions or comments about our paper, or notice any issues in the code, feel free to reach out at **yikewang@cs.washington.edu**. We will do our best to respond within one business day.

---

## Citing

If you found this work helpful, please consider starring this repository and citing our paper as shown below:
