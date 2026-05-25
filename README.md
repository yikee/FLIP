# Small Reward Models via Backward Inference

<img width="1680" height="458" alt="overview" src="https://github.com/user-attachments/assets/86910cfe-2516-4ca8-be3b-081dfdaa8e86" />

---

We propose 🔄 **FLIP**, a reference-free and rubric-free reward modeling approach: infer the instruction that would most plausibly produce a given response, and use the similarity between the inferred and the original instructions as the reward signal.


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

- **FLIP:**
- **LLM Judge:**


Both datasets share the same structure and content, consisting of 12k English prompts from the WildChat dataset. They only differ in the judge type specified by the `"dataset"` attribute.
You can use these datasets as-is, or adapt your own data to match the same schema.  

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

- **`DATASETS`** — Training mix. 
- **`MODEL_NAME_OR_PATH`** — Policy model (e.g. `allenai/Olmo-3-7B-Think-DPO`).
- **`--llm_judge_model`** — Judge model used for FLIP (infer instruction) or LLM-judge (e.g. `hosted_vllm/Qwen/Qwen3-4B`).

Inside Open Instruct, the FLIP pipeline uses the same idea as the standalone use: the judge model is prompted to infer the instruction from the response, then **`f1_score(inferred_instruction, ground_truth)`** (in `open-instruct/open_instruct/judge_utils.py` and `ground_truth_utils.py`) gives the reward.
