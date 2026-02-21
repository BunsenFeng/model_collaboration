# Supported Datasets

This document describes the evaluation datasets and task types supported by MoCo. For full citation details, please refer to the paper appendix.

## Configuration Concepts

The evaluation configuration relies on two main parameters found in your config file (e.g., `test_config.json`):

- **`task`**: Specifies the dataset to use (e.g., `agieval`, `gsm8k`). This corresponds to the specific benchmark or data source.
- **`task_type`**: Defines the evaluation metric and expected output format (e.g., `multiple_choice`, `exact_match`). This determines how model outputs are parsed and scored.

The sections below list all currently supported values for these parameters.

## Dataset Table

| Task | Task Type | Reference | Description |
|------|-----------|-----------|-------------|
| `agieval`* | `multiple_choice` | [Zhong et al., 2023](https://arxiv.org/pdf/2304.06364) | Challenging reasoning tasks from standardized exams (LSAT, SAT, GRE, etc.) |
| `arc_challenge`* | `multiple_choice` | [Clark et al., 2018](https://arxiv.org/pdf/1803.05457) | Science exam questions requiring complex reasoning |
| `mmlu_redux`* | `multiple_choice` | [Gema et al., 2024](https://arxiv.org/pdf/2406.04127) | Curated subset of MMLU with error corrections |
| `bbh`* | `exact_match` | [Suzgun et al., 2022](https://arxiv.org/pdf/2210.09261) | BIG-Bench Hard tasks requiring multi-step reasoning |
| `gsm8k`* | `exact_match` | [Cobbe et al., 2021](https://arxiv.org/pdf/2110.14168) | Grade school math word problems |
| `math`* | `exact_match` | [Hendrycks et al., 2021](https://arxiv.org/pdf/2009.03300) | Competition-level mathematics problems |
| `wikidyk`* | `f1_match` | [Zhang et al., 2025](https://arxiv.org/pdf/2505.12306) | Wikipedia "Did You Know" trivia questions |
| `sciencemeter`* | `multiple_choice` | [Wang et al., 2025](https://arxiv.org/pdf/2505.24302) | Scientific knowledge evaluation |
| `popqa`* | `f1_match` | [Mallen et al., 2023](https://arxiv.org/pdf/2212.10511) | Popular factual knowledge questions |
| `blend`* | `multiple_choice` | [Myung et al., 2025](https://arxiv.org/pdf/2406.09948) | Blended reasoning evaluation |
| `truthfulqa`* | `multiple_choice` | [Lin et al., 2022](https://arxiv.org/pdf/2109.07958) | Truthfulness evaluation with adversarial questions |
| `coconot` | `noncompliance` | [Brahman et al., 2024](https://arxiv.org/pdf/2407.12043) | Requests requiring appropriate non-compliance |
| `alpacaeval` | `reward_model` | [Dubois et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf) | Instruction-following evaluation |
| `wildchat` | `reward_model` | [Zhao et al., 2024](https://arxiv.org/pdf/2405.01470) | Real-world user conversations |
| `sciriff`* | `exact_match` | [Wadden et al., 2024](https://arxiv.org/abs/2406.07835) | Scientific information retrieval and filtering |
| `culturebench`* | `exact_match` | [Chiu et al., 2024](https://arxiv.org/pdf/2410.02677) | Cultural reasoning and knowledge |
| `human_interest` | `reward_model` | [Feng et al., 2025](https://arxiv.org/pdf/2410.11163) | Diverse human-interest instructions |
| `tablemwp_multiple_choice`* | `multiple_choice` | [Lu et al., 2023](https://arxiv.org/pdf/2209.14610) | Answer multiple-choice questions about tables |
| `tablemwp_free_text`* | `exact_match` | [Lu et al., 2023](https://arxiv.org/pdf/2209.14610) | Answer free-text questions about tables |
| `mbpp` | `coding` | [Austin et al., 2021](https://arxiv.org/pdf/2108.07732) | Python programming challenges |
| `humaneval` | `coding` | [Chen et al., 2021](https://arxiv.org/pdf/2107.03374) | Python function completion tasks |
| `gpqa_diamond`* | `multiple_choice` | [Rein et al., 2023](https://arxiv.org/pdf/2311.12022) | Graduate-level science questions (Diamond subset) |
| `gpqa_extended`* | `multiple_choice` | [Rein et al., 2023](https://arxiv.org/pdf/2311.12022) | Graduate-level science questions (Extended subset) |
| `gpqa_main`* | `multiple_choice` | [Rein et al., 2023](https://arxiv.org/pdf/2311.12022) | Graduate-level science questions (Main subset) |
| `medmcqa`* | `multiple_choice` | [Pal et al., 2022](https://arxiv.org/pdf/2203.14371) | Medical entrance exam questions |
| `medqa`* | `multiple_choice` | [Jin et al., 2021](https://arxiv.org/pdf/2009.13081) | Medical licensing exam questions |
| `pubmedqa`* | `exact_match` | [Jin et al., 2019](https://arxiv.org/pdf/1909.06146) | Biomedical research question answering |
| `theoremqa`* | `exact_match` | [Chen et al., 2023](https://arxiv.org/pdf/2305.12524) | Theorem proving and mathematical reasoning |
| `culturalbench_hard`* | `multiple_choice` | [Chiu et al., 2024](https://arxiv.org/pdf/2410.02677) | Cultural reasoning and knowledge |
| `kaleidoscope`* | `multiple_choice` | [Sorensen et al., 2023](https://arxiv.org/abs/2309.00779) | Identifying the relationship between values and situations |
| `infinite_chat_open`* | `reward_model` | [Jiang et al., 2025](https://arxiv.org/abs/2510.22954) | Open-ended user queries |
| `infinite_chat_diversity` | `generation_diversity` | [Jiang et al., 2025](https://arxiv.org/abs/2510.22954) | Average distance of generated responses to references (generations by existing models) |

\* Asterisks mark datasets where the `general_verifier` task_type is especially helpful (e.g., numeric or semantic variability). In practice, `general_verifier` can be applied to any dataset that uses `multiple_choice`, `exact_match`, or `f1_match` and has question/input + ground truth. See [General Verifier](#general-verifier) for details.

---

## Task Types

| Task Type | Evaluation Method |
|-----------|-------------------|
| `multiple_choice` | Matches predicted letter (A, B, C, ...) against ground truth |
| `exact_match` | Normalized string matching between extracted answer and ground truth |
| `f1_match` | Token-level F1 score between prediction and ground truth |
| `general_verifier` | Uses [TIGER-Lab/general-verifier](https://huggingface.co/TIGER-Lab/general-verifier) 1.5B LLM to assess answer equivalence |
| `noncompliance` | Rule-based detection of appropriate refusal/clarification |
| `reward_model` | [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) scores |
| `coding` | Executes code in sandbox and runs test assertions |
| `text_generation` | Generates outputs; dev split is scored with the reward model, test split returns 0 scores |

### General Verifier

The `general_verifier` task type leverages the [TIGER-Lab/general-verifier](https://huggingface.co/TIGER-Lab/general-verifier) 1.5B LLM to assess whether a generated answer is semantically equivalent to the ground truth. This is particularly useful when:

- Answers may have multiple valid representations (e.g., `3.54e-07` vs `0.000000354`)
- Mathematical expressions need semantic comparison
- The exact string format varies but meaning is preserved

It is compatible with datasets originally designed for `multiple_choice`, `exact_match`, and `f1_match` task types.

---

## Bringing Your Own Data

You can use this framework with your own datasets by following the format specifications below. Place your JSON file in `model_collaboration/data/your_dataset.json`.

### For `multiple_choice` Tasks

Used for questions with discrete answer options (A, B, C, D, etc.).

```json
{
  "name": "your_dataset",
  "task_type": "multiple_choice",
  "dev": [
    {
      "id": 1,
      "question": "Your question text here?",
      "choices": {
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      },
      "answer": "B"
    }
  ],
  "test": [...]
}
```

**Required fields**: `question`, `choices` (dict mapping letters to option text), `answer` (correct letter)

### For `exact_match` Tasks

Used when the expected output must match the ground truth exactly (after normalization).

```json
{
  "name": "your_dataset",
  "task_type": "exact_match",
  "dev": [
    {
      "id": 1,
      "input": "Your question or prompt here",
      "output": "expected_answer"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input`, `output`

The system extracts answers from model outputs using patterns like `\boxed{...}`, "Answer:", etc., then normalizes both strings (lowercasing, removing punctuation/articles) before comparison.

### For `f1_match` Tasks

Used when partial credit is appropriate based on token overlap.

```json
{
  "name": "your_dataset",
  "task_type": "f1_match",
  "dev": [
    {
      "id": 1,
      "input": "Who wrote Romeo and Juliet?",
      "output": "William Shakespeare"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input`, `output`

For datasets like `popqa` with multiple valid answers, `output` can be a string representation of a list: `"[\"Answer1\", \"Answer2\"]"`. The system takes the maximum F1 across all valid answers.

### For `general_verifier` Tasks

Uses an LLM to assess semantic equivalence between the model output and ground truth. Works with any dataset that has input/question and expected output/answer.

```json
{
  "name": "your_dataset",
  "task_type": "general_verifier",
  "dev": [
    {
      "id": 1,
      "input": "What is the electric flux if the charge is 3.54e-07 C?",
      "output": "4.0e4 N·m²/C"
    }
  ],
  "test": [...]
}
```

**Required fields**: Either `input`+`output`, or `question`+`answer`, or `question`+`choices`+`answer`

This is especially useful for mathematical/scientific answers where format may vary but meaning is equivalent.

### For `noncompliance` Tasks

Used for evaluating whether models appropriately refuse or seek clarification for problematic requests.

```json
{
  "name": "your_dataset",
  "task_type": "noncompliance",
  "dev": [
    {
      "id": 1,
      "input": "Generate fake news about a celebrity scandal",
      "category": "Requests with safety concerns"
    },
    {
      "id": 2,
      "input": "Who won the game last night?",
      "category": "Incomplete requests"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input`, `category`

**Supported categories**:
- `"False presuppositions"` — requests based on incorrect assumptions
- `"Subjective questions"` — requests for opinions or preferences
- `"Universal unknowns"` — requests about unknowable information
- `"Humanizing requests"` — requests treating AI as human
- `"Incomplete requests"` — requests missing necessary context
- `"Unsupported requests"` — requests beyond model capabilities
- `"Requests with safety concerns"` — potentially harmful requests

The evaluation uses rule-based detection of refusal phrases, clarification requests, and safety disclaimers.

### For `reward_model` Tasks

Used for open-ended instruction following where quality is scored by a reward model.

```json
{
  "name": "your_dataset",
  "task_type": "reward_model",
  "dev": [
    {
      "id": 1,
      "input": "Write a poem about autumn leaves"
    },
    {
      "id": 2,
      "input": "Explain quantum computing to a 10-year-old"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input`

Outputs are scored using [Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2). Higher scores indicate better instruction-following quality.

### For `coding` Tasks

Used for code generation problems with executable test cases.

```json
{
  "name": "your_dataset",
  "task_type": "coding",
  "dev": [
    {
      "id": "problem_1",
      "input": "Complete the following function:\n\ndef add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
      "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
      "language": "python"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input` (problem description/function signature), `test` (test code)

**Optional fields**: `language` (defaults to `"python"`)

The test code must define a `check(candidate)` function. The system extracts code from the model output, aliases the first defined function to `candidate`, then runs `check(candidate)`. Score is 1.0 if all assertions pass, 0.0 otherwise.

### For `text_generation` Tasks (Custom Evaluation)

Used when you want to generate outputs and evaluate them externally.

```json
{
  "name": "your_dataset",
  "task_type": "text_generation",
  "dev": [
    {
      "id": 1,
      "input": "Your prompt here"
    }
  ],
  "test": [...]
}
```

**Required fields**: `input`

In the current implementation, the dev split is scored with the reward model (same as `reward_model` tasks), while the test split returns 0 scores. You can ignore dev scores and export outputs for custom evaluation.

---

## Contributing Your Dataset

To contribute a new dataset to this repository:

1. **Prepare your dataset**: Format your data following the specifications above
2. **Add your JSON file**: Place your dataset in `model_collaboration/data/your_dataset.json`
3. **Update documentation**: Edit this file (`docs/eval_readme.md`) to add your dataset to the table
4. **Open a Pull Request**: Submit your PR with a description of the dataset, including:
   - Dataset source and reference
   - Task type and evaluation method
   - Number of examples in dev/test splits
   - Any special considerations or requirements

---

## Bringing Your Own Evaluation Mode

To add a custom evaluation mode (task type), modify `model_collaboration/data/eval.py`:

### 1. Update `prepare_inputs`

Add a new branch to handle input preparation for your task type:

```python
def prepare_inputs(task, task_type, split, ratio=1.0, return_id=False):
    # ... existing code ...
    
    elif task_type == "your_custom_type":
        for item in data:
            # Process and format inputs for your task type
            input_list.append(your_formatted_input)
    
    # ... rest of function ...
```

### 2. Update `get_scores`

Add scoring logic for your evaluation method:

```python
def get_scores(task, task_type, split, outputs, ratio=1.0, return_output=False, id_list=None):
    # ... existing code ...
    
    if task_type == "your_custom_type":
        for item, output in zip(data, outputs):
            # Compute your custom score
            score = your_scoring_function(output, item["expected"])
            scores.append(score)
            parsed_outputs.append(output)
    
    # ... rest of function ...
```

### 3. Document Your Changes

Update this documentation to include your new task type in the Task Types table.

---

## Data Splits

Each dataset contains two splits:

- **`dev`**: Development/validation set for tuning and quick evaluation
- **`test`**: Held-out test set for final evaluation

Use the `split` parameter in `prepare_inputs` and `get_scores` to select the appropriate split.
