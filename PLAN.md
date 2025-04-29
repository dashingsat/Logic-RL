# Plan: Adapting Logic-RL for Financial Reconciliation

**1. Goal:**
Develop a Logic-RL system using the existing `Logic-RL` repository (based on `Verl` and Qwen 2.5) to perform financial reconciliation. The system will identify mismatches between synthetic trade and position data, provide chain-of-thought reasoning (`<think>`), output a classification (`<answer>`), and learn via a custom rule-based reward function as defined in `spec.md`. The system must support both full and partial aggregation modes.

**2. Core Repository & Framework:**
- Base Code: `Logic-RL` repository (forked).
- RL Framework: `Verl` (already present in the repo).
- LLM: Qwen 2.5 (identified in the repo).

**3. Key Modifications Required:**
- Replace the K&K puzzle task with the financial reconciliation task.
- Implement synthetic data generation specific to trades/positions.
- Define new reward logic based on `spec.md`.
- Adapt data loading, environment interaction, and training configuration.

**4. Detailed Implementation Steps:**

**Step 1: Project Setup & Scaffolding**
   - **Action:** Create the necessary directories and empty Python files for the new task.
   - **Files/Dirs:**
     - Create root directory: `financial_rec/`
     - Inside `financial_rec/`:
       - `generate_synthetic_data.py`
       - `data_loader.py`
       - `environment.py`
       - `train_rl.py` (potential alternative/wrapper for `main_grpo.sh`)
       - `evaluate_rl.py`
       - `config.py` (Optional: for centralizing parameters like tolerances, file paths)
       - `data/` (subdirectory for generated data)
     - Inside `verl/utils/reward_score/`:
       - `financial_rec.py`

**Step 2: Synthetic Data Generation**
   - **Action:** Implement the logic to generate synthetic trade-position pairs with ground-truth labels **and mismatch details** according to `spec.md`.
   - **File:** `financial_rec/generate_synthetic_data.py`
   - **Details:**
     - Define data structures (e.g., dataclasses or dicts) for `Trade` and `Position`.
     - Implement generation logic:
       - Random selection of securities, currencies.
       - Controlled introduction of mismatches (Quantity, Price, FX, Date, MissingTrade, etc.) based on defined thresholds/rules.
       - Inclusion of novel/custom reason codes (e.g., "AccountError") with specified frequency.
     - **Add a `mismatch_details` field to each sample:** This field will contain structured information about the specific component and values involved in the generated mismatch (e.g., `{"component": "price", "trade_value": 100.5, "position_value": 102.0}`). For "NoBreak", this field can be null or empty.
     - Output format: List of JSON objects (including `trades`, `position`, `label`, and `mismatch_details`) saved to `financial_rec/data/synthetic_data.json`.

**Step 3: Data Loading**
   - **Action:** Implement functions to load and split the generated synthetic data.
   - **File:** `financial_rec/data_loader.py`
   - **Details:**
     - Function to read `synthetic_data.json` (including the new `mismatch_details` field).
     - Function to split data into train/validation/test sets.

**Step 4: Reward Function Implementation**
   - **Action:** Implement the reward calculation logic based on `spec.md` and modeled after `verl/utils/reward_score/kk.py`.
   - **File:** `verl/utils/reward_score/financial_rec.py`
   - **Details:**
     - Create a main `compute_score(llm_output_str, ground_truth_sample)` function. The `ground_truth_sample` will include the `label` and `mismatch_details`.
     - Implement helper functions:
       - `parse_llm_output`: Extract `<think>` and `<answer>` content. Use regex.
       - `validate_response_structure`: Check format (presence/order of tags). Return format reward (+1 / -1).
       - `calculate_classification_reward`: Compare extracted `<answer>` to `ground_truth_sample['label']`. Return classification reward (+2 / -2 / +0.5 optional). **Note: For Scope 1, `mismatch_details` is not directly used in reward calculation but is available for analysis/future use.**
       - `calculate_aggregation_reward`: (If full aggregation mode) Parse `<think>` for arithmetic (e.g., sum of quantities). Compare to expected values based on trades and position. Return aggregation reward (+1 / -1). *This requires careful regex/parsing.*
       - `calculate_efficiency_reward`: (Optional) Check token length of `<think>` block. Return efficiency penalty (-0.2).
     - `compute_score` sums up the partial rewards.

**Step 5: Environment Definition**
   - **Action:** Create a custom environment for the financial reconciliation task, likely inheriting from or conforming to a base environment structure within `Verl`.
   - **File:** `financial_rec/environment.py`
   - **Details:**
     - `__init__`: Takes configuration (aggregation mode, data paths, thresholds). Loads data using `data_loader.py`.
     - `reset`: Fetches the next data sample (trades, position, label). Formats the System Prompt and User Query according to `spec.md`, inserting the trade/position data and relevant thresholds. Returns the formatted prompt as the initial observation.
     - `step(action)`:
       - `action`: The raw output string from the LLM.
       - Calls `verl.utils.reward_score.financial_rec.compute_score` with the `action` and the current sample's ground truth label.
       - Returns `(next_observation, reward, done, info)`. `next_observation` might be the prompt for the next sample, or handled by `reset`. `done` is likely True after each sample.

**Step 6: LLM Integration & Prompting**
   - **Action:** Ensure the LLM (Qwen 2.5) is called correctly with the prompts generated by the environment.
   - **File(s):** Likely within `Verl`'s trainer or model handling modules, potentially needing configuration updates passed from the training script.
   - **Details:**
     - Verify how `Verl` handles model inference (likely using `vllm` based on `requirements.txt`).
     - Ensure the System Prompt (defining persona, format rules, thresholds) and User Query (containing trade/position data) from `financial_rec/environment.py` are correctly passed to the LLM.

**Step 7: Training Script Adaptation**
   - **Action:** Configure and set up the RL training process using the new components.
   - **File:** Modify `main_grpo.sh` or create `financial_rec/train_rl.py`.
   - **Details:**
     - Update script arguments/configuration to point to:
       - The new dataset path (`financial_rec/data/synthetic_data.json`).
       - The custom environment (`financial_rec.environment.FinancialRecEnv`).
       - The new reward function module (`verl.utils.reward_score.financial_rec`).
     - Specify the aggregation mode (full/partial) via configuration.
     - Configure hyperparameters (learning rate, batch size, etc.).
     - Set up logging (e.g., using WandB as suggested in `requirements.txt`).

**Step 8: Evaluation Script**
   - **Action:** Implement a script to evaluate the fine-tuned model on the test set.
   - **File:** `financial_rec/evaluate_rl.py`
   - **Details:**
     - Load the trained model checkpoint.
     - Instantiate the environment in evaluation mode (using the test split).
     - Iterate through the test set, get model predictions (`<answer>`), and compare with ground truth labels.
     - Calculate and report metrics: Classification Accuracy, Confusion Matrix, Average Reward.
     - Optionally, log/display sample predictions with their `<think>` blocks for qualitative analysis.

**Step 9: Configuration & Execution**
   - **Action:** Manage dependencies and run the pipeline.
   - **Files:** `requirements.txt`, `main_grpo.sh` (or `financial_rec/train_rl.py`)
   - **Details:**
     - Add any new dependencies (e.g., `pandas` if used for data handling) to `requirements.txt`.
     - Ensure environment setup (`conda activate logic`) is correct.
     - Execute data generation: `python financial_rec/generate_synthetic_data.py`
     - Execute training: `bash main_grpo.sh` (or `python financial_rec/train_rl.py`)
     - Execute evaluation: `python financial_rec/evaluate_rl.py`

**5. Deliverable:**
A fine-tuned Qwen 2.5 model checkpoint, trained using the `Verl` framework on the synthetic financial reconciliation dataset, capable of accurately classifying reconciliation breaks with explainable reasoning (`<think>` blocks) and evaluated based on the metrics defined. 