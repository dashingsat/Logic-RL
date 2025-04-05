Below is an **updated specification** for a **Logic-RL for Financial Reconciliation** pipeline, with added **examples in data entities** and a note on **synthetic data generation** to ensure fresh data that the LLM hasn’t seen in pre-training.

---

# Financial Reconciliation Logic-RL Spec (Enhanced)

## 1. Objective

We will build a **Logic-RL–style** system where a large language model (LLM) identifies **reconciliation breaks** (mismatches) between trades and positions, **explains** its reasoning, and **learns** via a rule-based reward structure. This approach must handle **full** and **partial** aggregation modes. Additionally, we will **generate synthetic data** (including novel reason codes) to ensure the LLM has not memorized or seen these exact samples during its pre-training.

---

## 2. Data / Environment Setup

### 2.1. Data Entities with Examples

We deal with two main data entities, **Trade** and **Position**, plus a **Reason Code** label:

1. **Trade Record**  
   - **Fields**:  
     - `trade_id` (string)  
     - `security` (string)  
     - `quantity` (integer/float)  
     - `price` (float)  
     - `trade_date` (date / string)  
     - `currency` (string, e.g. “USD,” “GBP,” etc.)  
     - `account_no` (optional)  
   - **Example**:
     ```json
     {
       "trade_id": "T123",
       "security": "AAPL",
       "quantity": 100,
       "price": 150.00,
       "trade_date": "2026-01-10",
       "currency": "USD",
       "account_no": "ACC777"
     }
     ```

2. **Position Record**  
   - **Fields**:  
     - `position_id` (string)  
     - `security` (string)  
     - `total_quantity` (integer/float)  
     - `position_price` (float)  
     - `position_date` (date / string)  
     - `currency` (string)  
     - `account_no` (optional)  
   - **Example**:
     ```json
     {
       "position_id": "P987",
       "security": "AAPL",
       "total_quantity": 300,
       "position_price": 149.75,
       "position_date": "2026-01-10",
       "currency": "USD",
       "account_no": "ACC777"
     }
     ```

3. **Reason Code** (Ground-Truth Label)  
   - A single classification describing the mismatch type, e.g.:  
     - `"NoBreak"`  
     - `"PriceMismatch"`  
     - `"QuantityMismatch"`  
     - `"MissingTrade"`  
     - `"FXMismatch"`  
     - `"DateMismatch"`  
     - or any *new codes* you define (e.g., `"BrokerMismatch"`, `"AccountError"`).  
   - **Example**: `"PriceMismatch"`

### 2.2. Multiple Trades vs. One Position

- A **Position** may be related to **one or more** trades.  
- If using **full aggregation**, pass *all trades* that presumably match the position to the LLM for summation and mismatch detection.  
- If using **partial aggregation**, your environment or some microservice pre-sums or groups the trades before giving data to the LLM.

### 2.3. Synthetic Data Generation

To ensure the LLM **hasn’t seen** these samples during pre-training, we will **synthetically generate** trade–position pairs with reason codes.  
- **Method**: 
  - Randomly pick `security` from a list (e.g., AAPL, TSLA, MSFT...).  
  - Vary the `quantity` and `price` slightly to create legitimate mismatches (exceeding a threshold for PriceMismatch, a difference in date for DateMismatch, etc.).  
  - Randomly decide if the currency is the same or different.  
  - Insert new or custom reason codes (like `"AccountError"`) for about 5–10% of the data, so they are truly fresh.  
- **Example of synthetic mismatch**:  
  ```json
  {
    "trade": [
      { "trade_id": "T305", "security": "AAPL", "quantity": 150, "price": 98.00, "trade_date": "2026-01-10", "currency": "USD" }
    ],
    "position": {
      "position_id": "P400", "security": "AAPL", "total_quantity": 150, "position_price": 101.00, "position_date": "2026-01-10", "currency": "USD"
    },
    "label": "PriceMismatch"
  }
  ```
  Here, we intentionally set `trade_price=98.00` vs. `position_price=101.00` to produce a mismatch beyond a certain tolerance, labeled `"PriceMismatch"`.

---

## 3. Prompt & Chain-of-Thought Format

1. **System Prompt**:  
   - Tells the LLM it’s a “Financial Reconciliation AI,” must output `<think>...</think>` for its chain-of-thought, and `<answer>...</answer>` for the final reason code.  
2. **User Query**:  
   - Provides the trade(s), the position, possibly thresholds (e.g., price tolerance = 1%, date tolerance = +/- 1 day), plus any extra domain info.  

Example Prompt Skeleton:

```
System: 
"You are a Financial Reconciliation AI. Produce reasoning in <think>...</think> tags, 
and your final break reason in <answer>...</answer>. 
Thresholds: PriceTolerance=1%, DateTolerance=1day."

User: 
"Trades: 
  T1: quantity=100, price=150, date=2026-01-10, currency=USD
  ...
Position: quantity=300, price=149.75, date=2026-01-10, currency=USD
Please identify if there's a mismatch."
```

The LLM’s response might be:
```
<think>
Aggregate trades...
Compare with position...
Conclude mismatch type...
</think>
<answer>QuantityMismatch</answer>
```

---

## 4. Reward Function Specification

1. **Format Reward**  
   - +1 if there is exactly one `<think>` block and one `<answer>` block, in correct order.  
   - –1 if the model violates these format requirements.  

2. **Correct Classification Reward**  
   - +2 if `<answer>` matches the ground-truth label exactly.  
   - +0.5 if it’s partially correct (optional).  
   - –2 if it’s wrong.

3. **Aggregation Reward** (Full-aggregation mode only)  
   - If we see correct arithmetic in `<think>` (like “100 + 50 = 150”), +1.  
   - If the chain-of-thought sum is obviously wrong, –1.  

4. **Token Length / Efficiency** (Optional)  
   - –0.2 if `<think>` block exceeds a chosen length (say 2,000 tokens).  
   - This keeps the model from rambling.

---

## 5. Pipeline Flow

1. **Synthetic Data Generation**  
   - `generate_synthetic_data.py`:  
     - Creates random trade–position pairs with controlled mismatches.  
     - Assigns reason codes like “PriceMismatch,” “QuantityMismatch,” etc.  
     - Ensures certain new codes or new securities appear that the LLM is unlikely to have memorized.

2. **Data Loading**  
   - `data_loader.py`: Splits into train/validation/test sets.  

3. **Environment**  
   - `environment.py`:  
     - Reads each synthetic sample.  
     - If **full aggregation**: pass all trades in the sample to the LLM prompt.  
     - If **partial aggregation**: do some grouping and pass the summarized figures to the LLM.

4. **LLM Inference**  
   - The environment calls the LLM, receives the chain-of-thought in `<think>` and the final `<answer>`.

5. **Reward Calculation**  
   - `reward_functions.py`:  
     - Parse format.  
     - Compare `<answer>` to the known label.  
     - (If full aggregation) parse the chain-of-thought for arithmetic checks.  
     - Summation of all partial rewards = final reward.

6. **RL Training**  
   - `train_rl.py`: uses the environment + reward function to do REINFORCE++, PPO, or GRPO updates.  
   - Keep track of metrics (accuracy, reward, confusion matrix, etc.).

7. **Evaluation**  
   - `evaluate_rl.py`: measure classification accuracy on a hold-out set of synthetic data.  
   - Inspect chain-of-thought for correctness.  

---

## 6. Handling Full vs. Partial Aggregation

- **Full Aggregation**:  
  - The environment feeds a list of trades to the LLM: `[T1, T2, T3, ...]` plus the position info.  
  - The LLM attempts to sum `T1.quantity + T2.quantity + ...` in `<think>`.  
  - The reward function checks if that sum is correct if the label indicates a quantity mismatch or not.

- **Partial Aggregation**:  
  - The environment or an upstream microservice sums some trades or organizes them by key.  
  - The environment passes the pre-summarized quantity or price data to the LLM.  
  - The LLM only verifies or finalizes the mismatch classification in `<think>`.  
  - The reward is simpler to compute (less arithmetic parsing).

---

## 7. Example Data

Below is a short excerpt of what might appear in **synthetic_data.json**:

```json
[
  {
    "trades": [
      { "trade_id": "T100", "security": "MSFT", "quantity": 100, "price": 250.0, "trade_date": "2026-03-01", "currency": "USD" },
      { "trade_id": "T101", "security": "MSFT", "quantity": 200, "price": 250.0, "trade_date": "2026-03-01", "currency": "USD" }
    ],
    "position": {
      "position_id": "P200", "security": "MSFT", "total_quantity": 300, "position_price": 250.0, "position_date": "2026-03-01", "currency": "USD"
    },
    "label": "NoBreak"
  },
  {
    "trades": [
      { "trade_id": "T200", "security": "AAPL", "quantity": 150, "price": 150.0, "trade_date": "2026-03-02", "currency": "USD" }
    ],
    "position": {
      "position_id": "P201", "security": "AAPL", "total_quantity": 150, "position_price": 158.0, "position_date": "2026-03-02", "currency": "USD"
    },
    "label": "PriceMismatch"
  }
]
```

*(In the second entry, we intentionally create a mismatch by making the position price 158.0 vs. the trade’s 150.0, to exceed a certain tolerance.)*

---

## 8. Final Summary

This specification ensures:

1. **Synthetic Data**: We generate brand-new trade–position scenarios (with novel reason codes, slightly random data) so the LLM has not seen them before.  
2. **Full vs. Partial Aggregation**: The pipeline can feed either raw trades (letting the LLM sum them) or partially aggregated data.  
3. **Chain-of-Thought + Answer**: The system prompt demands a `<think>` explanation and `<answer>` reason code.  
4. **Reward Logic**: The environment checks format correctness, final classification, and optionally arithmetic.  
5. **Goal**: Produce an RL-finetuned model that classifies break types accurately *and* explains how it derived them, all without having memorized the data from pre-training.

This approach fosters **transparency** (via chain-of-thought), **robustness** (via synthetic data), and **flexibility** (both aggregation modes) for a wide range of financial reconciliation use cases.