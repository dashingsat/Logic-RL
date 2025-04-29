import json
import pandas as pd
import os
from typing import List, Dict, Any

# Assuming these modules and variables are accessible
from financial_rec.data_loader import load_synthetic_data, split_data
from financial_rec.config import (
    PRICE_TOLERANCE_PERCENT, 
    DATE_TOLERANCE_DAYS, 
    MISMATCH_TYPES, 
    OUTPUT_DIR # Use OUTPUT_DIR from config for consistency
)

# --- Prompt Formatting Functions (adapted from FinancialRecEnv) ---

def _format_trades(trades: List[Dict[str, Any]]) -> str:
    """Formats the list of trades into a readable string for the prompt."""
    if not trades:
        return "  - No trades provided."
    formatted = []
    for i, trade in enumerate(trades):
        trade_str = json.dumps(trade, indent=4) 
        indented_trade_str = "\n".join([f"    {line}" for line in trade_str.splitlines()])
        formatted.append(f"  Trade {i+1}:\n{indented_trade_str}")
    return "\n".join(formatted)

def _format_position(position: Dict[str, Any]) -> str:
    """Formats the position dictionary into a readable string for the prompt."""
    if not position:
         return "  - No position provided."
    pos_str = json.dumps(position, indent=4)
    indented_pos_str = "\n".join([f"    {line}" for line in pos_str.splitlines()])
    return f"  Position:\n{indented_pos_str}"

def _build_system_prompt() -> str:
    """
    Creates the static system prompt defining the LLM's role, task, and rules.
    """
    price_tolerance_str = f"{PRICE_TOLERANCE_PERCENT * 100:.2f}%"
    date_tolerance_str = f"{DATE_TOLERANCE_DAYS} day(s)"
    possible_answers = ", ".join(MISMATCH_TYPES)
    
    system_prompt = f"""You are a meticulous financial analyst specializing in trade reconciliation.

Your task is to compare a set of trades against a final position report and identify any discrepancies (breaks).

Key Reconciliation Rules:
- Quantity: Trades should sum up to match the position quantity (consider this especially in 'full' aggregation mode).
- Price: Individual trade prices vs. position price must be within a {price_tolerance_str} tolerance.
- Date: Trade dates vs. position date must be within a {date_tolerance_str} tolerance.
- FX: Ensure currency consistency or flag FX mismatches.
- Account/Broker: Check for inconsistencies if present.
- Missing/Extra Trades: Identify if trades seem missing or if the position doesn't account for all trades.

Output Format:

1. First, provide your step-by-step reasoning within <think></think> tags. Explain your checks (quantity, price, date, etc.) and calculations.
2. Second, provide the final classification within <answer></answer> tags. Use ONLY ONE of the following labels:
   {possible_answers}

Ensure the <think> block comes BEFORE the <answer> block.
"""
    return system_prompt

def _build_user_query(sample: Dict[str, Any]) -> str:
    """Creates the user query containing the specific trade/position data."""
    trades = sample.get('trades', [])
    position = sample.get('position', {})
    
    formatted_trades = _format_trades(trades)
    formatted_position = _format_position(position)
    
    user_query = (
        f"Please reconcile the following trades and position:\n\n"
        f"Trades:\n{formatted_trades}\n\n"
        f"Position:\n{formatted_position}"
    )
    return user_query

# --- Main Pre-processing Logic ---

def preprocess_and_save(input_json_path: str, 
                        output_dir: str, 
                        train_filename: str = "train.parquet", 
                        val_filename: str = "val.parquet",
                        test_filename: str = "test.parquet", # Added test split
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        random_seed: int = 42):
    """
    Loads data, formats prompts, adds metadata, splits, and saves as Parquet files.
    """
    print(f"Starting pre-processing for {input_json_path}...")
    
    # Load original data
    all_data = load_synthetic_data(input_json_path)
    
    # Split data first
    train_data, val_data, test_data = split_data(
        all_data, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True, 
        random_seed=random_seed
    )
    print(f"Data split into Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Get the common system prompt
    system_prompt = _build_system_prompt()

    processed_splits = {}
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        processed_records = []
        if not split_data:
            print(f"Warning: {split_name} split is empty, skipping.")
            continue
            
        print(f"Processing {split_name} split ({len(split_data)} records)...")
        for i, sample in enumerate(split_data):
            user_query = _build_user_query(sample)
            # Combine prompts into the final format expected by Verl
            full_prompt = f"{system_prompt}\n\nUSER: {user_query}\nASSISTANT:"
            
            record = {
                "prompt": full_prompt,
                "data_source": "financial_rec",
                # Store the original sample dict directly for the RewardManager
                "ground_truth_sample": sample 
            }
            processed_records.append(record)
            
            # Optional: Print progress
            # if (i + 1) % 100 == 0:
            #     print(f"  Processed {i+1}/{len(split_data)} records for {split_name}...")

        processed_splits[split_name] = pd.DataFrame(processed_records)
        print(f"Finished processing {split_name} split.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data as Parquet files
    train_path = os.path.join(output_dir, train_filename)
    val_path = os.path.join(output_dir, val_filename)
    test_path = os.path.join(output_dir, test_filename) # Added test path

    if 'train' in processed_splits:
        processed_splits['train'].to_parquet(train_path, index=False)
        print(f"Saved training data to {train_path}")
    if 'val' in processed_splits:
        processed_splits['val'].to_parquet(val_path, index=False)
        print(f"Saved validation data to {val_path}")
    if 'test' in processed_splits:
         processed_splits['test'].to_parquet(test_path, index=False)
         print(f"Saved test data to {test_path}") # Added test save

    print("Pre-processing complete.")

if __name__ == "__main__":
    # Use the paths defined in config.py if possible
    input_file = os.path.join(OUTPUT_DIR, "synthetic_data.json") 
    output_directory = OUTPUT_DIR # Save parquet files in the same data directory
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input JSON file not found at {input_file}")
        print("Please ensure you have run generate_synthetic_data.py first.")
    else:
        try:
            # Run the pre-processing
            preprocess_and_save(input_json_path=input_file, output_dir=output_directory)
            print("Successfully created train.parquet, val.parquet, and test.parquet in", output_directory)
            # Reminder about dependencies
            print("NOTE: Ensure 'pandas' and 'pyarrow' are installed (pip install pandas pyarrow).")
            print("Add them to your requirements.txt if needed.")
        except Exception as e:
            print(f"An error occurred during pre-processing: {e}")
            import traceback
            traceback.print_exc() 