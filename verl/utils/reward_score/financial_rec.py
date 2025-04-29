import re
from typing import Dict, Any, Optional, Tuple

# Define reward values (can be moved to config later)
REWARD_FORMAT_CORRECT = 1
REWARD_FORMAT_INCORRECT = -1
REWARD_CLASSIFICATION_CORRECT = 2
REWARD_CLASSIFICATION_INCORRECT = -2
REWARD_AGGREGATION_CORRECT = 1 # Placeholder value
REWARD_AGGREGATION_INCORRECT = -1 # Placeholder value
PENALTY_EFFICIENCY = -0.2 # Placeholder value

# Regex patterns to extract thought and answer
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def parse_llm_output(llm_output_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts the content within <think> and <answer> tags."""
    think_match = THINK_PATTERN.search(llm_output_str)
    answer_match = ANSWER_PATTERN.search(llm_output_str)
    
    think_content = think_match.group(1).strip() if think_match else None
    answer_content = answer_match.group(1).strip() if answer_match else None
    
    return think_content, answer_content

def validate_response_structure(llm_output_str: str, think_content: Optional[str], answer_content: Optional[str]) -> float:
    """Validates the structure (<think> before <answer>, both present)."""
    if think_content is None or answer_content is None:
        return REWARD_FORMAT_INCORRECT # Missing tags
        
    # Check if <think> appears before <answer>
    try:
        think_start_index = llm_output_str.index("<think>")
        answer_start_index = llm_output_str.index("<answer>")
        if think_start_index > answer_start_index:
            return REWARD_FORMAT_INCORRECT # Wrong order
    except ValueError:
        # This case should technically be caught by the None checks above,
        # but added for robustness.
        return REWARD_FORMAT_INCORRECT
        
    return REWARD_FORMAT_CORRECT

def calculate_classification_reward(answer_content: Optional[str], ground_truth_label: str) -> float:
    """Calculates the reward based on classification accuracy."""
    if answer_content is None:
        return REWARD_CLASSIFICATION_INCORRECT # Treat missing answer as incorrect
        
    # Simple exact match for now. Could be extended for partial credit (e.g., using spec.md logic)
    if answer_content.lower() == ground_truth_label.lower():
        return REWARD_CLASSIFICATION_CORRECT
    else:
        return REWARD_CLASSIFICATION_INCORRECT

def calculate_aggregation_reward(think_content: Optional[str], ground_truth_sample: Dict[str, Any], aggregation_mode: str) -> float:
    """
    Calculates the reward based on aggregation correctness (Placeholder).
    Requires parsing the <think> block for calculations and comparing 
    against expected values derived from ground_truth_sample.
    This is dependent on the aggregation_mode.
    """
    if aggregation_mode != 'full' or think_content is None:
        return 0.0 # No aggregation reward if not in full mode or no thought block
    
    # --- Placeholder Logic --- 
    # TODO: Implement actual parsing of think_content (e.g., sum of quantities)
    # TODO: Calculate expected aggregation result from ground_truth_sample ('trades', 'position')
    # TODO: Compare parsed result with expected result
    # Example: Assume we parsed a calculated_sum from think_content
    # expected_sum = sum(trade['Quantity'] for trade in ground_truth_sample['trades'])
    # if calculated_sum == expected_sum: # Add tolerance checks
    #     return REWARD_AGGREGATION_CORRECT
    # else:
    #     return REWARD_AGGREGATION_INCORRECT
    # --- End Placeholder --- 

    print("Warning: Aggregation reward calculation is not fully implemented.")
    return 0.0 # Return neutral reward until implemented

def calculate_efficiency_reward(think_content: Optional[str], threshold: int = 200) -> float:
    """Calculates an optional penalty for overly long thought processes (Placeholder)."""
    # TODO: This is optional based on PLAN.md. Implement if needed.
    # if think_content and len(think_content.split()) > threshold: # Example: Word count
    #     return PENALTY_EFFICIENCY
    return 0.0

def compute_score(llm_output_str: str, ground_truth_sample: Dict[str, Any], **kwargs) -> float:
    """
    Computes the overall reward score for the LLM output based on the financial rec task.

    Args:
        llm_output_str: The raw output string from the language model.
        ground_truth_sample: A dictionary containing the ground truth data, 
                             expected to have at least a 'label' key. 
                             Should also contain 'trades', 'position', and 
                             'mismatch_details' for full functionality.
        **kwargs: Additional arguments, potentially including:
                  aggregation_mode (str): 'full' or 'partial'
                  efficiency_threshold (int): Token/word count threshold

    Returns:
        The total calculated reward score.
    """
    total_reward = 0.0
    
    # Extract content
    think_content, answer_content = parse_llm_output(llm_output_str)
    
    # 1. Validate Structure
    format_reward = validate_response_structure(llm_output_str, think_content, answer_content)
    total_reward += format_reward
    # print(f"Format Reward: {format_reward}") # Debugging
    
    # If format is incorrect, penalize and potentially stop further checks?
    # For now, we continue calculating other rewards even if format is wrong.

    # 2. Calculate Classification Reward
    ground_truth_label = ground_truth_sample.get('label')
    if ground_truth_label is None:
        print("Warning: Ground truth sample missing 'label'. Cannot compute classification reward.")
        classification_reward = 0.0
    else:
        classification_reward = calculate_classification_reward(answer_content, ground_truth_label)
    total_reward += classification_reward
    # print(f"Classification Reward: {classification_reward}") # Debugging

    # 3. Calculate Aggregation Reward (if applicable)
    aggregation_mode = kwargs.get('aggregation_mode', 'partial') # Default to partial
    aggregation_reward = calculate_aggregation_reward(think_content, ground_truth_sample, aggregation_mode)
    total_reward += aggregation_reward
    # print(f"Aggregation Reward: {aggregation_reward}") # Debugging
    
    # 4. Calculate Efficiency Reward (Optional)
    efficiency_threshold = kwargs.get('efficiency_threshold')
    if efficiency_threshold is not None:
        efficiency_penalty = calculate_efficiency_reward(think_content, efficiency_threshold)
        total_reward += efficiency_penalty
        # print(f"Efficiency Penalty: {efficiency_penalty}") # Debugging

    # print(f"Total Reward: {total_reward}") # Debugging
    return total_reward

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Example 1: Correct format, correct classification (Partial mode)
    gt_sample1 = {'label': 'QuantityBreak', 'trades': [], 'position': {}}
    llm_out1 = "<think>The trade quantity is 100, position is 110. Mismatch.</think><answer>QuantityBreak</answer>"
    score1 = compute_score(llm_out1, gt_sample1)
    print(f"Example 1 Score: {score1} (Expected: {REWARD_FORMAT_CORRECT + REWARD_CLASSIFICATION_CORRECT})")

    # Example 2: Incorrect format (missing answer)
    gt_sample2 = {'label': 'NoBreak', 'trades': [], 'position': {}}
    llm_out2 = "<think>Everything looks fine.</think>"
    score2 = compute_score(llm_out2, gt_sample2)
    print(f"Example 2 Score: {score2} (Expected: {REWARD_FORMAT_INCORRECT + REWARD_CLASSIFICATION_INCORRECT})")
    
    # Example 3: Correct format, incorrect classification
    gt_sample3 = {'label': 'PriceBreak', 'trades': [], 'position': {}}
    llm_out3 = "<think>The price is different.</think><answer>CurrencyBreak</answer>"
    score3 = compute_score(llm_out3, gt_sample3)
    print(f"Example 3 Score: {score3} (Expected: {REWARD_FORMAT_CORRECT + REWARD_CLASSIFICATION_INCORRECT})")

    # Example 4: Wrong tag order
    gt_sample4 = {'label': 'NoBreak', 'trades': [], 'position': {}}
    llm_out4 = "<answer>NoBreak</answer><think>All good.</think>"
    score4 = compute_score(llm_out4, gt_sample4)
    print(f"Example 4 Score: {score4} (Expected: {REWARD_FORMAT_INCORRECT + REWARD_CLASSIFICATION_CORRECT})") # Still gets classification right

    # Example 5: Full aggregation mode (but logic is placeholder)
    gt_sample5 = {'label': 'QuantityBreak', 'trades': [{'Quantity': 50}, {'Quantity': 50}], 'position': {'Quantity': 110}}
    llm_out5 = "<think>Sum of trade quantities is 100. Position is 110. Mismatch.</think><answer>QuantityBreak</answer>"
    score5 = compute_score(llm_out5, gt_sample5, aggregation_mode='full')
    print(f"Example 5 Score (Full Mode): {score5} (Expected: {REWARD_FORMAT_CORRECT + REWARD_CLASSIFICATION_CORRECT} + 0.0 aggregation)") 