import json
import random
from typing import List, Dict, Tuple, Any, Optional

# Import necessary components from the project
from financial_rec.data_loader import load_synthetic_data, split_data
from verl.utils.reward_score.financial_rec import compute_score
# Import config variables directly (alternative: pass config object/path)
from financial_rec.config import PRICE_TOLERANCE_PERCENT, DATE_TOLERANCE_DAYS, MISMATCH_TYPES

class FinancialRecEnv:
    """
    A custom RL environment for the financial reconciliation task.
    Conforms to a standard RL environment interface (reset, step).
    """
    def __init__(self,
                 data_path: str,
                 config: Optional[Dict[str, Any]] = None, # Placeholder for future config injection
                 split: str = 'train', 
                 aggregation_mode: str = 'partial',
                 random_seed: int = 42):
        """
        Initializes the environment.

        Args:
            data_path: Path to the synthetic data JSON file.
            config: Optional dictionary for configuration (e.g., thresholds, prompt templates).
            split: Which data split to use ('train', 'val', or 'test').
            aggregation_mode: 'full' or 'partial', passed to reward function.
            random_seed: Seed for data splitting shuffle.
        """
        print(f"Initializing FinancialRecEnv for {split} split...")
        self.config = config if config is not None else {}
        self.aggregation_mode = aggregation_mode
        self.random_seed = random_seed
        
        # Load and split data
        try:
            all_data = load_synthetic_data(data_path)
            train_data, val_data, test_data = split_data(
                all_data,
                shuffle=True, 
                random_seed=self.random_seed
            )
        except Exception as e:
            print(f"Error initializing environment: Failed to load or split data from {data_path}. {e}")
            raise

        if split == 'train':
            self.dataset = train_data
        elif split == 'val':
            self.dataset = val_data
        elif split == 'test':
            self.dataset = test_data
        else:
            raise ValueError(f"Invalid split specified: {split}. Choose 'train', 'val', or 'test'.")
            
        if not self.dataset:
             raise ValueError(f"The selected data split '{split}' is empty. Check data generation and splitting ratios.")

        self.dataset_size = len(self.dataset)
        self.current_index = -1 # Start before the first element
        self.current_ground_truth: Optional[Dict[str, Any]] = None

        print(f"Environment initialized with {self.dataset_size} samples for {split} split.")

    def _format_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Formats the list of trades into a readable string for the prompt."""
        if not trades:
            return "  - No trades provided."
        formatted = []
        for i, trade in enumerate(trades):
            # Use json.dumps for consistent formatting of trade details
            trade_str = json.dumps(trade, indent=4) 
            # Indent the whole block for readability in the prompt
            indented_trade_str = "\n".join([f"    {line}" for line in trade_str.splitlines()])
            formatted.append(f"  Trade {i+1}:\n{indented_trade_str}")
        return "\n".join(formatted)

    def _format_position(self, position: Dict[str, Any]) -> str:
        """Formats the position dictionary into a readable string for the prompt."""
        if not position:
             return "  - No position provided."
        # Use json.dumps for consistent formatting
        pos_str = json.dumps(position, indent=4)
        # Indent the whole block
        indented_pos_str = "\n".join([f"    {line}" for line in pos_str.splitlines()])
        return f"  Position:\n{indented_pos_str}"

    def _build_system_prompt(self) -> str:
        """
        Creates the static system prompt defining the LLM's role, task, and rules.
        """
        # Convert tolerance percentage to string
        price_tolerance_str = f"{PRICE_TOLERANCE_PERCENT * 100:.2f}%"
        date_tolerance_str = f"{DATE_TOLERANCE_DAYS} day(s)"
        
        # Get possible break types from config
        possible_answers = ", ".join(MISMATCH_TYPES)
        
        # Use triple quotes for the multi-line f-string
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

    def _build_user_query(self, sample: Dict[str, Any]) -> str:
        """Creates the user query containing the specific trade/position data."""
        trades = sample.get('trades', [])
        position = sample.get('position', {})
        
        formatted_trades = self._format_trades(trades)
        formatted_position = self._format_position(position)
        
        user_query = (
            f"Please reconcile the following trades and position:\n\n"
            f"Trades:\n{formatted_trades}\n\n"
            f"Position:\n{formatted_position}"
        )
        return user_query

    def reset(self) -> str:
        """
        Resets the environment to the next state (data sample) and returns the new observation (prompt).
        Handles moving to the next sample in the dataset.
        
        Returns:
            A formatted string combining the system prompt and user query for the LLM.
        """
        self.current_index = (self.current_index + 1) % self.dataset_size
        if self.current_index == 0 and self.dataset_size > 1:
            print("Environment dataset wrapped around.") # Optional: Log epoch completion
            
        self.current_ground_truth = self.dataset[self.current_index]
        
        system_prompt = self._build_system_prompt()
        user_query = self._build_user_query(self.current_ground_truth)
        
        # Combine prompts (standard practice)
        # Depending on the LLM API, you might need to format this differently 
        # (e.g., as a list of messages with roles)
        full_prompt = f"{system_prompt}\n\nUSER: {user_query}\nASSISTANT:"
        
        return full_prompt # This is the observation

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Takes an action (LLM output string), calculates the reward, 
        and returns the next state transition.

        Args:
            action: The raw string output from the language model.

        Returns:
            A tuple (next_observation, reward, done, info):
            - next_observation (str): The prompt for the *next* data sample.
            - reward (float): The reward calculated for the given action.
            - done (bool): Always True for this environment (each step is one episode).
            - info (dict): Additional information (ground truth, etc.).
        """
        if self.current_ground_truth is None:
            raise RuntimeError("Environment must be reset before stepping.")

        # Calculate reward using the imported function
        reward = compute_score(
            llm_output_str=action, 
            ground_truth_sample=self.current_ground_truth,
            aggregation_mode=self.aggregation_mode # Pass mode to reward function
        )
        
        # Determine if the task is done (always True after one step in this setup)
        done = True 
        
        # Prepare info dictionary (optional)
        info = {
            'ground_truth_label': self.current_ground_truth.get('label'),
            'ground_truth_details': self.current_ground_truth.get('mismatch_details')
            # Add more debug info if needed (e.g., parsed answer from action)
        }
        
        # Get the observation (prompt) for the *next* step
        # Note: This resets the state to the next sample immediately.
        # Some RL frameworks might handle this differently (e.g., expect the observation
        # corresponding to the *end* of the current step, which might be None if done=True).
        # Adapt if necessary based on Verl's expected behavior.
        next_observation = self.reset()

        return next_observation, reward, done, info

# Example usage (optional)
if __name__ == '__main__':
    print("\n--- Running Environment Example --- DONT FORGET TO CREATE A SPEC.MD")
    # Use the correct path relative to the root when running with 'python -m'
    data_file_path = 'financial_rec/data/synthetic_data.json' 
    
    try:
        # Initialize for the validation split
        env = FinancialRecEnv(data_path=data_file_path, split='val', aggregation_mode='partial')
        
        # Get the first prompt
        initial_prompt = env.reset()
        print("\n--- Initial Prompt (Observation) ---")
        print(initial_prompt)
        
        # Simulate an LLM action (replace with actual LLM call in practice)
        simulated_action = "<think>Comparing trade 1 quantity 100 and price 50.5 to position quantity 100 and price 50.5. Dates match. Looks ok.</think><answer>NoBreak</answer>"
        print(f"\n--- Simulated LLM Action ---\n{simulated_action}")
        
        # Take a step
        next_prompt, reward, done, info = env.step(simulated_action)
        
        print(f"\n--- Step Result ---")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        # print("\n--- Next Prompt (Observation) ---") # Often long, uncomment if needed
        # print(next_prompt)
        
        # Reset again to see the next sample
        next_prompt_2 = env.reset()
        # print("\n--- Prompt after second reset ---") # Often long, uncomment if needed
        # print(next_prompt_2)
        print("\nEnvironment example run complete.")
        
    except Exception as e:
        print(f"\nAn error occurred during environment example: {e}")
        import traceback
        traceback.print_exc() 