import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import time 

# Import helper to parse LLM output (from reward script)
# Ensure this path is correct based on your project structure and PYTHONPATH
try:
    from verl.utils.reward_score.financial_rec import parse_llm_output
except ImportError:
    print("Warning: Could not import parse_llm_output from verl.utils.reward_score.financial_rec")
    # Define a basic fallback parser if needed, or rely on user having correct path
    def parse_llm_output(llm_output_str):
        import re
        THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        think_match = THINK_PATTERN.search(llm_output_str)
        answer_match = ANSWER_PATTERN.search(llm_output_str)
        think_content = think_match.group(1).strip() if think_match else None
        answer_content = answer_match.group(1).strip() if answer_match else None
        return think_content, answer_content

def evaluate(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 32, # vLLM uses continuous batching, this is more like a guide
    max_new_tokens: int = 512, # Match response length used in training config if possible
    temperature: float = 0.0, # Use 0 for deterministic greedy decoding
    top_p: float = 1.0,
    top_k: int = -1,
):
    """
    Evaluates a fine-tuned model on the financial reconciliation test set.

    Args:
        model_path: Path to the fine-tuned model checkpoint directory.
        test_data_path: Path to the test Parquet file (e.g., test.parquet).
        output_dir: Directory to save evaluation results (e.g., predictions.csv).
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: GPU memory fraction for vLLM.
        batch_size: Target batch size for processing prompts.
        max_new_tokens: Maximum number of tokens to generate for the response.
        temperature: Sampling temperature (0 for greedy).
        top_p: Top-p sampling nucleus.
        top_k: Top-k sampling.
    """
    print(f"Starting evaluation for model: {model_path}")
    print(f"Using test data: {test_data_path}")

    # --- 1. Load Data --- 
    try:
        df_test = pd.read_parquet(test_data_path)
        # Ensure 'prompt' and 'ground_truth_sample' columns exist
        if 'prompt' not in df_test.columns or 'ground_truth_sample' not in df_test.columns:
            raise ValueError(f"Missing required columns 'prompt' or 'ground_truth_sample' in {test_data_path}")
        prompts = df_test['prompt'].tolist()
        ground_truth_samples = df_test['ground_truth_sample'].tolist()
        # Extract ground truth labels for easier comparison later
        ground_truth_labels = [sample.get('label', '[MISSING_LABEL]') for sample in ground_truth_samples]
        print(f"Loaded {len(prompts)} test samples.")
    except Exception as e:
        print(f"Error loading test data from {test_data_path}: {e}")
        return

    # --- 2. Initialize vLLM --- 
    print("Initializing vLLM engine...")
    start_time = time.time()
    try:
        # trust_remote_code=True might be needed for some models like Qwen
        llm = LLM(
            model=model_path, 
            tensor_parallel_size=tensor_parallel_size, 
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True, 
            # swap_space=4 # Optional: configure swap space if memory is tight
        )
        sampling_params = SamplingParams(
            n=1, # Generate one sequence per prompt
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            # stop=["</answer>"] # Optional: Stop generation early if desired
        )
    except Exception as e:
        print(f"Error initializing vLLM from {model_path}: {e}")
        return
    init_time = time.time() - start_time
    print(f"vLLM engine initialized in {init_time:.2f} seconds.")

    # --- 3. Generate Responses --- 
    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    # vLLM handles batching internally, just pass the list of prompts
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_time
    print(f"Finished generation in {gen_time:.2f} seconds ({len(prompts)/gen_time:.2f} prompts/sec).")

    # --- 4. Process Outputs and Calculate Metrics --- 
    predictions = []
    parse_errors = 0
    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
        # The generated text is in output.outputs[0].text
        generated_text = output.outputs[0].text
        # Use the imported parser
        _, answer_content = parse_llm_output(generated_text)
        
        if answer_content:
            predictions.append(answer_content)
        else:
            predictions.append("[PARSE_ERROR]") # Placeholder for failed parsing
            parse_errors += 1
            # Optional: Log the problematic output
            # print(f"Warning: Could not parse answer from output {i}:\n{generated_text[:500]}...")

    print(f"Finished processing. Encountered {parse_errors} parsing errors.")
    
    # Ensure the lengths match
    if len(predictions) != len(ground_truth_labels):
        print(f"Error: Mismatch between number of predictions ({len(predictions)}) and ground truth labels ({len(ground_truth_labels)}).")
        return

    # Get all unique labels for confusion matrix
    all_labels = sorted(list(set(ground_truth_labels + predictions)))

    # Calculate metrics
    accuracy = accuracy_score(ground_truth_labels, predictions)
    conf_matrix = confusion_matrix(ground_truth_labels, predictions, labels=all_labels)
    class_report = classification_report(ground_truth_labels, predictions, labels=all_labels, zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    # Format confusion matrix nicely with labels
    conf_matrix_df = pd.DataFrame(conf_matrix, index=all_labels, columns=all_labels)
    print(conf_matrix_df)

    # --- 5. Save Results (Optional) --- 
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame({
            'prompt': prompts,
            'ground_truth_label': ground_truth_labels,
            'predicted_label': predictions,
            # Include original sample for detailed analysis if needed
            # 'ground_truth_sample': ground_truth_samples 
        })
        output_path = os.path.join(output_dir, "evaluation_predictions.csv")
        try:
            results_df.to_csv(output_path, index=False)
            print(f"\nSaved detailed predictions to {output_path}")
        except Exception as e:
            print(f"\nError saving predictions to {output_path}: {e}")
            
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Financial Reconciliation model.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model checkpoint directory (output from training).")
    parser.add_argument("--test_data_path", type=str, default="financial_rec/data/test.parquet",
                        help="Path to the pre-processed test data Parquet file.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation outputs (e.g., predictions CSV).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for vLLM tensor parallelism.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction for vLLM.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Target batch size (vLLM uses continuous batching).")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate in the response.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding).")
    
    args = parser.parse_args()

    # Basic check for model path existence
    if not os.path.isdir(args.model_path):
        print(f"Error: Model directory not found at {args.model_path}")
    # Basic check for test data existence
    elif not os.path.exists(args.test_data_path):
         print(f"Error: Test data file not found at {args.test_data_path}")
    else:
        evaluate(
            model_path=args.model_path,
            test_data_path=args.test_data_path,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ) 