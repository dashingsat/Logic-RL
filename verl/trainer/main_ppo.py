# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk
from verl.utils.reward_score import financial_rec
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    elif data_source == 'financial_rec':
        return financial_rec.compute_score
    else:
        raise NotImplementedError(f"No reward score function implemented for data_source: {data_source}")


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config
        self.aggregation_mode = self.config.get('custom_env', {}).get('aggregation_mode', 'partial') if self.config else 'partial'
        print(f"[RewardManager] Initialized with aggregation_mode: {self.aggregation_mode}")

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            print("Warning: Pre-computed 'rm_scores' found in batch, but RewardManager will recompute based on data_source.")
            # return data.batch['rm_scores'] # Commenting out to ensure our logic runs

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            score = 0.0 # Default score

            try:
                prompt_ids = data_item.batch['prompts']
                response_ids = data_item.batch['responses']
                attention_mask = data_item.batch['attention_mask']
                prompt_length = prompt_ids.shape[-1]
                
                total_valid_length = attention_mask.sum()
                valid_response_length = max(0, total_valid_length - prompt_length)
                
                valid_response_ids = response_ids[:valid_response_length]
                
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth_sample = data_item.non_tensor_batch.get('ground_truth_sample') 
                if ground_truth_sample is None:
                    print(f"Warning: 'ground_truth_sample' not found in non_tensor_batch for item {i}. Checking legacy location.")
                    ground_truth_sample = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth')
                    if ground_truth_sample is None:
                         print(f"Error: Ground truth missing entirely for item {i}. Skipping reward calculation.")
                         continue # Skip this item if no ground truth
                    elif not isinstance(ground_truth_sample, dict):
                         print(f"Error: Legacy ground truth is not a dict for item {i}. Skipping reward calculation.")
                         continue # Skip if legacy GT is not the expected dict

                if not isinstance(ground_truth_sample, dict):
                     print(f"Error: 'ground_truth_sample' is not a dictionary for item {i}. Skipping reward calculation.")
                     continue # Skip if GT is not a dict

                data_source = data_item.non_tensor_batch['data_source']
                compute_score_fn = _select_rm_score_fn(data_source)
                
                kwargs = {
                    'aggregation_mode': self.aggregation_mode
                }

                score = compute_score_fn(
                    llm_output_str=response_str, 
                    ground_truth_sample=ground_truth_sample, 
                    **kwargs
                )
                
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = score
                else:
                     print(f"Warning: Zero length response detected for item {i}. Assigning score 0.")

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    print(f"--- Debug Item {i} (DataSource: {data_source}) ---")
                    print(f"Prompt: ...{prompt_str[-200:]}")
                    print(f"Response: {response_str}")
                    print(f"Ground Truth Label: {ground_truth_sample.get('label', 'N/A')}")
                    print(f"Computed Score: {score}")
                    print(f"-------------------------------------------------")
            
            except Exception as e:
                print(f"Error processing reward for item {i}: {e}")
                import traceback
                traceback.print_exc()

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
