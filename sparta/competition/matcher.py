import os
import json
import random
import logging
from typing import List, Dict
from datetime import datetime
from time import sleep

import torch
from multiprocessing import Pool

from sparta.inference.engine import Inference
from sparta.model_init.init import ModelInit, ModelInitFair
from sparta.rating.tracker import RatingTracker, StopCriteria
from sparta.rating.system import RatingSystem, RatingSystemDynamicWeighted, RatingSystemStaticWeighted
from sparta.judge.judge import Judge, run_judges, calculate_judge_averages


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_model_group(args):
    """
    process the model group on a single GPU
    """
    group, model_tasks, gpu_id, model_paths, model_types = args  # model_types: dict
    model_responses = {}

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

        for model_name in group:
            if model_name not in model_tasks or not model_tasks[model_name]:
                continue

            unique_tasks = list(set(model_tasks[model_name]))
            print(f"Processing model {model_name} on GPU {gpu_id}")

            try:
                inference = Inference(
                    model_name=model_name,
                    gpu_id=gpu_id,
                    model_path=model_paths[model_name],
                    base_model=model_types[model_name]  # use the model_type of the model
                )

                result = inference.batch_generate_responses(
                    instructions=unique_tasks,
                    batch_size=24,
                    max_new_tokens=512,
                    use_chat_template=True
                )

                model_responses[model_name] = {
                    instruction: response
                    for instruction, response in zip(unique_tasks, result)
                }

            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue

            finally:

                if 'inference' in locals():
                    del inference.model
                    del inference
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sleep(3)

        return model_responses

    except Exception as e:
        print(f"Error in process_model_group: {e}")
        return {}

    finally:
        cleanup_resources()


class Competition:
    def __init__(self, model_configs, model_info, num_opponents, random_match_prob, is_random_select=True):
        self.model_configs = model_configs
        self.model_names = [config["name"] for config in model_configs]
        self.model_paths = {config["name"]: config["path"] for config in model_configs}
        self.model_types = {config["name"]: config["model_type"] for config in model_configs}
        self.model_info = model_info
        self.num_opponents = num_opponents
        self.random_match_prob = random_match_prob
        self.is_random_select = is_random_select

    def generate_idx(self, model_name: str, instructions: List[str], gpu_id: int) -> List[str]:
        inference = None
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, -1))

            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

            model_path = self.model_paths[model_name]
            model_type = self.model_types[model_name]
            inference = Inference(
                model_name=model_name,
                gpu_id=gpu_id,
                model_path=model_path,
                base_model=model_type
            )

            responses = inference.batch_generate_responses(
                instructions=instructions,
                batch_size=12,
                max_new_tokens=512,
                use_chat_template=True
            )

            return responses

        except Exception as e:
            print(f"Generate error for {model_name}: {e}")
            return [f"Error: {str(e)}"] * len(instructions)

        finally:
            if inference is not None:
                del inference.model
                del inference
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_opponent(self, current_model: str) -> str:
        """choose the opponent based on the score"""
        current_score = self.model_info[current_model]['score']

        if random.random() < self.random_match_prob:
            opponents = [m for m in self.model_names if m != current_model]
            return random.choice(opponents)

        potential_opponents = []
        for other_model in self.model_names:
            if other_model != current_model:
                score_diff = abs(current_score - self.model_info[other_model]['score'])
                potential_opponents.append((other_model, score_diff))

        potential_opponents.sort(key=lambda x: x[1])
        print(f"random_match_prob: {self.random_match_prob}")
        return random.choice(potential_opponents[:self.num_opponents])[0]

    def run(self, instructions: List[str]) -> List[Dict]:
        model_tasks = {model: [] for model in self.model_names}
        instruction_pairs = []

        if self.is_random_select:
            # Calculate number of complete loops
            num_loops = len(instructions) // len(self.model_names)

            # Process each complete loop
            for loop_idx in range(num_loops):
                # Get instructions for this loop
                start_idx = loop_idx * len(self.model_names)
                end_idx = start_idx + len(self.model_names)
                loop_instructions = instructions[start_idx:end_idx]

                # Randomly shuffle models for this loop
                loop_models = random.sample(self.model_names, len(self.model_names))

                # Create pairs for each model in the shuffled order
                for model_idx, first_model in enumerate(loop_models):
                    instruction = loop_instructions[model_idx]
                    opponent_model = self.get_opponent(first_model)

                    instruction_pairs.append((instruction, first_model, opponent_model))
                    model_tasks[first_model].append(instruction)
                    model_tasks[opponent_model].append(instruction)

            # Handle remaining instructions
            remaining_start = num_loops * len(self.model_names)
            for idx, instruction in enumerate(instructions[remaining_start:]):
                first_model = random.choice(self.model_names)
                opponent_model = self.get_opponent(first_model)

                instruction_pairs.append((instruction, first_model, opponent_model))
                model_tasks[first_model].append(instruction)
                model_tasks[opponent_model].append(instruction)
        else:
            # Original sequential selection logic
            for idx, instruction in enumerate(instructions):
                first_model = self.model_names[idx % len(self.model_names)]
                opponent_model = self.get_opponent(first_model)

                instruction_pairs.append((instruction, first_model, opponent_model))
                model_tasks[first_model].append(instruction)
                model_tasks[opponent_model].append(instruction)

        # Process model responses in groups
        model_groups = []
        models_list = list(self.model_names)
        for i in range(0, len(models_list), 2):
            if i + 1 < len(models_list):
                model_groups.append((models_list[i], models_list[i+1]))
            else:
                model_groups.append((models_list[i],))

        model_responses = {}
        with Pool(processes=len(model_groups)) as pool:
            process_args = [
                (group, model_tasks, gpu_id, self.model_paths, self.model_types)
                for gpu_id, group in enumerate(model_groups)
            ]

            results = pool.map(process_model_group, process_args)

            for group_responses in results:
                model_responses.update(group_responses)

        # Generate raw pairs
        raw_pairs = []
        for instruction, model_a, model_b in instruction_pairs:
            if (model_a in model_responses and
                model_b in model_responses and
                instruction in model_responses[model_a] and
                instruction in model_responses[model_b]):

                response_list = [
                    {
                        'model_name': model_a,
                        'response': model_responses[model_a][instruction]
                    },
                    {
                        'model_name': model_b,
                        'response': model_responses[model_b][instruction]
                    }
                ]
                raw_pairs = self.pair(raw_pairs, instruction, response_list)

        return raw_pairs

    def pair(self, raw_pairs: List[Dict], instruction: str, response_list: List[Dict]) -> List[Dict]:
        """
        organize the responses into pairs

        Args:
            raw_pairs (List[Dict]): existing pairs list
            instruction (str): current instruction
            response_list (List[Dict]): list of responses, each element is a dictionary containing 'model_name' and 'response'

        Returns:
            List[Dict]: updated pairs list
        """
        # check if the response list is valid
        if len(response_list) != 2:
            print(f"Invalid response list length: {len(response_list)}")
            return raw_pairs

        new_pair = {
            'instruction': instruction,
            'models': [resp['model_name'] for resp in response_list],
            'responses': [resp['response'] for resp in response_list],
            'judges': {}
        }

        raw_pairs.append(new_pair)
        return raw_pairs


def judge_with_gpu(args):
    """
    Helper function for multiprocessing
    Args:
        args: tuple of (judge_name, judge_path, pairs, gpu_id)
    Returns:
        tuple: (judge_name, judged_pairs)
    """
    judge_name, pairs, judge_path, gpu_id = args
    judge = Judge(judge_name, judge_path, gpu_id)
    judged_pairs = judge.judge(pairs)
    return judge_name, judged_pairs


def save_judged_pairs(judged_pairs, base_dir, iteration):
    """
    save the judged pairs to the specified directory

    Args:
        judged_pairs (List[Dict]): judged pairs
        base_dir (str): base directory
        iteration (int): current iteration
    """
    try:
        save_dir = os.path.join(base_dir, f"iteration_{iteration}", "judged_results")
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, "judged_pairs.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)

        print(f"\nJudged pairs saved to: {file_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(save_dir, f"judged_pairs_{timestamp}.json")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)

        print(f"Backup saved to: {backup_path}")

    except Exception as e:
        print(f"Error saving judged pairs: {e}")


def save_preference_pairs_to_json(preference_pairs, base_dir, filename):
    """
    Save preference pairs to a JSON file in the specified directory.

    Args:
        preference_pairs (list): The list of preference pairs to save
        base_dir (str): Base directory path
        filename (str): Name of the JSON file
    """
    try:
        os.makedirs(base_dir, exist_ok=True)

        file_path = os.path.join(base_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preference_pairs, f, indent=2, ensure_ascii=False)
        print(f"Preference pairs saved to {file_path}")

    except Exception as e:
        print(f"Error saving preference pairs to JSON: {e}")


def save_rating_history(rating_history, base_dir, iteration):
    """
    Save the detailed rating history to a JSON file in the specified directory.

    Args:
        rating_history (list): List of rating snapshots after each pair
        base_dir (str): Base directory path
        iteration (int): Current iteration number
    """
    try:
        os.makedirs(base_dir, exist_ok=True)

        file_path = os.path.join(base_dir, f"iteration_{iteration}_rating_history.json")

        history_data = {
            'iteration': iteration,
            'total_pairs': len(rating_history),
            'history': rating_history,
            'final_ratings': rating_history[-1]['ratings'] if rating_history else None,
            'timestamp': datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed rating history saved to {file_path}")

    except Exception as e:
        print(f"Error saving rating history to JSON: {e}")


def cleanup_resources():
    """more thorough resource cleanup function"""
    try:

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass

        import gc
        gc.collect()

        import psutil
        process = psutil.Process()
        try:
            process.memory_full_info()
        except:
            pass

        sleep(5)

    except Exception as e:
        print(f"Error during cleanup: {e}")


def save_model_info(model_info: Dict, base_dir: str, iteration: int):
    """save model_info to JSON file"""
    try:
        save_path = os.path.join(base_dir, f"iteration_{iteration}", "model_info.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        serializable_info = {}
        for model_name, info in model_info.items():
            serializable_info[model_name] = {
                'score': float(info['score']),
                'deviation': float(info['deviation'])
            }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2)

        logger.info(f"Model info saved to {save_path}")

    except Exception as e:
        logger.error(f"Error saving model info: {e}")


def save_judge_pairs(judge_pairs, output_dir, iteration):
    """
    save judge_pairs to the specified directory

    Args:
        judge_pairs (list): list of judge pairs
        output_dir (str): output directory
        iteration (int): iteration number
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, f"iteration_{iteration}_judge_pairs.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(judge_pairs, f, indent=2, ensure_ascii=False)

        print(f"Judge pairs saved to: {filepath}")

    except Exception as e:
        print(f"Error saving judge pairs: {e}")


def load_model_info(base_dir: str, iteration: int) -> Dict:
    """load model_info from JSON file"""
    try:
        load_path = os.path.join(base_dir, f"iteration_{iteration}", "model_info.json")

        if os.path.exists(load_path):
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                model_name: {
                    'score': 100,
                    'deviation': 0.5
                }
                for model_name in ["code_alpaca", "cot", "flan_v2", "gemini_alpaca",
                                 "lima", "oasst1", "open_orca", "science",
                                 "sharegpt", "wizardlm"]
            }

    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        return None


def filter_tie(preference_pairs):
    """
    filter the tie pairs
    """
    return [pair for pair in preference_pairs if pair['score_diff'] != 0]
