import os
import json
import random
from typing import List


def extract_from_knowledge_crossword(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the knowledge crossword dataset

    Args:
        file_path: Path to the JSONL file
        num: Number of questions to extract

    Returns:
        List of instruction strings
    """
    try:
        instructions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'prompt' in item:
                        instructions.append(item['prompt'])
                    else:
                        print(f"Warning: Missing 'prompt' field in an item")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in file")
                    continue

        if not instructions:
            print(f"Warning: No valid instructions found in {file_path}")
            return []

        # Check if num is greater than available instructions
        if num > len(instructions):
            print(f"Warning: Requested {num} instructions but only {len(instructions)} are available")
            return instructions  # Return all available instructions

        # Shuffle and limit the number of instructions
        random.shuffle(instructions)
        return instructions[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []


def extract_from_alpaca(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the alpaca dataset

    Args:
        file_path: Path to the JSON file
        num: Number of questions to extract

    Returns:
        List of instruction strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract instructions from the data
        instructions = []
        for item in data:
            if isinstance(item, dict) and 'instruction' in item:
                instructions.append(item['instruction'])

        # Check if we found any instructions
        if not instructions:
            print(f"Warning: No instructions found in {file_path}")
            return []

        # Shuffle and limit the number of instructions
        random.shuffle(instructions)
        return instructions[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []


def extract_from_culture(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the culture dataset, only from the 'train' split.

    Args:
        file_path: Path to the JSON file
        num: Number of questions to extract

    Returns:
        List of instruction strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Only extract from the 'train' split
        questions = []

        if 'train' in data:
            train_data = data['train']
            # For each item in the train split (indexed by IDs)
            for item_id in train_data:
                item = train_data[item_id]
                if 'instruction' in item:
                    questions.append(item['instruction'])
        else:
            print("Warning: No 'train' split found in the data")

        # Shuffle and limit the number of questions
        random.shuffle(questions)
        return questions[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return []


def extract_from_json_truthfulqa(file_path: str, num: int) -> List[str]:
    """
    From the JSONL file, extract the specified number of unique questions.

    Args:
        file_path (str): Path to the JSONL file
        num (int): Number of questions to extract

    Returns:
        List[str]: Extracted questions list (format: Q: ...\nA: ...)

    Raises:
        FileNotFoundError: When the file does not exist
        json.JSONDecodeError: When JSON parsing fails
    """
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    item = json.loads(line.strip())

                    # Extract question from prompt field
                    if 'prompt' not in item:
                        print(f"Warning: Missing 'prompt' field at line {line_num}")
                        continue

                    questions.append(item['prompt'])

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")
                    continue

        if not questions:
            print(f"Warning: No valid questions found in {file_path}")
            return []

        # Remove duplicates and shuffle
        unique_questions = list(set(questions))
        random.shuffle(unique_questions)

        # Handle requested number
        available_num = len(unique_questions)
        if num > available_num:
            print(f"Warning: Requested {num} questions but only {available_num} available")

            return unique_questions
        print(unique_questions[:num])
        return unique_questions[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error while processing file: {str(e)}")
        return []


def extract_from_json_gsm(file_path: str, num: int) -> List[str]:
    """
    Extract a specified number of unique inputs from a JSON file.

    Args:
        file_path (str): Path to the JSON file
        num (int): Number of inputs to extract

    Returns:
        List[str]: List of extracted inputs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            inputs = [item['input'] for item in data]

        random.shuffle(inputs)

        return inputs[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting inputs: {e}")
        return []


def extract_from_math(file_path: str, num: int, difficulty: str) -> List[str]:
    """
    Extract questions from the math dataset

    Args:
        file_path (str): Path to the JSONL file
        num (int): Number of problems to extract
        difficulty (str): Difficulty level ('easy', 'medium', or 'hard')

    Returns:
        List[str]: List of extracted problems with formatted prompts
    """
    try:
        problems = []
        if difficulty == 'easy':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'easy.jsonl')
        elif difficulty == 'medium':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'medium.jsonl')
        elif difficulty == 'hard':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'hard.jsonl')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt = data['problem'] + "Let's solve this problem step by step."
                problems.append(prompt)

        if num > len(problems):
            print(f"Warning: Requested {num} problems but only {len(problems)} available")
            random.shuffle(problems)
            return problems

        random.shuffle(problems)
        return problems[:num]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting problems: {e}")
        return []


def extract_from_com2(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the com2 dataset
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            instructions = [item['prompt'] for item in data]

        if num > len(instructions):
            print(f"Warning: Requested {num} instructions but only {len(instructions)} available")
            random.shuffle(instructions)
            return instructions

        random.shuffle(instructions)
        return instructions[:num]
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []


# Registry mapping task names to (extractor_function, file_pattern, extra_kwargs)
TASK_REGISTRY = {
    'gsm8k': {
        'extractor': extract_from_json_gsm,
        'file_pattern': 'gsm8k/questions.json',
    },
    'truthfulqa_mc1,truthfulqa_mc2': {
        'extractor': extract_from_json_truthfulqa,
        'file_pattern': 'truthfulqa/data_train.jsonl',
    },
    'culture_country': {
        'extractor': extract_from_culture,
        'file_pattern': 'culture/country_dataset.json',
    },
    'culture_value': {
        'extractor': extract_from_culture,
        'file_pattern': 'culture/country_value_dataset.json',
    },
    'culture_rule_of_thumb': {
        'extractor': extract_from_culture,
        'file_pattern': 'culture/rule_of_thumb_dataset.json',
    },
    'alpaca': {
        'extractor': extract_from_alpaca,
        'file_pattern': 'alpaca/alpaca_data_processed.json',
    },
    'kc_knowledge': {
        'extractor': extract_from_knowledge_crossword,
        'file_pattern': 'knowledge_crossword/KC_train_knowledge.jsonl',
    },
    'kc_zeroshot': {
        'extractor': extract_from_knowledge_crossword,
        'file_pattern': 'knowledge_crossword/KC_train_zeroshot.jsonl',
    },
    'math': {
        'extractor': extract_from_math,
        'file_pattern': 'MATH',
    },
    'com2': {
        'extractor': extract_from_com2,
        'file_pattern': 'com2/train.json',
    },
}


def load_instructions(task: str, data_dir: str, num_instructions: int, difficulty: str = 'easy') -> List[str]:
    """
    Load instructions for a given task by dispatching to the appropriate extractor.

    Args:
        task (str): Task name. Must be a key in TASK_REGISTRY.
        data_dir (str): Base directory containing the data files.
        num_instructions (int): Number of instructions to extract.
        difficulty (str): Difficulty level for tasks that support it (default: 'easy').

    Returns:
        List[str]: Extracted instructions.

    Raises:
        ValueError: If the task name is not found in TASK_REGISTRY.
    """
    if task not in TASK_REGISTRY:
        valid_tasks = ', '.join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Invalid task: {task}. Valid tasks are: {valid_tasks}")

    entry = TASK_REGISTRY[task]
    extractor = entry['extractor']
    file_pattern = entry['file_pattern']
    file_path = os.path.join(data_dir, file_pattern)

    if task == 'math':
        return extractor(file_path, num_instructions, difficulty)
    else:
        return extractor(file_path, num_instructions)
