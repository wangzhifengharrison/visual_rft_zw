import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    return iou_results

def sort_and_calculate_label(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    label_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            matched_list1_indices.add(matched_bbox1)
            # 检查标签是否一致
            label_match = bbox2['Label'] == list1[matched_bbox1]['Label']
            label_results.append((label_match, bbox2['Confidence']))
        else:
            label_results.append((False, bbox2['Confidence']))

    return label_results

def sort_and_calculate_label_and_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    label_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
            # 检查标签是否一致
            label_match = bbox2['Label'] == list1[matched_bbox1]['Label']
            label_results.append((label_match, bbox2['Confidence']))
        else:
            iou_results.append((0, bbox2['Confidence']))
            label_results.append((False, bbox2['Confidence']))

    return iou_results, label_results


def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes
def get_bounding_box(sol, content):
    # Extract answer from solution if it has think/answer tags
    ground_truth = sol.strip()
    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content) 
    # content: <think> The image shows a man dressed in a suit, standing in what appears to be a formal or somber setting. Given the context and his posture, it seems likely he is feeling sad or contemplative. The man has a furrowed brow and downcast eyes, which are common signs of sadness. </think>
    # <answer>[{'Position': [190, 12, 455, 360], 'Confidence': 0.89, 'Label': 'sad'}]</answer>
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_answer = '<answer>'+student_answer+'</answer>'

    # fix format error
    student_answer = student_answer.replace("[[",'[')  
    student_answer = student_answer.replace("]]",']')  
    student_answer = student_answer.replace("\n",'')  
    # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9, 'Label': 'sad'}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8, 'Label': 'happy'}]
    ground_truth_bbox = extract_bbox(ground_truth)
    student_answer_bbox = extract_bbox(student_answer)

    return student_answer_bbox, ground_truth_bbox



def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_label(label_results, len_gt):
    label_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(label_results)):
        temp_label = label_results[i][0]
        temp_confidence = label_results[i][1]

        temp_label_reward = 0.95 if temp_label else 0.05
        label_reward += temp_label_reward
    
    if len_gt>=len(label_results):
        label_reward = label_reward/len_gt
    else:
        label_reward = label_reward/len(label_results)
    return label_reward

def compute_reward_confidence(iou_results, label_results):
    iou_reward = 0.0
    label_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_iou_confidence = iou_results[i][1]
        temp_iou_reward = temp_iou

        temp_label = label_results[i][0]
        temp_label_value = 0.95 if temp_label else 0.05
        temp_label_confidence = label_results[i][1]
        temp_label_reward = temp_label_value

        if temp_iou == 0:
            temp_iou_confidence_reward = (1-temp_iou)*(1-temp_iou_confidence)
        else:
            temp_iou_confidence_reward = temp_iou_confidence

        if temp_label_value == 0.05:
            temp_label_confidence_reward = (1-temp_label_value)*(1-temp_label_confidence)
        else:
            temp_label_confidence_reward = temp_label_confidence


        iou_reward += temp_iou_reward
        label_reward += temp_label_reward
        confidence_reward += (0.5 * temp_iou_confidence_reward + 0.5 * temp_label_confidence_reward)
        
    iou_reward = iou_reward/len(iou_results)
    label_reward = label_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward


#### reward functions
def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = 0.05
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 0.95 #1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                student_answer_bbox, ground_truth_bbox = get_bounding_box(sol, content)
                
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    ### new iou reward
                    reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    # clip to [baseline, 0.95]
                    reward = max(0.05, min(reward, 0.95))
                    # if reward>1:
                    #     reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # change the reward value 
        rewards.append(reward)
    return rewards

def accuracy_reward_confidence(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # reward = 0.0
        reward = 0.05
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 0.95 #1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                student_answer_bbox, ground_truth_bbox = get_bounding_box(sol, content)

                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_label_and_iou(ground_truth_bbox, student_answer_bbox)
                    reward = compute_reward_confidence(iou_results, label_results)
                    reward = max(0.05, min(reward, 0.95))
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # 
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # baseline = 0.1
    # max_gain = 1.0 - baseline
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.95 if match else 0.05 for match in matches]

def accuracy_reward_label(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.05
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 0.95 #1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        label_results = []

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                student_answer_bbox, ground_truth_bbox = get_bounding_box(sol, content)

                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    label_results = sort_and_calculate_label(ground_truth_bbox, student_answer_bbox)
                    ### new iou reward
                    reward = compute_reward_label(label_results, len(ground_truth_bbox))
                    # clip to [baseline, 0.95]
                    reward = max(0.05, min(reward, 0.95))
                    # if reward>1:
                    #     reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # change the reward value 
        rewards.append(reward)
    return rewards

###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
    "accuracy_label": accuracy_reward_label,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format','accuracy_label']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)