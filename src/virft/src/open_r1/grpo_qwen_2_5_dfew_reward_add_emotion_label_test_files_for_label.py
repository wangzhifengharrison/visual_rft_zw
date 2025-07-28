import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
#
# from math_verify import parse, verify
# # from open_r1.trainer import Qwen2VLGRPOTrainer
# from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
# from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json



# [{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'},{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, {'Position': [308, 9, 487, 188], 'Confidence': 0.93, 'Label': 'happy'}, {'Position': [505, 126, 675, 296], 'Confidence': 0.92, 'Label': 'happy'}, {'Position': [640, 10, 760, 160], 'Confidence': 0.88, 'Label': 'happy'}, {'Position': [765, 9, 900, 144], 'Confidence': 0.9, 'Label': 'happy'}]


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

def extract_bbox2(response):
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


def sort_and_calculate_iou(list1, list2, iou_threshold=0.09):
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

        print(130, best_iou, bbox2, bbox1)
        if best_iou > iou_threshold:
            print('132-------------', bbox2['Label'], matched_bbox1, list1[matched_bbox1]['Label'], bbox2['Label'] == list1[matched_bbox1]['Label'])
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
            print('135')

            # 检查标签是否一致
            label_match = bbox2['Label'] == list1[matched_bbox1]['Label']
            print('136', label_match, bbox2['Label'], list1[matched_bbox1]['Label'])
            print('------------------', label_match)
            label_results.append((label_match, bbox2['Confidence']))

        else:
            iou_results.append((0, bbox2['Confidence']))
            label_results.append((False, bbox2['Confidence']))

    print(label_results, '146')
    ### [(0.7192676547515258, 1.0), (0, 0.7)] best_iou,bbox2['Confidence']
    # print("134, iou_results: ", iou_results)
    return iou_results, label_results


def compute_reward_label(label_results, len_gt):
    label_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(label_results)):
        temp_label = label_results[i][0]
        temp_confidence = label_results[i][1]
        print('158 ^^^^^^^^^^^^^^^^^^^^', temp_label)

        temp_iou_reward = 0.95 if temp_label else 0.05
        label_reward += temp_iou_reward
    
    if len_gt>=len(label_results):
        label_reward = label_reward/len_gt
    else:
        label_reward = label_reward/len(label_results)
    return label_reward

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


def accuracy_reward_label(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        # reward = 0.0
        # baseline to avoid zero reward
        # <answer>[{'Position': [190, 12, 455, 360], 'Confidence': 0.89, 'Label': 'sad'}]</answer>
        # sol: <answer>[{'Position': [295, 74, 393, 224], 'Confidence': 1, 'Label': neutral}]</answer>
        # student_answer_bbox: [{'Position': [190, 12, 455, 360], 'Confidence': 0.89, 'Label': 'sad'}]
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
        label_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                show_flage = 1

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
                student_answer_bbox = extract_bbox2(student_answer)
                print('213', student_answer_bbox)

                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    print('********242********', label_results)

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
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a") as f:
        #         f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
        #         f.write(f"content: {content}\n")
        #         f.write(f"sol: {sol}\n")
        #         f.write(f"completions: {completions}\n")
        #         if show_flage==1:
        #             f.write(f"student_answer_bbox: {student_answer_bbox}\n")
        #             f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
        #             if student_answer_bbox!=None:
        #                 f.write(f"label_results: {label_results}\n")
        # show_flage = 0 
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

completions = [
    [{'role': 'assistant', 'content': "<think> The person on the left is wearing a dark hoodie, and you are looking at the side view of their back. The person on the right is facing the camera and raising their right hand with the palm facing outwards. The person on the right has a more upright posture compared to the person on the left. The context seems to depict a tense or aggressive situation, which is supported by the person on the right's raised hand and the overall dark, confined environment. </think>\n<answer>[{'Position': [280, 47, 577, 332], 'Confidence': 0.96, 'Label': 'angry'}]</answer>"}],
     [{'role': 'assistant', 'content': "<think> The image shows two people: one with short hair and one with medium-length hair. The person with short hair is holding out their hand, while the person with medium-length hair is standing close. The person with short hair is wearing a dark shirt, and the person with medium-length hair is wearing a green shirt. The scene has a somber or serious atmosphere, as suggested by the posture and the context of what could be a struggle or confrontation. The person with short hair is likely the aggressor, given the posture and the act of holding out their hand.\n\nTo determine the emotion of the individual's face, I will analyze the facial expression, posture, and context. The person with short hair has their hands raised, which typically indicates a defensive or aggressive stance. However, the person with medium-length hair appears to be looking down in a somewhat contemplative or defeated manner. The context suggests a possibly intense or pivotal moment, which could be due to confrontation or struggle. Given these observations, the emotion seems to be one of neutral or contemplation.\n\nThe face's emotion can be classified as neutral, as it seems to express no clear display of fear, anger, happiness, sadness, disgust, surprise, or disgust. </think>\n<answer>[{'Position"}], 
     [{'role': 'assistant', 'content': "<think> In the image, I observe a woman with her hand raised, as if she is reaching for something or someone. Her expression appears to be one of determination or urgency, possibly indicating that she is about to perform a significant action, such as pushing a door open. There's also a man with his back turned facing her, and another man with his hand raised to his mouth. The woman's posture suggests she is in control and is likely trying to persuade the man nearby to let her pass. </think>\n<answer>[{'Position': [280, 31, 579, 280], 'Confidence': 0.791, 'Label': 'fear'}]</answer>"}],
     [{'role': 'assistant', 'content': '<think> The image shows two people in a dimly lit, outdoor setting. The person on the left is holding something up in the air with their right hand, and their facial expression appears to be one of fear or surprise. The person on the right has their left hand slightly raised as if gesturing or waving. The overall context suggests a scene of urgency or surprise. Given the posture and expression, it is likely that the person on the right is the one speaking, but the exact emotion cannot be determined with certainty without a visible facial expression. </think> \n<answer>[\n    {"Position": [280, 33, 609, 362], "Confidence": 0.95, "Label": "surprise"},\n    {"Position": [0, 64, 360, 364], "Confidence": 0.85, "Label": "fear"}\n]\n</answer>'}
     ]]
solution = ["<answer>[{'Position': [395, 57, 468, 160], 'Confidence': 1, 'Label': 'fear'}]</answer>",
 "<answer>[{'Position': [395, 57, 468, 160], 'Confidence': 1, 'Label': 'fear'}]</answer>",
  "<answer>[{'Position': [395, 57, 468, 160], 'Confidence': 1, 'Label': 'fear'}]</answer>",
   "<answer>[{'Position': [395, 57, 468, 160], 'Confidence': 1, 'Label': 'fear'}]</answer>"]
accuracy_reward_label_print= accuracy_reward_label(completions, solution)

print('222', accuracy_reward_label_print)

# python grpo_qwen_2_5_dfew_reward_add_emotion_label_test_files_for_label.py







# [{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, ...]
#  conda activate Visual-RFT
#  cd /scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1
#  python grpo_qwen_2_5_dfew_reward_add_emotion_label_test_files_for_label.py
# qsub -I -P kf09 -q gpuvolta -l ngpus=1,ncpus=12,mem=64GB,walltime=00:50:00