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

response = "<answer>[{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, {'Position': [308, 9, 487, 188], 'Confidence': 0.93, 'Label': 'happy'}, {'Position': [505, 126, 675, 296], 'Confidence': 0.92, 'Label': 'happy'}, {'Position': [640, 10, 760, 160], 'Confidence': 0.88, 'Label': 'happy'}, {'Position': [765, 9, 900, 144], 'Confidence': 0.90, 'Label': 'happy'}]</answer>"
# [{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'},{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, {'Position': [308, 9, 487, 188], 'Confidence': 0.93, 'Label': 'happy'}, {'Position': [505, 126, 675, 296], 'Confidence': 0.92, 'Label': 'happy'}, {'Position': [640, 10, 760, 160], 'Confidence': 0.88, 'Label': 'happy'}, {'Position': [765, 9, 900, 144], 'Confidence': 0.9, 'Label': 'happy'}]

student_answer_bbox = extract_bbox(response)
print(student_answer_bbox)
ground_truth_bbox = extract_bbox(response)

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []

    for bbox in bbox_list:
        # print(bbox) {'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        # print(position_tuple) (86, 134, 283, 331)

        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)

    return unique_bboxes


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

    ### [(0.7192676547515258, 1.0), (0, 0.7)] best_iou,bbox2['Confidence']
    # print("134, iou_results: ", iou_results)
    return iou_results


def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1 - temp_iou) * (1 - temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward

    if len_gt >= len(iou_results):
        iou_reward = iou_reward / len_gt
    else:
        iou_reward = iou_reward / len(iou_results)
    return iou_reward
student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
print(student_answer_bbox)
iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
print(iou_results)
#  ### new iou reward
reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    # clip to [baseline, 0.95]
print(reward)

reward = max(0.05, min(reward, 0.95))
print(reward)




# [{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, ...]
#  cd /scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1
#  python grpo_qwen_2_5_dfew_reward_add_emotion_label_test_files.py
# qsub -I -P kf09 -q gpuvolta -l ngpus=1,ncpus=12,mem=64GB,walltime=00:50:00