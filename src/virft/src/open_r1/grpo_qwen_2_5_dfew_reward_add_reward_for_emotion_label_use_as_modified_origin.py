# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
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
#   change  [{'Position': [254, 303, 291, 365], 'Confidence': 0.9, 'Label': 'sad'}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8, 'Label': 'happy'}]

# input: <answer>[{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}]</answer>
# output: [{"Position": [86, 134, 283, 331], "Confidence": 0.95, "Label": "happy"}]
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

# 计算两个边界框（bounding boxes）之间的 IoU（Intersection over Union，交并比）。
# 值越接近 1，说明两个框越相似（重叠越多）
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


# input: ground_truth:   [{'Position': [85, 130, 280, 330], 'Confidence': 1, 'Label': 'happy'}]
#        student_answer: [{'Position': [86, 134, 283, 331], 'Confidence': 0.77, 'Label': 'happy'}]
# output: [(0.95, 0.77)] # 0.95: calculated from function calculate_iou(), 0.77: Confidence
def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    # 按Confidence从高到低对预测框 list2 进行排序，优先匹配高置信度的框。
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    label_results = []
    
    # 用来记录哪些真实框(groud truth label)已经匹配过，避免重复匹配。
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

        # 如果找到的最大 IoU 超过设定阈值（说明预测框和真实框匹配成功）：
        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
            label_results.append((bbox2['Label'], bbox1['Label'], bbox2['Confidence']))
        else:
            iou_results.append((0, bbox2['Confidence']))
            label_results.append((bbox2['Label'], 'null', bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)] best_iou,bbox2['Confidence']
    # print("134, iou_results: ", iou_results)
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

# V1 not used
def compute_reward_iou(iou_results):
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
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return iou_reward

# V2 len_gt: ground truth label length. e.g.[{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}] length is 1
# 根据 IoU 和Confidence，计算一个iou奖励分数.
# input: iou list from sort_and_calculate_iou, 真实框（ground truth）的数量。
# for example: compute_reward_iou_v2([(0.95, 0.77)], 1):
def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0 # 累加所有 IoU 奖励
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        #如果 IoU 为 0（即预测框没有匹配到任何真实框）：
        #说明这个预测是错误的
        #奖励 = (1 - IoU) * (1 - 置信度) → 如果你错得离谱还信心满满，那就惩罚大。
        #如果 IoU 大于 0（即有匹配）：
        #奖励就是置信度本身，越自信越好。
        temp_iou_reward = temp_iou 
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    #如果真实框数量比较多，用 len_gt 归一化。
    #否则，用预测框数量归一化。
   #这一步是为了避免预测框数或真实框数不平衡带来的影响。
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

#根据 IoU 和Confidence，计算一个Conficence奖励分数
def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        #如果 IoU 为 0（没有命中真实框）：
        #表示预测错误。
        #奖励 = (1 - IoU) * (1 - 置信度)
        #→ 如果预测错了还非常自信（置信度高），那么这部分奖励会变得很低（惩罚）。    
        #如果 IoU > 0（命中了）：
        #奖励 = 置信度本身。
        #越自信越好。
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
    
    #对累加的奖励进行归一化处理（平均值）：
    #避免预测框多时奖励值过大
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward


def compute_reward_label(label_results, len_gt):
    label_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(label_results)):
        temp_label = label_results[i][0]
        temp_gt_label = label_results[i][1]
        temp_confidence = label_results[i][2]
        temp_label_reward = 0
        if temp_label == temp_gt_label: #奖励就是置信度本身，越自信越好。
            temp_label_reward = 0.95
            temp_confidence_reward = temp_confidence
        else: #说明这个预测是错误的.  如果你错得离谱还信心满满，那就惩罚大。
            temp_label_reward = 0.05
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        
        label_reward += temp_label_reward
        confidence_reward += temp_confidence_reward

    #如果真实框数量比较多，用 len_gt 归一化。
    #否则，用预测框数量归一化。
    #这一步是为了避免预测框数或真实框数不平衡带来的影响。
    if len_gt>=len(label_reward):
        label_reward = label_reward/len_gt
    else:
        label_reward = label_reward/len(label_results)
    return label_reward



#### 1. need to change to stable the iou reward
# one example:
# content: <think>
#Analyze the facial features of the man:
#- He has a moderate-sized head with closely cropped hair.
#- He's wearing earrings, which may affect the perception of his facial features.
#- His expression looks serious or contemplative, as his mouth is closed and eyes slightly narrowed.
#- He's possibly sitting, facing slightly to the left, with his shoulders slightly hunched.
#Based on these observations, it's likely that the 'emotion' could be 'neutral', but the specific emotion might lean more towards 'sad' due to his serious expression.
#</think> 
#<answer> [{' Position': [127, 2, 358, 241], 'Confidence': 0.95, 'Label': 'neutral'}]</answer>
#sol: <answer>[{'Position': [142, 43, 271, 219], 'Confidence': 1, 'Label': 'neutral'}]</answer>
#completions: [[{'role': 'assistant', 'content': "<think> The person in the image is wearing a dark jacket, and his hair seems to be short and dark. However, specific facial details such as eye color, expression, and posture are not clearly visible. Therefore, it is difficult to accurately determine the emotion of the person. Based on the limited visible clues, the most likely emotion could be neutral as there are no strong indicators of fear, anger, happiness, sadness, disgust, or surprise. </think>\n<answer>[{'Position': [128, 0, 504, 364], 'Confidence': 0.20, 'Label': 'neutral'}]</answer>"}], [{'role': 'assistant', 'content': "<think> The image shows a man with black hair and a beard, wearing a dark jacket and a white shirt. He has an attentive and serious expression, suggesting he is engaged in a serious conversation or making an important statement. No objects such as a TV are visible in the image. </think> \n<answer>[{'Position': [66, 2, 501, 364], 'Confidence': 0.98, 'Label': 'neutral'}]</answer>"}], [{'role': 'assistant', 'content': "<think>\nAnalyze the facial features of the man:\n- He has a moderate-sized head with closely cropped hair.\n- He's wearing earrings, which may affect the perception of his facial features.\n- His expression looks serious or contemplative, as his mouth is closed and eyes slightly narrowed.\n- He's possibly sitting, facing slightly to the left, with his shoulders slightly hunched.\nBased on these observations, it's likely that the 'emotion' could be 'neutral', but the specific emotion might lean more towards 'sad' due to his serious expression.\n</think> \n<answer> [{' Position': [127, 2, 358, 241], 'Confidence': 0.95, 'Label': 'neutral'}]</answer>"}], [{'role': 'assistant', 'content': "<think> I will analyze the facial expression, posture, and context of the person in the image. The person appears to have a serious and contemplative expression, which might indicate a mix of skepticism or uncertainty.\n\nThe text in the image is likely providing context or dialogue related to the character's emotions.\n\nGiven the analysis, the emotion appears to be 'sad'.\n</think> \n<answer>[{'position': [137, 333, 416, 373], 'confidence': 0.95, 'label': 'sad'}]</answer>"}]]
#student_answer_bbox: [{' Position': [127, 2, 358, 241], 'Confidence': 0.95, 'Label': 'neutral'}]
#ground_truth_bbox: [{'Position': [142, 43, 271, 219], 'Confidence': 1, 'Label': 'neutral'}]
#iou_results: []
def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    #从completions里拿出所有回答内容（content字段），组成一个列表contents。
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
            #如果parse, verify验证结果大于0，说明答案正确，奖励设为0.95。
            if float(verify(answer, parse(sol))) > 0:
                reward = 0.95 #1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
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
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
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
        #                 f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards

def accuracy_reward_confidence(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
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
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')
                student_answer = student_answer.replace("]]",']')
                student_answer = student_answer.replace("\n",'')
                
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    reward = compute_reward_confidence(iou_results)
                    reward = max(0.05, min(reward, 0.95))
                    # if reward>1:
                    #     reward = 1.0
                    # if reward<0:
                    #     reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # 
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a") as f:
        #         f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
        #         f.write(f"content: {content}\n")
        #         f.write(f"sol: {sol}\n")
        #         if show_flage==1:
        #             f.write(f"student_answer_bbox: {student_answer_bbox}\n")
        #             f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
        #             if student_answer_bbox!=None:
        #                 f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # baseline = 0.1
    # max_gain = 1.0 - baseline
    #这是一个正则表达式，表示：
    #字符串应该以 <think> 开头，并以 </think> 结尾。
    #然后可以有任意空格或换行（\s*）。
    #接着出现 <answer>...</answer>。
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
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
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
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')
                student_answer = student_answer.replace("]]",']')
                student_answer = student_answer.replace("\n",'')
                
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    reward = compute_reward_label(label_results, len(ground_truth_bbox))
                    reward2 = reward
                    reward = max(0.05, min(reward, 0.95))
                    # if reward>1:
                    #     reward = 1.0
                    # if reward<0:
                    #     reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # 
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
                # f.write(f"content: {content}\n")
                f.write(f"completions: {completions}\n")
                # f.write(f"sol: {sol}\n")
                f.write(f"solution: {solution}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox_label: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox_label: {ground_truth_bbox}\n")
                    f.write(f"student_answer: {student_answer}\n")
                    f.write(f"ground_truth: {ground_truth}\n")
                    f.write(f"-------------------------- end --------------------------\n")

                #     if student_answer_bbox!=None:
                #         f.write(f"iou_results: {iou_results}\n")
                #         f.write(f"label_results: {label_results}\n")
        show_flage = 0 
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
    script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format', 'accuracy_label']
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