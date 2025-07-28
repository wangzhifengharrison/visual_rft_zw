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
import torch                            # ← make sure this is here!

from bert_score import BERTScorer

# --- initialize one scorer per process, on CPU or its assigned GPU ---
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    device = f"cuda:{local_rank}"
else:
    device = "cpu"

scorer = BERTScorer(
    model_type="distilbert-base-uncased",
    lang="en",
    rescale_with_baseline=True,
    device=device
)


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

def sort_and_calculate_iou(list1, list2, student_think='', iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    label_results = []
    reasoning_attribute_results = []
    
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

            if student_think:
                reasoning_attribute_match_results = reasoninng_attribute_match_result_in_think(bbox2['Label'], student_think)
                reasoning_attribute_results.append((reasoning_attribute_match_results, bbox2['Confidence']))

        else:
            iou_results.append((0, bbox2['Confidence']))
            label_results.append((False, bbox2['Confidence']))
            if student_think:
                reasoning_attribute_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)] best_iou,bbox2['Confidence']
    # print("134, iou_results: ", iou_results)
    if student_think:
        return iou_results, label_results, reasoning_attribute_results

    return iou_results, label_results

def compute_bertscore(candidates, references,
                      batch_size=64,
                      verbose=True):
    print('--------162----------')
    """
    Compute BERTScore precision, recall, and F1 for two lists of sentences.

    Args:
        candidates (List[str]): generated sentences.
        references (List[str]): ground-truth sentences.
        model_type (str): HuggingFace model name (e.g. 'roberta-large').
        lang (str): language code for tokenizer (e.g. 'en', 'zh').
        batch_size (int): how many sentences per batch.
        verbose (bool): show progress bar.

    Returns:
        P, R, F1 (torch.Tensor): scores per sentence.
    """
    P, R, F1 = scorer.score(
            cands=candidates,
            refs=references,
            batch_size=batch_size,
            verbose=verbose
        )
    return P, R, F1


def reasoninng_attribute_match_result_in_think(ground_truth_label, student_think = ''):
    in_context = 0
    in_body_language = 0
    in_emotion = 0

    if not student_think:
        return 0

    predefined_emotion_attributes = get_predefined_emotion_attributes(ground_truth_label)
    # using bert score
    emotion_predictions = [student_think.lower()] * len(predefined_emotion_attributes)
    emotion_references  = [attr.lower() for attr in predefined_emotion_attributes]

    e_P, e_R, e_F1 = compute_bertscore(
        candidates=emotion_predictions,
        references=emotion_references,
        batch_size=64,
        verbose=False
    )
    # average f1 across attributes
    e_avg_f1 = e_F1.mean().item()
    print('Avg F1:', e_avg_f1)
    in_emotion = e_avg_f1

    predefined_context_attributes = get_predefined_context_attributes(ground_truth_label)
    context_predictions = [student_think.lower()] * len(predefined_context_attributes)
    context_references  = [attr.lower() for attr in predefined_context_attributes]
    c_P, c_R, c_F1 = compute_bertscore(
        candidates=context_predictions,
        references=context_references,
        batch_size=64,
        verbose=False
    )
    # average f1 across attributes
    c_avg_f1 = c_F1.mean().item()
    print('Avg F1:', c_avg_f1)
    in_context = c_avg_f1

    predefined_body_language_attributes = get_predefined_body_language_attributes(ground_truth_label)
    body_language_predictions = [student_think.lower()] * len(predefined_body_language_attributes)
    body_language_references  = [attr.lower() for attr in predefined_body_language_attributes]
    b_P, b_R, b_F1 = compute_bertscore(
        candidates=body_language_predictions,
        references=body_language_references,
        batch_size=64,
        verbose=False
    )
    # average f1 across attributes
    b_avg_f1 = b_F1.mean().item()
    print('Avg F1:', b_avg_f1)
    in_body_language = b_avg_f1


    return 0.2 * in_context + 0.3 * in_body_language + 0.5 * in_emotion


def get_predefined_context_attributes(ground_truth_label):
    contexts = {
        'happy': ['cozy indoor environment', 'cheerful tone', 'outdoors', 'positive emotional state', 'indoor setting', 'positive interaction', 'warm lighting', 'outdoor setting', 'warm and cozy indoor environment', 'formal attire', 'unexpected information', 'dim lighting', 'mysterious atmosphere', 'indoor environment', 'serious conversation', 'cozy restaurant', 'cozy indoor setting', 'traditional Chinese attire', 'friendly interaction', 'natural environment', 'minimalist indoor setting', 'private space', 'enjoyable conversation', 'engaged in conversation', 'traditional attire', 'fast-paced speech', 'unexpected good news', 'dimly lit car interior', 'pleasant conversation', 'receiving good news', 'dimly lit indoor setting', 'dimly lit environment', 'laughter in the voice', 'unexpected event', 'expression of gratitude', 'friendly conversation', 'warmly decorated indoor setting', 'tranquil outdoor scene', 'blurred background', 'office setting', 'engaged in a conversation', 'good news', 'softly lit indoor setting', 'warm and cozy indoor setting', 'dimly lit room', 'expressing gratitude', 'dimly lit indoor environment', 'outdoor environment', 'soft lighting', 'natural light', 'military uniform', 'outdoor scene', 'joyful atmosphere', 'pleasant moment', 'positive feedback'],
        'fear': ['dimly lit indoor scene', 'unexpected event', 'tense atmosphere', 'unknown challenge', 'dimly lit environment', 'intense emotional response', 'traditional attire', 'inner struggle', 'indoor setting', 'dim lighting', 'dimly lit', 'experiencing intense emotional turmoil', 'intense emotional distress', 'industrial environment', 'dimly lit room', 'sense of urgency and crisis', 'struggle for survival', 'atmosphere of mystery and tension', 'potential danger', 'indoor environment', 'intense emotional turmoil', 'extreme tension', 'anger', 'intense inner turmoil', 'imminent threat', 'sense of urgency', 'anxiety', 'dilapidated room', 'sense of urgency and tension', 'rugged outdoor environment', 'imminent danger', 'faint light source', 'intense emotional experience', 'military uniform', 'oppressive atmosphere', 'extreme fear and distress', 'sudden fright', 'immediate threat', 'high-pitched tone', 'urgency', 'emotional turmoil', 'extreme fear', 'rapid breathing', 'mysterious atmosphere', 'dimly lit indoor setting', 'intense emotional upheaval', 'emotional upheaval', 'potential threats', 'psychological struggle', 'desert setting', 'mysterious and gloomy environment', 'dimly lit setting', 'closed door', 'minimalist bathroom setting'],
        'sad': ['indoor setting', 'intense emotional turmoil', 'dissatisfaction', 'anger', 'sense of loss', 'indoor environment', 'dimly lit room', 'emotional distress', 'dimly lit indoor setting', 'minimalist indoor setting', 'loss', 'helplessness', 'emotional challenge', 'inner struggle', 'outdoor setting', 'intense emotional pain', 'outdoor scene', 'blurred background', 'close-up scene', 'emotional fluctuations', 'deep sadness', 'profound sadness', 'despair', 'grief', 'warm lighting', 'dimly lit indoor environment', 'simple background', 'tranquil atmosphere', 'loss of something important', 'soft lighting', 'mysterious atmosphere', 'sadness', 'anxiety', 'psychological pressure', 'traditional Chinese attire', 'dim lighting', 'emotional turmoil', 'extreme sadness', 'tense atmosphere', 'unexpected news', 'tension', 'tranquil yet oppressive atmosphere', 'outdoor natural environment', 'tranquil scene', 'dimly lit environment', 'traditional attire', 'intense emotional distress', 'remorse', 'experiencing intense emotional turmoil', 'warm and cozy indoor environment', 'love'],
        'surprise': ['unexpected news', 'sense of tension and unease', 'unexpected event', 'dimly lit room', 'tension', 'anxiety', 'emotional upheaval', 'minimalist indoor setting', 'indoor setting', 'dissatisfaction', 'internal conflict', 'anger', 'dilapidated environment', 'emotional challenge', 'sudden event', 'emotional fluctuations', 'tranquil natural setting', 'unexpected situation', 'indoor environment', 'warm lighting', 'formal attire', 'confusion', 'unexpected information or events', 'intense emotional fluctuations', 'dimly lit environment', 'warm and cozy indoor environment', 'tranquil yet mysterious atmosphere', 'outdoor setting', 'intense emotional turmoil', 'mysterious atmosphere', 'unexpected information', 'dimly lit scene', 'unknown situation', 'dimly lit indoor setting', 'sense of urgency', 'dimly lit car interior', 'tense atmosphere', 'urgency', 'traditional attire', 'inner turmoil', 'intense emotional shift', 'office setting', 'blurred background', 'outdoor natural environment', 'atmosphere of mystery and tension', 'unexpected information or event', 'interaction with another person', 'dim lighting', 'natural light', 'high-pitched tone', 'indistinct background', 'unexpected situations', 'dark clothing', 'outdoor environment', 'dim background', 'emotional journey'],
        'angry': ['dissatisfaction', 'strong emotions', 'indoor setting', 'dimly lit indoor setting', 'urgent tone', 'urgency', 'anger', 'emotional upheaval', 'tense atmosphere', 'tense conversation', 'fast-paced speech', 'dimly lit environment', 'argumentative situation', 'heated argument', 'accusation', 'emotional turmoil', 'indoor environment', 'loud volume', 'engaged in a heated argument', 'sense of urgency', 'dimly lit room', 'high-pitched tone', 'formal attire', 'emotional agitation', 'anxiety', 'minimalist indoor setting', 'dim lighting', 'serious conversation', 'traditional Chinese attire', 'intense emotional turmoil', 'intense conversation', 'intense emotional state', 'accusatory tone', 'intense emotional agitation', 'aggressive tone', 'warm lighting', 'tension', 'mysterious atmosphere', 'intense emotional fluctuations', 'intense anger', 'intense emotions', 'outdoor environment', 'intense argument', 'outdoor setting', 'expressing dissatisfaction', 'private space', 'expressing anger and dissatisfaction', 'frustration', 'tense conversation or argument', 'fast speech', 'confrontation', 'internal struggle', 'high volume', 'military uniform'],
        'disgust': ['calm interaction', "lack of visibility of the other person's reaction suggests possible displeasure or annoyance", 'responding to another person or thought', 'may indicate resignation or disappointment', 'experienced setback or disappointment', 'expressing inner struggle and emotional depth', 'expressing unpleasant thoughts or worries', 'air of helplessness and despair in her voice', 'indoor restaurant or café environment', 'relaxed social atmosphere', "complaining about another person's behavior", 'expressing dissatisfaction with the driving situation', 'accusing others of bad driving behavior', 'private space adorned with elegant decorations', 'soft lighting', 'commitment to personal style', "dedication to life's purpose", 'complaint about substandard housing', 'accusation regarding high-rise apartments', 'traditional Chinese environment', 'playing a wooden drum', 'urgent need for expression', "reaction to another person's comment", 'feeling of dissatisfaction or anger'],
        'neutral': ['indoor setting', 'anxiety', 'dimly lit room', 'important conversation', 'sense of urgency', 'sense of tension and unease', 'engaged in conversation', 'dim lighting', 'tranquil yet slightly oppressive atmosphere', 'sense of tension or anxiety', 'indoor environment', 'indoor scene', 'dimly lit indoor setting', 'dimly lit scene', 'traditional Chinese attire', 'indoors', 'minimalist indoor setting', 'engaged in a conversation', 'traditional attire', 'mysterious atmosphere', 'serious conversation', 'intense emotional fluctuations', 'engaged in a heated argument', 'dissatisfaction', 'unexpected information', 'formal attire', 'dimly lit environment', 'outdoor setting', 'fast-paced speech', 'soft lighting', 'heated argument', 'private space', 'warm lighting', 'dimly lit setting', 'internal struggle', 'tense atmosphere', 'military uniform', 'office setting', 'facing challenges', 'elderly man', 'intense emotional turmoil', 'tranquil atmosphere', 'tense conversation', 'emotional upheaval', 'deep contemplation', 'minimalist indoor environment']
    }
    return contexts.get(ground_truth_label.lower(), [])

def get_predefined_emotion_attributes(ground_truth_label):
    emotion = {
        'happy': ['warm smile', 'helplessness', 'joy', 'happy', 'relaxed', 'slight smile', 'wide eyes', 'wide open eyes', 'slightly open mouth', 'smile', 'contentment', 'smiling', 'eyes slightly closed', 'laughing', 'bright eyes', 'lips slightly parted', 'friendly expression', 'eyes closed', 'laughter', 'open eyes', 'relaxed facial muscles', 'joyful', 'calm expression', 'anger', 'excitement', 'radiant smile', 'more pronounced smile', 'smile becomes more pronounced', 'gentle smile', 'happy eyes', 'smiling warmly', 'relaxed expression', 'happy facial expression', 'serious expression', 'laughing heartily', 'joyful smile', 'furrowed brows', 'tightly pressed lips', 'radiating joy', 'joyful expression', 'shining eyes', 'anxiety', 'slightly widened eyes', 'slight frown', 'open mouth', 'furrowed brow', 'open mouth showing teeth', 'direct gaze', 'rapid blinking', 'wide open mouth', 'happiness', 'focused gaze', 'smiles', 'tightly closed lips', 'slightly upturned lips', 'calm', 'widened eyes', 'gently closed eyes', 'happy expression', 'mouth slightly upturned', 'slightly raised eyebrows', 'wide-open eyes', 'genuine joy', 'slightly squinted eyes', 'upturned lips', 'direct eye contact', 'relaxed gaze', 'broad smile', 'focused expression', 'mouth slightly open', 'raised eyebrows', 'serious', 'confusion', 'slightly closed eyes', 'downturned mouth corners', 'slightly furrowed brows', 'wide smile', 'joyful eyes', 'upturned mouth', 'eyes slightly squinted', 'frequent blinking', 'unease', 'eyes gently closed', 'slight upward curve at the corners of her mouth', 'closed lips', 'wider smile', 'slight upward curve at the corners of his mouth', 'gentle expression', 'curiosity', 'relaxed facial expression', 'eyebrows raised', 'bright smile', 'delight', 'pronounced smile', 'squinted eyes', 'extreme joy', 'subtle changes in eyes', 'gentle gaze', 'slightly parted lips', 'focused', 'slightly agape mouth', 'relaxed eyes', 'relaxed demeanor', 'closed eyes', 'eyes open', 'exaggerated facial expression', 'relaxation', 'narrowed eyes'],
        'fear': ['furrowed brows', 'wide-open eyes', 'tightly pressed lips', 'extreme fear', 'wide eyes', 'slightly open mouth', 'raised eyebrows', 'wide open eyes', 'slightly agape mouth', 'fear', 'pain', 'determination', 'anger', 'urgency', 'wide open mouth', 'tense facial muscles', 'tension', 'confusion', 'anxiety', 'slightly parted lips', 'serious expression', 'unease', 'calm', 'fatigue', 'distress', 'downturned mouth', 'helplessness', 'tense expression', 'furrowed eyebrows', 'despair', 'expression of fear', 'terror', 'downturned mouth corners', 'agape mouth', 'mouth slightly open', 'open mouth', 'intense anxiety', 'agony', 'brows furrowed', 'expression of tension', 'extreme pain', 'tense', 'mouth wide open', 'expression of extreme fear', 'mouth agape', 'worry', 'widened eyes', 'shock', 'downturned eyes', 'calm expression', 'furrowed brow', 'tightly closed lips', 'showing teeth', 'slightly agape lips'],
        'sad': ['furrowed brows', 'slightly open mouth', 'furrowed brow', 'downturned corners of mouth', 'downturned corners of the mouth', 'confusion', 'sadness', 'unease', 'downturned mouth', 'sad', 'slightly furrowed brows', 'eyes closed', 'despair', 'fatigue', 'helplessness', 'anger', 'tears in her eyes', 'raised eyebrows', 'slight smile', 'downturned facial muscles', 'slightly parted lips', 'negative expression', 'negative facial expression', 'tears welling up', 'deep sadness', 'wide eyes', 'contemplation', 'downturned mouth corners', 'furrowed eyebrows', 'downturned eyes', 'tight lips', 'sad expression', 'tightly pressed lips', 'moist eyes', 'wide open eyes', 'agape mouth', 'parted lips', 'tears streaming down her face', 'determination', 'smile', 'serious expression', 'slight frown', 'pain', 'closed eyes', 'wide-open eyes', 'open mouth', 'tears', 'frustration', 'sad facial expression', 'solemn expression', 'signs of fatigue', 'tension', 'contemplative expression', 'calm', 'serious', 'slightly widened eyes', 'slightly downturned mouth', 'downward turn of the mouth', 'anxiety', 'downturned lips', 'slightly open eyes', 'extreme sadness', 'tears streaming down cheeks', 'sad eyes', 'focused', 'slightly furrowed brow', 'lips slightly parted', 'eyes slightly closed', 'wide open mouth', 'sorrow', 'slightly agape mouth', 'tense expression', 'fixed gaze', 'slightly closed eyes', 'tightly closed lips', 'blurred face', 'slightly moist eyes', 'concern', 'tears glistening', 'downward-turned mouth', 'crying', 'calm expression', 'tense facial muscles', 'widened eyes', 'tears streaming down his face', 'pressed lips', 'mouth slightly open', 'extreme pain', 'wrinkles', 'downward gaze', 'focused gaze', 'melancholic expression'],
        'surprise': ['slightly furrowed brows', 'serious expression', 'wide open eyes', 'surprise', 'confusion', 'slightly agape mouth', 'furrowed brows', 'open mouth', 'wide eyes', 'slightly open mouth', 'focused', 'serious', 'widened eyes', 'slightly parted lips', 'tension', 'focused expression', 'focused gaze', 'wide-open eyes', 'anxiety', 'anger', 'downturned mouth corners', 'surprised expression', 'mouth agape', 'eyes widen', 'raised eyebrows', 'mouth slightly open', 'surprised', 'calm', 'tense expression', 'tightly closed lips', 'tightly pressed lips', 'open mouth showing teeth', 'furrowed brow', 'unease', 'concern', 'expression of surprise', 'slight frown', 'wide open mouth', 'calm expression', 'tense facial expression', 'downturned mouth', 'pressed lips', 'agape mouth', 'tense facial muscles', 'excitement', 'complex expression', 'astonishment', 'curiosity', 'surprised eyes', 'intense gaze', 'extreme surprise', 'slightly widened eyes', 'deep eyes', 'shock', 'alertness', 'determination', 'furrowed eyebrows', 'eyes wide open'],
        'angry': ['wide eyes', 'slightly open mouth', 'raised eyebrows', 'slightly parted lips', 'serious look', 'angry look', 'anger', 'wide open eyes', 'furrowed brows', 'open mouth', 'slightly agape mouth', 'furrowed brow', 'intense gaze', 'open mouth showing teeth', 'wide-open eyes', 'serious expression', 'downturned corners of the mouth', 'widened eyes', 'tense facial muscles', 'expression of anger', 'anxiety', 'confusion', 'tightly pressed lips', 'tension', 'wide open mouth', 'dissatisfaction', 'tense expression', 'direct gaze', 'mouth slightly open', 'eyebrows raised', 'focused expression', 'angry facial expression', 'mouth moving', 'discontent', 'intense anger', 'downturned mouth', 'furrowed eyebrows', 'tightened facial muscles', 'tense', 'disbelief', 'negative facial expression', 'extreme pain', 'downturned mouth corners', 'serious facial expression', 'tightly closed lips', 'determination', 'downturned eyes', 'intense expression', 'sharp eyes', 'tense facial expression', 'slight frown', 'serious', 'focused', 'focused gaze', 'calm', 'agape mouth', 'mouth wide open', 'exaggerated facial expression', 'angry', 'angry expression', 'eyes widen', 'mouth opens wide', 'frowning', 'fierce expression', 'clenched teeth', 'revealing teeth', 'intense eyes', 'narrowed eyes', 'frustration', 'sharp gaze', 'calm expression', 'wide-open mouth', 'excitement', 'agitation', 'intense dissatisfaction', 'slightly furrowed brows', 'negative expression', 'extreme anger', 'glaring eyes', 'opened mouth', 'showing teeth', 'intense emotional turmoil', 'mouth open', 'excited', 'frequent blinking', 'mouth agape', 'helplessness', 'deep-seated anger'],
        'disgust': ['raised eyebrows', 'opened mouth', 'lowered position', 'slightly furrowed brows', 'downturned mouth corners', 'furrowed brows', 'slightly open eyes', 'tears in her eyes', 'eyes slightly closed', 'mouth moving as if speaking', 'facial expression somewhat displeased', 'look of disgust', 'open mouth showing teeth', 'concentration', 'determination', 'subtle changes in gaze', 'lip movements indicating inner effort and resolve', 'slightly open mouth', 'negative expression', 'anger', 'pain', 'tightly closed lips', 'downturned corners of the mouth', 'displeasure', 'discontent'],
        'neutral': ['serious expression', 'furrowed brows', 'wide-open eyes', 'tense', 'downturned eyes', 'confusion', 'tension', 'anxiety', 'focused expression', 'look of confusion', 'wrinkles', 'tightly pressed lips', 'wide open eyes', 'agape mouth', 'mouth slightly open', 'serious', 'focused', 'tightly closed lips', 'direct gaze', 'open mouth', 'calm expression', 'narrowed eyes', 'focused gaze', 'slightly open mouth', 'slight frown', 'moist eyes', 'widened eyes', 'open mouth showing teeth', 'wide eyes', 'downturned corners of the mouth', 'downturned mouth', 'serious facial expression', 'intense gaze', 'contemplative', 'furrowed brow', 'unease', 'slightly agape mouth', 'negative expression', 'closed lips', 'slightly furrowed brows', 'raised eyebrows', 'frowning', 'negative facial expression', 'fatigue', 'helplessness', 'slightly open eyes', 'furrowed eyebrows', 'focused eyes', 'calm', 'relaxed expression', 'tense expression', 'downturned mouth corners', 'deep eyes', 'tight lips', 'closed eyes', 'eyes closed', 'slightly parted lips', 'displeasure', 'sharp eyes', 'downturned lips', 'open eyes', 'puzzled', 'serious look', 'slightly furrowed brow', 'frustration', 'solemn expression', 'slightly downturned mouth', 'slightly closed eyes', 'slightly raised eyebrows', 'melancholic expression', 'downward gaze', 'signs of fatigue', 'determination', 'tense facial expression', 'concentration', 'anger', 'mouth moving as if speaking']
    }
    return emotion.get(ground_truth_label.lower(), [])

def get_predefined_body_language_attributes(ground_truth_label):
    body_language = {
        'happy': ['standing', 'looking down', 'sitting on a sofa', 'open posture', 'relaxed stance', 'hair neatly tied back', 'mouth moving as if speaking', 'relaxed posture', 'head moves slightly', 'sitting comfortably', 'engaged in conversation', 'standing still', 'standing confidently', 'sitting alone', 'tense posture', 'leaning slightly forward', 'relaxed demeanor', 'steady posture', 'engaged demeanor', 'upright posture', 'leaning forward', 'engaged posture', 'relaxation', 'engaging in conversation', 'confident posture', 'direct gaze', 'possibly leaning slightly forward', 'open body language', 'arms outstretched', 'animated gestures', 'ease', 'friendly demeanor', 'attentive posture', 'fast-paced speech', 'focused posture', 'smiling', 'engaged stance', 'expressive gestures', 'holding a drink', 'shifting gaze', 'standing alone', 'neatly tied back hair', 'sitting at a table', 'calm posture', 'lively speech', 'relaxed', 'comfortable posture', 'open mouth', 'enthusiastic gestures', 'laughing heartily', 'sitting posture'],
        'fear': ['rigid posture', 'leaning forward', 'tense posture', 'disheveled hair', 'alert posture', 'tense stance', 'sitting on the ground', 'arms outstretched', 'tensed posture', 'lying on the ground', 'furrowed brows', 'tense body', 'rapid breathing', 'alertness', 'holding a firearm', 'high alertness', 'attempting to escape', 'resisting', 'trying to escape', 'tension', 'holding a gun', 'high-pitched exclamations', 'helplessness', 'crouched position', 'anxiety', 'urgency'],
        'sad': ['leaning forward', 'slight head movement', 'hands covering face', 'furrowed brows', 'trembling voice', 'neatly tied back hair', 'tense posture', 'crying', 'possibly tense posture', 'back to the camera', 'rapid speech', 'embracing', 'helplessness', 'slumped shoulders', 'slightly open mouth', 'mouth moving as if speaking', 'standing still', 'sighing', 'sitting indoors', 'standing alone', 'head tilted forward', 'anxiety', 'hair pulled back', 'holding a gun', 'relaxed posture', 'sitting alone', 'tension in facial muscles', 'slumped posture', 'hesitant movements', 'trying to control emotions', 'lying on a bed', 'sobbing', 'tense expression', 'gaze downward', 'slow speech', 'helpless posture', 'looking downward', 'gaze shifting', 'kneeling', 'disheveled hair', 'focused posture', 'lack of movement', 'grief', 'holding an object', 'looking down', 'head bowed', 'hands resting on knees', 'despair', 'sitting on the ground', 'possibly slumped posture', 'lying on bed', 'sitting still', 'tensed posture', 'closing eyes', 'tension', 'close interaction', 'stillness', 'sitting on a sofa', 'hands clasped together', 'attempting to control emotions', 'lying on the ground', 'close embrace'],
        'surprise': ['tense posture', 'standing position', 'rapid speech', 'focused posture', 'disheveled hair', 'back to the camera', 'raised eyebrows', 'leaning forward', 'tension', 'focused', 'unease', 'standing still', 'alertness', 'gaze shifting', 'leaning slightly forward', 'tension in body', 'anxious posture', 'slumped posture', 'standing at the forefront', 'standing by the window', 'alert posture', 'steady posture', 'tensed posture', 'intense focus', 'still posture', 'furrowed brows', 'wide eyes', 'tense body posture', 'standing at the center', 'upright posture', 'confusion', 'shifting gaze', 'disheveled appearance', 'rapid breathing', 'looking around', 'standing', 'rigid stance', 'tense', 'tension in posture', 'fast-paced speech', 'slightly open mouth'],
        'angry': ['tense stance', 'tense posture', 'tension in posture', 'quick speech', 'possibly tense posture', 'engaged stance', 'upright posture', 'frequent mouth movements', 'urgent tone', 'rigid posture', 'intense posture', 'leaning forward', 'holding a gun', 'focused posture', 'fixed gaze', 'defensive posture', 'forceful gestures', 'exaggerated movements', 'animated gestures', 'aggressive stance', 'stiff posture', 'hands on hips', 'steady posture', 'fast-paced speech', 'focused gaze', 'tension', 'engaged in heated argument', 'rapid speech', 'aggressive posture', 'hands raised high', 'intense engagement', 'anxiety', 'fast speech', 'clenched fists', 'aggressive gestures', 'crossed arms', 'tensed posture', 'frequent mouth movement', 'furrowed brows', 'engaged in a heated argument', 'looking around', 'possibly leaning forward', 'head moving up and down', 'intense gestures', 'shifting gaze', 'arms outstretched', 'focused stance', 'shouting', 'emphasizing viewpoint', 'intense anger', 'tense muscles', 'pointing', 'agitated posture', 'tense body language', 'intense expression', 'urgency'],
        'disgust': ['engaged in conversation', 'standing with back to the camera', 'wearing only underwear', 'negative facial expression', 'tense posture', 'gesturing towards the driving environment', 'meticulously organizing clothes', 'hair neatly tied back', 'focused demeanor', 'increased speech rate', 'dramatic raise in volume'],
        'neutral': ['focused posture', 'standing still', 'gaze shifting', 'hands resting on knees', 'tense posture', 'attentively listening', 'active participation', 'standing alone', 'intense posture', 'direct gaze', 'attentive listening', 'relaxed posture', 'tension', 'stillness', 'sitting alone', 'leaning forward', 'leaning slightly forward', 'tensed posture', 'standing at the center', 'subtle head movements', 'attentive posture', 'concentration', 'gaze directed off-screen', 'focused stance', 'crossed arms', 'sitting on a sofa', 'fixed gaze', 'contemplative posture', 'engaged posture', 'sitting at a desk', 'sitting at a table', 'engaged in conversation', 'clenched fists', 'serious posture', 'possibly tense posture', 'slight head movement', 'steady posture', 'contemplating', 'slight head movements', 'looking down', 'unease', 'focused gaze', 'upright posture', 'standing', 'shifting gaze', 'mouth moving as if speaking', 'standing posture', 'engaged stance', 'looking around', 'head movements', 'sitting upright', 'standing rigidly', 'animated gestures', 'intense concentration', 'disheveled hair', 'open posture', 'possibly leaning forward', 'sitting posture', 'looking downward', 'leaning forward slightly', 'focused demeanor'],
    }
    return body_language.get(ground_truth_label.lower(), [])

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

# V1
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

# V2
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

def compute_reward_reasoning_attributes(reasoning_attribute_results, len_gt):
    reasoning_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(reasoning_attribute_results)):
        temp_reasoning = reasoning_attribute_results[i][0]
        temp_confidence = reasoning_attribute_results[i][1]

        temp_reasoning_reward = 0.95 if temp_reasoning else 0.05
        reasoning_reward += temp_reasoning_reward
    
    if len_gt>=len(reasoning_attribute_results):
        reasoning_reward = reasoning_reward/len_gt
    else:
        reasoning_reward = reasoning_reward/len(reasoning_attribute_results)
    return reasoning_reward

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


#### 1. need to change to stable the iou reward
def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"***********\n")
                f.write(f"solution: {solution}\n")
                f.write(f"***********\n")


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
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                f.write(f"completions: {completions}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
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
                    reward = compute_reward_confidence(iou_results, label_results)
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
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
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
    print("accuracy_reward_label")
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
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
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
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                f.write(f"completions: {completions}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"label_results: {label_results}\n")
        show_flage = 0 
    return rewards

def accuracy_reward_reasoning_attributes(completions, solution, **kwargs):
    print("accuracy_reward_reasoning")

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
        reasoning_attribute_results = []

        # If symbolic verification failed, try string matching
        if reward == 0.05:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content) 
                think_match = re.search(r'<think>(.*?)</think>', content)
                # content: <think> The image shows a man dressed in a suit, standing in what appears to be a formal or somber setting. Given the context and his posture, it seems likely he is feeling sad or contemplative. The man has a furrowed brow and downcast eyes, which are common signs of sadness. </think>
                # <answer>[{'Position': [190, 12, 455, 360], 'Confidence': 0.89, 'Label': 'sad'}]</answer>
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')  
                student_answer = student_answer.replace("]]",']')  
                student_answer = student_answer.replace("\n",'')  

                student_think = think_match.group(1).strip() if think_match else content.strip()
                student_think = '<think>'+student_think+'</think>'
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9, 'Label': 'sad'}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8, 'Label': 'happy'}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results, reasoning_attribute_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox, student_think)
                    reward = compute_reward_reasoning_attributes(reasoning_attribute_results, len(ground_truth_bbox))
                    # clip to [baseline, 0.95]
                    reward = max(0.05, min(reward, 0.95))
                    # if reward>1:
                    #     reward = 1.0
                    if os.getenv("DEBUG_MODE") == "true":
                        log_path = os.getenv("LOG_PATH")
                        # local_rank = int(os.getenv("LOCAL_RANK", 0))
                        with open(log_path, "a") as f:
                            f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                            if show_flage==1:
                                f.write(f"student_think: {student_think}\n")
                                f.write(f"reasoning_attribute_results: {reasoning_attribute_results}\n")
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        reward += 1e-4  # change the reward value 
        rewards.append(reward)
        
        show_flage = 0 
    return rewards

###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
    "accuracy_label": accuracy_reward_label,
    "accuracy_reasoning_attributes": accuracy_reward_reasoning_attributes
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format','accuracy_label','accuracy_reasoning_attributes']
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