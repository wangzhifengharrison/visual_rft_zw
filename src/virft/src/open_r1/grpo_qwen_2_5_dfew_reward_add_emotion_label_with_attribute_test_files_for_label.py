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

from evaluate import load
bertscore = load("bertscore")

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


def sort_and_calculate_iou(list1, list2, student_think, iou_threshold=0.09):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    label_results = []
    reasoning_attribute_results = []
    
    matched_list1_indices = set()

    print('119', list2_sorted, list1, student_think)

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

        # print(130, best_iou, bbox2, bbox1)
        if best_iou > iou_threshold:
            # print('132-------------', bbox2['Label'], matched_bbox1, list1[matched_bbox1]['Label'], bbox2['Label'] == list1[matched_bbox1]['Label'])
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
            # print('135')

            # 检查标签是否一致
            label_match = bbox2['Label'] == list1[matched_bbox1]['Label']
            # print('136', label_match, bbox2['Label'], list1[matched_bbox1]['Label'])
            # print('------------------', label_match)
            label_results.append((label_match, bbox2['Confidence']))

            # 检查标签属性是否存在在学生的回答中及打分
            label_attribute_match_result = label_attribute_match_result_in_think(list1[matched_bbox1]['Label'], student_think)
            reasoning_attribute_results.append((label_attribute_match_result, bbox2['Confidence']))

            print('150', reasoning_attribute_results)

        else:
            iou_results.append((0, bbox2['Confidence']))
            label_results.append((False, bbox2['Confidence']))
            reasoning_attribute_results.append((0, bbox2['Confidence']))
            print('153')


    # print(label_results, '146')
    ### [(0.7192676547515258, 1.0), (0, 0.7)] best_iou,bbox2['Confidence']
    # print("134, iou_results: ", iou_results)
    return iou_results, label_results, reasoning_attribute_results

def label_attribute_match_result_in_think(ground_truth_label, student_think):
    in_context = 0
    in_body_language = 0
    in_emotion = 0

    predefined_context_attributes = get_predefined_context_attributes(ground_truth_label)
    predefined_body_language_attributes = get_predefined_body_language_attributes(ground_truth_label)
    predefined_emotion_attributes = get_predefined_emotion_attributes(ground_truth_label)
    # count how many words in student_think appeared in predefined_context_attributes
    # for attribute in predefined_context_attributes:
        # if attribute.lower() in student_think.lower():
        #     in_context += 1
    
    # predictions = [student_think.lower()]
    # references = predefined_emotion_attributes
    # results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    print('179 ---------------', predefined_emotion_attributes)

    predictions = [student_think.lower()] * len(predefined_emotion_attributes)
    references = predefined_emotion_attributes

    results = bertscore.compute(
        predictions= predictions,
        references=references,
        model_type="distilbert-base-uncased"
    )
    print(results, '184---------')

    # Get the most related reference
    best_index = results["f1"].index(max(results["f1"]))
    best_match = references[best_index]
    best_score = results["f1"][best_index]

    print(f"\n -------------- Best Match: {best_match}")
    print(f" ---------------  BERTScore F1: {best_score:.4f}")



    # for attribute_2 in predefined_body_language_attributes:
    #     if attribute_2.lower() in student_think.lower():
    #         in_body_language += 1
    # for attribute_3 in predefined_emotion_attributes:
    #     if attribute_3.lower() in student_think.lower():
    #         in_emotion += 1
    
    # print('150', in_context, in_body_language, in_emotion)
    # return 0.25*in_context + 0.25*in_body_language + 0.5*in_emotion
            


def get_predefined_context_attributes(ground_truth_label):
    contexts = {
        'happy': ['birthday', 'wedding', 'vacation', 'reunion', 'sunshine', 'fireworks', 'gift', 'hug', 'laughter',
                    'festival', 'success', 'love', 'dancing', 'achievement', 'picnic', 'icecream', 'spring', 'weekend',
                    'sunset', 'celebration', 'surprise', 'joke', 'holiday', 'song', 'music', 'freedom', 'blessing',
                    'family', 'friends', 'coffee', 'nature', 'blossom', 'balloon', 'ocean', 'rainbow', 'kindness',
                    'painting', 'children', 'smile', 'games', 'camping', 'flowers', 'puppies', 'snack', 'story', 'concert',
                    'breeze', 'mountain', 'joyride', 'hammock', 'chocolate'
                ],
        'disgust': ['vomit', 'maggot', 'rot', 'stench', 'filth', 'decay', 'pus', 'slime', 'mold', 'dirt', 'grime',
                    'cockroach', 'sewage', 'garbage', 'feces', 'urine', 'snot', 'spit', 'phlegm', 'infestation',
                    'wound', 'infection', 'bloody', 'grease', 'oil', 'sticky', 'ooze', 'clog', 'worm', 'squish',
                    'contamination', 'plague', 'smear', 'pollution', 'toxic', 'fungus', 'rancid', 'gross', 'repulsive',
                    'unclean', 'stenchy', 'crust', 'scab', 'bacteria', 'germs', 'disease', 'dirty', 'spoiled', 'putrid'
                ],
        'sad': ['funeral', 'tears', 'grief', 'loss', 'pain', 'farewell', 'goodbye', 'sorrow', 'crying', 'alone',
                    'broken', 'rejection', 'failure', 'death', 'darkness', 'storm', 'separation', 'loneliness', 'regret',
                    'mourning', 'rain', 'cold', 'night', 'isolation', 'abandonment', 'gloom', 'wound', 'burial',
                    'melancholy', 'tragedy', 'sadness', 'hurt', 'neglect', 'parting', 'distress', 'heartbreak', 'bleak',
                    'empty', 'misery', 'hopeless', 'despair', 'weeping', 'silence', 'graveside', 'gray', 'suffering',
                    'nostalgia', 'resentment', 'desolate'
                ],
        'angry': ['fight', 'shout', 'yell', 'rage', 'fury', 'conflict', 'punch', 'slam', 'scream', 'argument',
                    'war', 'explosion', 'resentment', 'revenge', 'grudge', 'betrayal', 'aggression', 'violence',
                    'fire', 'storm', 'temper', 'hostility', 'riot', 'threat', 'glare', 'scowl', 'clash', 'protest',
                    'outburst', 'quarrel', 'frustration', 'insult', 'jealousy', 'hate', 'blame', 'attack', 'retaliation',
                    'uprising', 'brawl', 'rageful', 'tension', 'provocation', 'tantrum', 'strike', 'resent', 'snarl',
                    'intimidation', 'shoving', 'vengeance', 'dispute'
                ],
        'fear': ['darkness', 'scream', 'terror', 'panic', 'danger', 'ghost', 'nightmare', 'blood', 'chase', 'abduction',
                    'threat', 'monster', 'creak', 'isolation', 'trap', 'ambush', 'dungeon', 'shock', 'dread', 'haunting',
                    'crash', 'killer', 'evil', 'snake', 'spider', 'storm', 'horror', 'fear', 'howl', 'tremble',
                    'anxiety', 'alarm', 'wound', 'war', 'death', 'knife', 'explosion', 'jump', 'cry', 'unknown',
                    'lost', 'abuse', 'collapse', 'fall', 'cliff', 'hostage', 'invasion', 'murder', 'flood', 'earthquake'
                ],
        'surprise': ['shock', 'unexpected', 'sudden', 'reveal', 'twist', 'gasp', 'wow', 'discovery', 'gift', 'accident',
                        'burst', 'arrival', 'news', 'fireworks', 'announcement', 'prize', 'miracle', 'explosion', 'win',
                        'reunion', 'change', 'appearance', 'event', 'drop', 'return', 'reaction', 'laugh', 'jump', 'flash',
                        'entrance', 'shift', 'message', 'response', 'mirage', 'climax', 'alert', 'uproar', 'opening', 'arrival',
                        'encounter', 'stunt', 'scene', 'startle', 'excitement', 'surprise', 'expression', 'confetti', 'applause'
                    ],
        'neutral': ['table', 'chair', 'room', 'wall', 'door', 'floor', 'object', 'stone', 'pen', 'book', 'paper',
                        'window', 'light', 'clock', 'cup', 'plate', 'shoe', 'sock', 'bed', 'lamp', 'mirror', 'ceiling',
                        'pencil', 'notebook', 'carpet', 'bottle', 'bag', 'screen', 'phone', 'laptop', 'key', 'lock',
                        'button', 'glass', 'handle', 'desk', 'case', 'box', 'cable', 'plug', 'drawer', 'shelf', 'folder',
                        'picture', 'fan', 'air', 'plug', 'cord', 'container', 'object'
                    ]
    }
    return contexts.get(ground_truth_label.lower(), [])

def get_predefined_body_language_attributes(ground_truth_label):
    body_language = {
        'happy': ['smile', 'grin', 'laugh', 'giggle', 'hug', 'wave', 'clap', 'nod', 'cheer', 'highfive', 'dance',
                    'twirl', 'bounce', 'jump', 'leap', 'skip', 'spin', 'wink', 'tilt', 'openarms', 'lightsteps',
                    'handshake', 'relaxedposture', 'brighteyes', 'raisedeyebrows', 'openstance', 'headtilt', 'beaming',
                    'radiate', 'twinklingeyes', 'foottap', 'jollywalk', 'headnod', 'sway', 'widegrin', 'fistpump',
                    'energeticgesture', 'headbob', 'happyshrug', 'shoulderbounce', 'playfulgesture', 'warmsmile',
                    'quicksteps', 'armlift', 'gentletouch', 'openhands', 'freearms', 'expressivehands', 'glowingface'
                ],
        'disgust': ['grimace', 'frown', 'recoil', 'flinch', 'turnaway', 'wrinklenose', 'sneer', 'cringe', 'pursedlips',
                    'avertgaze', 'stepback', 'covernose', 'tightlips', 'eyeaversion', 'shudder', 'pullback', 'jawclench',
                    'stifledbreath', 'narroweyes', 'stiffposture', 'squint', 'snarl', 'gag', 'twitch', 'nosecurl',
                    'sourface', 'tightjaw', 'lookaway', 'closedposture', 'slump', 'shrug', 'lipcurl', 'bodytilt',
                    'stareblank', 'tenseface', 'shoulderraise', 'hastystep', 'neckpull', 'armfold', 'wince',
                    'wrinkleforehead', 'crossarms', 'leanaway', 'jerkback', 'squirm', 'lowerhead', 'sidelook', 'sniff'
                ],
        'sad': ['slump', 'sigh', 'tear', 'weep', 'sob', 'cry', 'lowerhead', 'hunch', 'shrug', 'downcasteyes',
                    'stillness', 'slowwalk', 'dragfeet', 'frown', 'quiet', 'collapse', 'foldedarms', 'stoop',
                    'shiver', 'wilt', 'blankstare', 'shouldersdrop', 'closedposture', 'starefloor', 'headtiltdown',
                    'wrinklebrow', 'tremble', 'turnaway', 'armscrossed', 'tightlips', 'sleeveswipe', 'lipbite',
                    'nervoushands', 'downwardgaze', 'curlup', 'holdknees', 'cling', 'slowbreath', 'slouchedback',
                    'headshake', 'backturn', 'collapseknees', 'sadsmile', 'wipedowncheek', 'silenttears', 'deepbreath',
                    'sunkeneyes', 'dragstep', 'slowblink'
                ],
        'angry': [ 'glare', 'scowl', 'frown', 'clenchedfist', 'crossarms', 'stomping', 'grittedteeth', 'flarednostrils',
                    'tenseposture', 'stiffneck', 'puffedchest', 'aggressivegesture', 'jabfinger', 'snarl', 'yell',
                    'shout', 'tightjaw', 'stare', 'throw', 'slam', 'march', 'huff', 'pace', 'poundfist', 'sneer',
                    'tensedshoulders', 'eyebulge', 'jawclench', 'breathheavily', 'swingarm', 'snap', 'jerk',
                    'sharpgesture', 'raisevoice', 'necktense', 'brushturn', 'slap', 'kick', 'charge', 'crackknuckles',
                    'eyesquint', 'crushobject', 'shakeneck', 'twistface', 'tilthead', 'strike', 'bang', 'stepforward'
                ],
        'fear': ['freeze', 'tremble', 'shiver', 'shrink', 'stepback', 'flinch', 'duck', 'hide', 'cower',
                    'coverface', 'wideeyes', 'stare', 'gasp', 'grip', 'hugself', 'shake', 'holdbreath', 'curlup',
                    'scream', 'squint', 'lookaround', 'bitefingernail', 'suddenjerk', 'hunch', 'sway', 'backoff',
                    'hesitate', 'pant', 'cling', 'jump', 'crouch', 'retreat', 'wince', 'avoidgaze', 'tenseposture',
                    'stiffen', 'pressback', 'whimper', 'blinkfast', 'grab', 'slouch', 'stepaside', 'slowstep',
                    'openmouth', 'fidget', 'tilthead', 'stepaway', 'armsup', 'quickglance'
                ],
        'surprise': ['gasp', 'jump', 'blink', 'wideneyes', 'raisedeyebrows', 'openmouth', 'stepback', 'tilthead',
                        'suddenfreeze', 'handtochest', 'stare', 'startle', 'jerk', 'mouthcover', 'pause', 'recoil',
                        'lookaround', 'quickstep', 'backwardstep', 'headturn', 'handsup', 'spinflinch', 'stiffen', 'gape',
                        'nod', 'smilepause', 'claphand', 'shrug', 'lookup', 'standstill', 'armsraise', 'doubletake',
                        'shiftnervously', 'flickeyes', 'eyebrowlift', 'mouthpart', 'grinpause', 'eyeslight', 'shockpause',
                        'armfling', 'spinstop', 'shiftposture', 'breathhold', 'covermouth', 'lookaside', 'alertposture',
                        'handjerk', 'stepaside', 'watchclosely'
                ],
        'neutral': [ 'still', 'relaxed', 'upright', 'stand', 'sit', 'walk', 'blink', 'nod', 'look', 'breathe',
                    'glance', 'turn', 'pause', 'shift', 'adjust', 'tilthead', 'crosslegs', 'armsrest', 'neutralface',
                    'casualstance', 'lookforward', 'blinkslowly', 'step', 'standstraight', 'normalbreath', 'readjust',
                    'uncrossarms', 'headstraight', 'lookaround', 'normalposture', 'watch', 'gesturelightly',
                    'stepforward', 'armsbytheside', 'sitback', 'listen', 'inhale', 'exhale', 'neutralnod', 'expressionless',
                    'mouthclosed', 'handsrest', 'stepaside', 'lightshift', 'quietstance', 'holdstill', 'briefnod', 'slowturn'
                ]
    }
    return body_language.get(ground_truth_label.lower(), [])

def get_predefined_emotion_attributes(ground_truth_label):
    emotion = {
        'happy': [
                    "surprised", "warm smile", "helplessness", "tension", "surprise", "joy", "happy", "relaxed",
                    "slight smile", "wide eyes", "wide open eyes", "slightly open mouth", "smile", "contentment",
                    "smiling", "eyes slightly closed", "laughing", "bright eyes", "sharp eyes", "lips slightly parted",
                    "friendly expression", "eyes closed", "laughter", "open eyes", "relaxed facial muscles", "joyful",
                    "calm expression", "anger", "excitement", "radiant smile", "more pronounced smile",
                    "smile becomes more pronounced", "gentle smile", "happy eyes", "smiling warmly",
                    "relaxed expression", "happy facial expression", "serious expression", "laughing heartily",
                    "joyful smile", "furrowed brows", "tightly pressed lips", "radiating joy", "look of confusion",
                    "joyful expression", "shining eyes", "anxiety", "slightly widened eyes", "slight frown",
                    "open mouth", "furrowed brow", "open mouth showing teeth", "direct gaze", "rapid blinking",
                    "wide open mouth", "happiness", "focused gaze", "smiles", "tightly closed lips",
                    "slightly upturned lips", "calm", "widened eyes", "gently closed eyes", "happy expression",
                    "mouth slightly upturned", "slightly raised eyebrows", "wide-open eyes", "genuine joy",
                    "slightly squinted eyes", "upturned lips", "melancholic expression", "slight confusion",
                    "direct eye contact", "relaxed gaze", "neutral expression", "broad smile", "focused expression",
                    "mouth slightly open", "raised eyebrows", "serious", "angry", "sadness", "confusion",
                    "satisfaction", "slightly closed eyes", "sad expression", "deep sadness",
                    "downturned mouth corners", "slightly furrowed brows", "initial surprise", "wide smile",
                    "joyful eyes", "upturned mouth", "eyes slightly squinted", "frequent blinking", "friendly smile",
                    "smiles warmly", "smiles gently", "unease", "eyes gently closed",
                    "slight upward curve at the corners of her mouth", "closed lips", "wider smile", "showing teeth",
                    "natural smile", "slight upward curve at the corners of his mouth", "gentle expression",
                    "curiosity", "relaxed facial expression", "eyebrows raised", "bright smile", "surprised expression",
                    "delight", "pronounced smile", "eyes wide open", "squinted eyes", "excited",
                    "slight upward curve of the mouth", "extreme joy", "tense facial muscles",
                    "subtle changes in eyes", "warm expression", "eyes shining", "corners of mouth turning up",
                    "serious facial expression", "gentle gaze", "slightly parted lips", "focused",
                    "slight upward curve of mouth", "slightly agape mouth", "signs of fatigue", "moist eyes",
                    "relaxed eyes", "eyes widen", "slightly furrowed brow", "squinting eyes", "relaxed demeanor",
                    "downturned mouth", "intense gaze", "closed eyes", "eyes open", "slightly upturned mouth",
                    "slightly upturned corners of the mouth", "agape mouth", "cheerful expression",
                    "negative facial expression", "exaggerated facial expression", "relaxation",
                    "downturned corners of the mouth", "narrowed eyes", "steady gaze", "focused look",
                    "expression of joy", "mouth opening and closing", "large eyes",
                    "smile evolves from slight to broad"
                ],
        'disgust': [
                        "raised eyebrows", "opened mouth", "lowered position", "sad facial expression",
                        "slightly furrowed brows", "downturned mouth corners", "furrowed brows", "slightly open eyes",
                        "tears in her eyes", "eyes slightly closed", "mouth moving as if speaking",
                        "facial expression somewhat displeased", "look of disgust", "open mouth showing teeth",
                        "concentration", "determination", "subtle changes in gaze",
                        "lip movements indicating inner effort and resolve", "slightly open mouth",
                        "negative expression", "anger", "pain", "tightly closed lips",
                        "downturned corners of the mouth", "displeasure", "discontent", "negative facial expression",
                        "wide-open eyes", "tightly pressed lips", "showing teeth", "deep frown lines",
                        "slightly disheveled hair", "furrowed eyebrows", "serious expression", "downcast gaze",
                        "closed eyes", "pursed lips"
                ],
        'sad': [
                    "furrowed brows", "slightly open mouth", "furrowed brow", "downturned corners of mouth",
                    "downturned corners of the mouth", "surprise", "confusion", "sadness", "unease", "downturned mouth",
                    "sad", "slightly furrowed brows", "eyes closed", "despair", "fatigue", "helplessness", "anger",
                    "tears in her eyes", "raised eyebrows", "slight smile", "downturned facial muscles",
                    "slightly parted lips", "negative expression", "negative facial expression", "tears welling up",
                    "deep sadness", "serious facial expression", "intense gaze", "wide eyes", "contemplation",
                    "downturned mouth corners", "furrowed eyebrows", "downturned eyes", "tight lips", "sad expression",
                    "tightly pressed lips", "moist eyes", "wide open eyes", "agape mouth", "direct gaze", "parted lips",
                    "tears streaming down her face", "determination", "smile", "serious expression", "slight frown",
                    "tears flowing", "fear", "pain", "agony", "closed eyes", "wide-open eyes", "open mouth", "tears",
                    "frustration", "sad facial expression", "solemn expression", "signs of fatigue", "tension",
                    "contemplative expression", "calm", "serious", "tense facial expression", "slightly widened eyes",
                    "slightly downturned mouth", "downward turn of the mouth", "anxiety", "downturned lips",
                    "slightly open eyes", "extreme sadness", "tears streaming down cheeks", "sad eyes", "focused",
                    "deep sorrow", "slightly furrowed brow", "lips slightly parted", "eyes slightly closed",
                    "wide open mouth", "sorrow", "slightly agape mouth", "tense expression", "fixed gaze",
                    "slightly closed eyes", "tightly closed lips", "blurred face", "slightly moist eyes",
                    "hint of sadness", "concern", "tears glistening", "downward-turned mouth", "look of confusion",
                    "tears falling", "crying", "calm expression", "tense facial muscles", "widened eyes",
                    "tears streaming down his face", "pressed lips", "mouth slightly open", "extreme pain", "wrinkles",
                    "expression of sadness", "disgust", "downward gaze", "focused gaze", "sorrowful",
                    "melancholic expression", "angry", "narrowed eyes"
                ],
        'angry': [
                    "wide eyes", "slightly open mouth", "raised eyebrows", "slightly parted lips", "serious look",
                    "angry look", "surprise", "anger", "wide open eyes", "furrowed brows", "open mouth",
                    "slightly agape mouth", "furrowed brow", "intense gaze", "open mouth showing teeth",
                    "wide-open eyes", "downward gaze", "serious expression", "downturned corners of the mouth",
                    "widened eyes", "tense facial muscles", "expression of anger", "anxiety", "confusion",
                    "tightly pressed lips", "tension", "wide open mouth", "moving mouth", "dissatisfaction",
                    "tense expression", "direct gaze", "mouth slightly open", "eyebrows raised", "focused expression",
                    "angry facial expression", "mouth moving", "discontent", "intense anger", "downturned mouth",
                    "furrowed eyebrows", "tightened facial muscles", "tense", "disbelief", "negative facial expression",
                    "extreme pain", "downturned mouth corners", "serious facial expression", "tightly closed lips",
                    "determination", "downturned eyes", "intense expression", "sharp eyes", "slightly raised eyebrows",
                    "tense facial expression", "slight frown", "serious", "focused", "focused gaze", "calm",
                    "agape mouth", "mouth wide open", "exaggerated facial expression", "angry", "disgust", "pain",
                    "angry expression", "fear", "eyes widen", "mouth opens wide", "frowning", "fierce expression",
                    "surprised expression", "clenched teeth", "revealing teeth", "fixed gaze", "sadness",
                    "focused eyes", "intense eyes", "tense facial expressions", "narrowed eyes", "frustration",
                    "sharp gaze", "calm expression", "surprised", "wide-open mouth", "excitement", "agitation",
                    "intense dissatisfaction", "slightly furrowed brows", "calm to angry expression",
                    "negative expression", "extreme anger", "glaring eyes", "opened mouth", "showing teeth",
                    "extreme surprise", "intense emotional turmoil", "tense brows", "mouth open", "excited",
                    "frequent blinking", "mouth agape", "shock", "helplessness", "deep-seated anger"
                ],
        'fear': [
                    "furrowed brows", "wide-open eyes", "tightly pressed lips", "extreme fear", "wide eyes",
                    "slightly open mouth", "raised eyebrows", "wide open eyes", "slightly agape mouth", "fear",
                    "surprise", "pain", "determination", "anger", "urgency", "wide open mouth",
                    "tense facial muscles", "intense emotions", "tension", "confusion", "anxiety",
                    "slightly parted lips", "serious expression", "unease", "calm", "fatigue", "distress",
                    "downturned mouth", "helplessness", "tense expression", "furrowed eyebrows", "despair",
                    "expression of fear", "expression of surprise", "surprised expression", "terror",
                    "downturned mouth corners", "agape mouth", "mouth slightly open", "open mouth",
                    "eyes wide open", "intense anxiety", "agony", "brows furrowed", "expression of tension",
                    "extreme pain", "tense", "mouth wide open", "eyes widen", "expression of extreme fear",
                    "surprised", "mouth agape", "sadness", "worry", "widened eyes", "shock", "downturned eyes",
                    "calm expression", "intense pain", "extreme surprise", "furrowed brow",
                    "tightened facial muscles", "tightly closed lips", "showing teeth", "slightly agape lips",
                    "panic"
                ],
        'surprise': [
                        "slightly furrowed brows", "serious expression", "wide open eyes", "surprise", "confusion",
                        "slightly agape mouth", "furrowed brows", "open mouth", "wide eyes", "slightly open mouth",
                        "focused", "serious", "sharp eyes", "widened eyes", "slightly parted lips", "tension", "fear",
                        "focused expression", "focused gaze", "wide-open eyes", "anxiety", "anger",
                        "downturned mouth corners", "surprised expression", "mouth agape", "eyes widen", "mouth opens",
                        "raised eyebrows", "mouth slightly open", "surprised", "angry", "eyebrows raised", "calm",
                        "tense expression", "tightly closed lips", "tightly pressed lips", "open mouth showing teeth",
                        "furrowed brow", "unease", "concern", "expression of surprise", "slight frown", "wide open mouth",
                        "calm expression", "tense facial expression", "downturned mouth", "pressed lips", "agape mouth",
                        "tense facial muscles", "puzzled", "confused", "excitement", "complex expression",
                        "astonishment", "curiosity", "puzzled expression", "surprised eyes", "intense gaze",
                        "extreme surprise", "expression of fear", "slightly widened eyes", "deep eyes", "fatigue", "shock",
                        "alertness", "determination", "furrowed eyebrows", "tense", "mouth slightly agape", "sadness",
                        "eyes wide open"
                ],
        'neutral': [
                    "serious expression", "furrowed brows", "wide-open eyes", "tense", "downturned eyes",
                    "confusion", "tension", "anxiety", "focused expression", "look of confusion", "wrinkles",
                    "tightly pressed lips", "wide open eyes", "agape mouth", "mouth slightly open", "serious",
                    "focused", "tightly closed lips", "direct gaze", "open mouth", "calm expression",
                    "narrowed eyes", "surprised expression", "focused gaze", "slightly open mouth", "slight frown",
                    "moist eyes", "widened eyes", "open mouth showing teeth", "wide eyes",
                    "downturned corners of the mouth", "downturned mouth", "serious facial expression",
                    "intense gaze", "contemplative", "angry", "furrowed brow", "unease", "surprise",
                    "slightly agape mouth", "negative expression", "closed lips", "slightly furrowed brows",
                    "subtle facial movements", "raised eyebrows", "frowning", "sadness",
                    "negative facial expression", "fatigue", "helplessness", "slightly open eyes",
                    "furrowed eyebrows", "focused eyes", "calm", "relaxed expression", "tense expression",
                    "downturned mouth corners", "deep eyes", "tight lips", "closed mouth", "closed eyes",
                    "eyes closed", "slightly parted lips", "displeasure", "fear", "sharp eyes", "downturned lips",
                    "open eyes", "puzzled", "fixed gaze", "serious look", "slightly furrowed brow", "frustration",
                    "slightly widened eyes", "solemn expression", "slightly downturned mouth",
                    "slightly closed eyes", "slightly raised eyebrows", "smile", "melancholic expression",
                    "downward gaze", "calmness", "disgust", "signs of fatigue", "subtle changes in facial muscles",
                    "relaxed brow", "pensive", "determination", "tense facial expression", "concentration", "anger",
                    "sad expression", "pensive expression", "nervous expression", "pain",
                    "smiling", "mouth moving as if speaking"
                ]
    }
    return emotion.get(ground_truth_label.lower(), [])
   

def compute_reward_reasoning_attributes(reasoning_attribute_results, len_gt):
    reasoning_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(reasoning_attribute_results)):
        temp_reasoning = reasoning_attribute_results[i][0]
        temp_confidence = reasoning_attribute_results[i][1]
        print('158 ^^^^^^^^^^^^^^^^^^^^', temp_reasoning)

        temp_reasoning_reward = 0.95 if temp_reasoning else 0.05
        reasoning_reward += temp_reasoning_reward
    
    if len_gt>=len(reasoning_attribute_results):
        reasoning_reward = reasoning_reward/len_gt
    else:
        reasoning_reward = reasoning_reward/len(reasoning_attribute_results)
    return reasoning_reward

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

def accuracy_reward_reasoning_attributes(completions, solution, **kwargs):
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

                print('236', student_think)
                print('237', student_answer)

                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9, 'Label': 'sad'}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8, 'Label': 'happy'}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox2(student_answer)
                print('242', student_answer_bbox)

                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results, reasoning_attribute_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox, student_think)
                    print('********457********', reasoning_attribute_results)

                    ### new iou reward
                    reward = compute_reward_reasoning_attributes(reasoning_attribute_results, len(ground_truth_bbox))
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

                print('236', student_think)
                print('237', student_answer)

                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9, 'Label': 'sad'}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8, 'Label': 'happy'}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox2(student_answer)
                print('242', student_answer_bbox)

                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.05
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results, label_results, reasoning_attribute_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox, student_think)
                    print('********250********', label_results)

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
accuracy_reward_reasoning_attributes_print = accuracy_reward_reasoning_attributes(completions, solution)
print('603', accuracy_reward_reasoning_attributes_print)


# python grpo_qwen_2_5_dfew_reward_add_emotion_label_with_attribute_test_files_for_label.py







# [{'Position': [86, 134, 283, 331], 'Confidence': 0.95, 'Label': 'happy'}, ...]
#  conda activate Visual-RFT
#  cd /scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1
#  python grpo_qwen_2_5_dfew_reward_add_emotion_label_with_attribute_test_files_for_label.py
# qsub -I -P kf09 -q gpuvolta -l ngpus=1,ncpus=12,mem=64GB,walltime=00:50:00