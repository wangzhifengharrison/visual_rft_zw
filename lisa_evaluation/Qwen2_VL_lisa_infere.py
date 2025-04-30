import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
import logging
from tqdm import tqdm
import re
# from process_utils import pred_2_point, extract_bbox
import math

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)
img2description = dict()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Load Qwen2-VL-2B model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/path/to/your/checkpoint-498", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
).eval()

processor = AutoProcessor.from_pretrained("/path/to/your//checkpoint-498")

logging.info("Model and processor loaded successfully")

def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)

def prepare_inputs(img_path, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")

def extract_bbox(response):
    try:
        match = re.search(r"\[(\d+),(\d+),(\d+),(\d+)\]", response)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        logging.error(f"Error extracting bbox: {e}")
        return None

def compute_iou(boxA, boxB):
    """
    计算 IoU (Intersection over Union)
    boxA, boxB 格式: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_model(tasks):
    results = []
    box_res =[]
    for task in tasks:
        logging.info(f"Processing task: {task}")
        ious = []
        screenspot_data = json.load(open(f"path/to/lisa_{task}.json", 'r'))
        data_per_gpu = math.ceil(len(screenspot_data) / int(os.environ['SPLIT_NUM']))
        start_idx = int(os.environ['SPLIT']) * data_per_gpu
        end_idx = min(start_idx + data_per_gpu, len(screenspot_data))
        screenspot_data = screenspot_data[start_idx:end_idx]
        
        for item in tqdm(screenspot_data):
            img_path = item['image_path']
            try:
                image = process_image(img_path)
                w, h = image.size
                instruction = item["instruction"][0]
                bbox = item["boxes"][0]
                bbox = [
                    bbox[0] / image.size[0],
                    bbox[1] / image.size[1],
                    bbox[2] / image.size[0],
                    bbox[3] / image.size[1],
                ]
                inputs = prepare_inputs(img_path, instruction)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(response)
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                matches = re.findall(pattern, response)
                x1, y1, x2, y2 = map(int, matches[0])
                pred_bbox = [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000]

                iou = compute_iou(pred_bbox, bbox)
                box_res.append(
                    {
                        "image_pth": item['image_path'],
                        "pred_bbox": pred_bbox,
                        "thinking_process": response
                    }
                )
                ious.append(iou)
            except Exception as e:
                ious.append(0)
        json.dump(box_res, open(f"tmp/resbox_{os.environ['SPLIT']}_r1_w_think_7b.json", 'w'), indent=4)
        json.dump(ious, open(f"tmp/res_{os.environ['SPLIT']}.json", 'w'), indent=4)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["val", "test"]
    else:
        tasks = [args.task]

    results = evaluate_model(tasks)
