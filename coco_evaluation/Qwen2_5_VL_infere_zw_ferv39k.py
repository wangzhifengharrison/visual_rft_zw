import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
import pandas as pd

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


import logging
# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

LOG_FILE = "/home/zhe030/zhe030/ZFW/Visual-RFT/logs/inference_ferv39k_rank_{rank}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE.format(rank=os.environ.get("LOCAL_RANK", "0"))),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

def setup_distributed():
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

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

def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

import matplotlib.patches as patches
def plot_bbox(image_path, bbox_list):
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for bbox in bbox_list:
        position = bbox["Position"]
        x1, y1, x2, y2 = position
        # x1 /= 1000.0
        # y1 /= 1000.0
        # x2 /= 1000.0
        # y2 /= 1000.0

        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height
    
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

def trans_bbox(image_height, image_width, position):
        x1, y1, x2, y2 = position
    
        # x1 /= 1000.0
        # y1 /= 1000.0
        # x2 /= 1000.0
        # y2 /= 1000.0
    
        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height

        return [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ]
    

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

### model path and model base
# model_path = "./share_models/Qwen2-VL-2B-Instruct"
# ori_processor_path = "./share_models/Qwen2-VL-2B-Instruct"

model_path = "/home/zhe030/zhe030/ZFW/Visual-RFT/share_models/Qwen2.5-VL-3B-Instruct_GRPO_dfew_train_slurm"
ori_processor_path = "/home/zhe030/zhe030/ZFW/Visual-RFT/share_models/Qwen2.5-VL-3B-Instruct"
excel_path = "/home/zhe030/zhe030/ZFW/Visual-RFT/share_data/Inference_data/ferv39k_inference/valid_valid_ferv39k_data_for_inference_add_width_height.xlsx"
images_path  = "/home/zhe030/zhe030/ZFW/Visual-RFT/share_data/Inference_data/ferv39k_inference/ferv39k/"
def run():
    # 1) Distributed setup
    # torch.cuda.init()
    rank, world_size, local_rank = setup_distributed()
    torch.cuda.set_device(local_rank)
    # 2) Load model + processor
    # Qwen2_5_VLForConditionalGeneration # Qwen2VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained( 
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(ori_processor_path) 
    model = model.to(f"cuda:{local_rank}")
    model = model.eval()

    # Synchronize before data loading
    # torch.distributed.barrier()
    # 3) Read Excel and prep image paths
    df = pd.read_excel(excel_path)
    image_paths_temp = []
    for idx, row in df.iterrows():
        video_name = str(row["video_name"])
        # build path as: <images_path>/<video_name>/<video_name>.jpg
        img_file = f"{video_name}.jpg"
        full_path = os.path.join(images_path, img_file)
        image_paths_temp.append(full_path)

    df["image_path"] = image_paths_temp
    # add a column for the responses
    df["full_response"] = None
    # Convert to list of dicts and shard across ranks
    records = df.to_dict(orient="records")
    split_records = records[rank::world_size]
    indices = list(range(rank, len(df), world_size))
    # make a copy for just this rank
    df_rank = df.loc[indices].copy()
    logger.info(f"Rank {rank} will process {len(split_records)} frames")

    # # Create output file per rank
    output_path = f'/home/zhe030/zhe030/ZFW/Visual-RFT/logs/ferv39k_each_predictions_rank_{rank}.jsonl'

    ### Traverse all images in val.
    error_count = 0
    pred_results = []
    for rec in tqdm(split_records, desc=f"Rank {rank}"):
        log_2 = []
        image_path = rec["image_path"]
        # Use Excel’s width/height columns:
        image_width, image_height = rec["width"], rec["height"]  
        # Build your prompt/messages exactly as before
        question = (
                f"Detect the face in the image"
                ", and provide the bounding boxes of the face (integer)"
                ", and confidence (between 0 and 1, with two decimal places) and identify the emotion of the face from ['fear','angry','happy','sad', 'disgust','neutral','surprise'] based on the image by analysing the facial expression, posture and context.\n"
                f"If no face in the image, "
                "return 'No Objects'.\n"
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n"
                "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number, 'Label': ''}, ...]</answer>\n"
                "Please strictly follow the format."
            )
        query = '<image>\n' + question
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ] + [{"type": "text", "text": query}],
                }
            ]
        try:
            # 4a) Tokenize + preprocess vision
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            log_2.append(f"{image_path}")

            # 4b) Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = response[0]
            full_response = response
            # write it back into df
            # write back into the rank‐specific DataFrame
            df_rank.loc[df_rank["image_path"] == image_path, "full_response"] = full_response
            # df.loc[df["image_path"] == image_path, "full_response"] = full_response
            # df[image_path] = full_response #### add full response to df
            logger.info(response)
            # Fix possible formatting errors in the response.
            response = response.replace("[[",'[')
            response = response.replace("]]",']')
            response = response.replace("\n",'')
            response = response.replace(", ...",'')
            # logger.info("\033[92m" + response + "\033[0m")

            # extract answer
            content_match = re.search(r'<answer>(.*?)</answer>', response)
            response = content_match.group(1).strip() if content_match else response.strip()
            response = '<answer>'+response+'</answer>'
            log_2.append(f"come to line 325 {full_response}")
            # log_2.append(f"come to line 331 {response}")

            # extract bbox
            try:
                bbox_list = extract_bbox(response)
                log_2.append(f"come to line 330 {image_path},{bbox_list}")

                if bbox_list==None or type(bbox_list[0])==str:
                    pass
                else:
                    bbox_list = remove_duplicates(bbox_list)
                    log_2.append(f"come to line 336 {image_path}")

                    for bbox in bbox_list:
                        temp_bbox = trans_bbox(image_height, image_width, bbox['Position'])
                        temp_confidence = bbox['Confidence']
                        new_pred_dict = {
                            'image_id': image_path,
                            'groundtruth': rec['face_bboxes_resized'],  # Assuming category_id is fixed for this task
                            'bbox': temp_bbox,
                            'score': temp_confidence,
                            'response': response,
                            'full_response': full_response
                        }
                        pred_results.append(new_pred_dict)
                        bbox_count += 1
                        logger.info('bbox_count: '+str(bbox_count))
                        logger.info('full response: '+str(full_response))
                        print(323, new_pred_dict)
                        # Save to JSON Lines file
                        with open('/home/zhe030/zhe030/ZFW/Visual-RFT/predictions/each_predictions_ferv39k.jsonl', 'a') as f:
                            json.dump(new_pred_dict, f)
                            f.write('\n')
                    log_2.append(f"come to line 357 {image_path}")
            except Exception as e:
                log_2.append(f"has error 1 {image_path}")
                error_count+=1
                logger.info('Error number: ' + str(error_count)) 
        except Exception as e:
                log_2.append(f"has error 2 {image_path}")
                error_count+=1
                logger.info('Error number: ' + str(error_count)) 
        with open(output_path, 'a') as f:
            json.dump([log_2], f)
            f.write('\n')
    # df.to_csv('/home/zhe030/zhe030/ZFW/Visual-RFT/predictions/dfew_inference_results.csv', index=False)
    #  at the end, save out just this rank’s CSV
    out_csv = f"/home/zhe030/zhe030/ZFW/Visual-RFT/predictions/ferv39k_csv/ferv39k_inference_results_rank_{rank}.csv"
    df_rank.to_csv(out_csv, index=False)
    return [error_count, pred_results]
    

def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus # n_gpus
        result_lists = run()
        # with Pool(world_size) as pool:
        #     func = functools.partial(run, world_size=world_size)
        #     result_lists = pool.map(func, range(world_size))
        # logger.info(world_size)
        # global_count_error = 0
        # global_results = []
        # for i in range(world_size):
        #     global_count_error += int(result_lists[i][0])
        #     global_results = global_results + result_lists[i][1]

        # logger.info('Error number: ' + str(global_count_error))  
        ### save path
        with open('/home/zhe030/zhe030/ZFW/Visual-RFT/predictions/prediction_results_ferv39k.json', 'w') as json_file:
            json.dump(result_lists, json_file)
        logger.info("Done")
        logger.info('finished running')
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
## must run with torchrun in interactive mode
# torchrun  --nnodes=1  --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d  Qwen2_VL_coco_infere_zw.py 