import os
import os.path as osp
import yaml
import argparse
import torch
import pandas as pd
from uuid import uuid4
from openai import OpenAI
from tqdm import tqdm
import pdb

from dataset import DatasetSampler

import random
import time
import re
from PIL import Image
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ["OPENAI_API_KEY"] = "sb-60b04634ed26bc0d447ff7a02198c6b5652a465e3928f5b6"
os.environ["OPENAI_BASE_URL"] = "https://api.openai-sb.com/v1"

import reka

# You can either set the API key as below, or use the
# environment variable export REKA_API_KEY="your-api-key"
reka.API_KEY = "e3a4ad937e438a7839bc29d3dc1620fe82cef52da60aa24c935971cda36d610f"
# reka.API_KEY = "1aa0c2dfcf801e1934f3d896db38cef724cf6fff0a975aa0e8697bac674b9bd4"

def encode_image_file_to_base64(image_path):
    if image_path.endswith('.png'):
        tmp_name = f'{timestr(second=True)}.jpg'
        img = Image.open(image_path)
        img.save(tmp_name)
        result = encode_image_file_to_base64(tmp_name)
        os.remove(tmp_name)
        return result
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode('utf-8')


def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    ret = encode_image_file_to_base64(tmp)
    os.remove(tmp)
    return ret
    


def call_gpt(history_chat, model="gpt-4", temp_gpt=0.0):
    client = OpenAI()
    chatgpt_messages = []
    for role, message in history_chat:
        ret = dict()
        if message:
            ret["role"] = role
            ret['content'] = message
            chatgpt_messages.append(ret)    
    success = False
    while not success:
        try:
            response = client.chat.completions.create(model="gpt-4-vision-preview",messages = chatgpt_messages,max_tokens=512)
            #print("chatgpt_messages",chatgpt_messages)
            reply = response.choices[0].message.content
            print("reply", reply)
            total_tokens = response.usage.total_tokens
            success = True
            return reply, total_tokens
        except Exception as e:
            print('[Worker] an exception occured: %s (%s). retrying in 3 minutes.' % (type(e), str(e)))
            time.sleep(30)
def parse_final_answer(gpt_response):
    # ===== Parse the paragraph starting with analysis. =====
    try:
        analysis_result = re.search('Response:(.*)', gpt_response)
        analysis_string = analysis_result.group(1).strip()
        return analysis_string
    except Exception as e:
        print("Can not parse analysis")
        return None



def GPT(dataset, data_ids,  save_path='/mnt/lustre/liushuo/VLMEvalKit/work_dirs/reka_flash/2'):
    for data_id in tqdm(data_ids):
        img_id, img_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption, first_turn_category, second_turn_category, third_turn_category, first_turn_answer, second_turn_answer, third_turn_answer, third_turn_demands  = dataset.fetch_data(data_id)

        result_path = os.path.join(save_path, '{}.yaml'.format(img_id))
        if os.path.isfile(result_path):
            continue
        from pathlib import Path
        import shutil
        import requests
        from io import BytesIO
        from PIL import Image

        source_file = img_path

        # 目标文件路径
        destination_file = os.path.join('images', Path(img_path).name)

        # 复制文件
        # shutil.copyfile(source_file, destination_file)

        def download_image(url, name):
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(f'download/{name}')

            return img

        # # 使用示例
        image_name = Path(img_path).name
        if image_name == "340_tested_skill_technical_support_android_phone_apple_device_blur_business_close_up_computer_connection_contemporary-1535567.jpg!d.png":
            image_name = "340_tested_skill_technical_support_android_phone_apple_device_blur_business_close_up_computer_connection_contemporary-1535567.jpg.d.png"
        if image_name == "329_tested_skill_ocr_math_IdentifyError&Correct.png":
            image_name = "329_tested_skill_ocr_math_IdentifyError.Correct.png"
        if image_name == "443_tested_skill_physical_knowledge_billiard-balls-vector(1).png":
            image_name = "443_tested_skill_physical_knowledge_billiard-balls-vector.1.png"

        image_url = f'https://github.com/shirlyliu64/InsightChain/visit_bench_images/{image_name}'

        if image_name == "273_tested_skill_who_to_call_shoplifting.png":
            image_url = "https://github.com/KainingYing/image/releases/download/two/273_tested_skill_who_to_call_shoplifting.png"
        if image_name == "136_tested_skill_game_playing_a1a6468b7b666b1b copy.png":
            image_url = "https://github.com/KainingYing/image/releases/download/two/136_tested_skill_game_playing_a1a6468b7b666b1b.copy.png"
        # try:
        # download_image(image_url, Path(img_path).name)
        # except:
        #     print(Path(img_path).name)
        #     continue


        response1 = first_turn_answer

        # time.sleep(10)

        response2 = second_turn_answer

        # time.sleep(10)

        response3 = reka.chat(
            third_turn_instruction,
            conversation_history=[
                {
                    "type": "human",
                    "text": first_turn_instruction,
                    "media_url": image_url,
                    "media_type": "image",
                },
                {"type": "model", "text": response1},
                {
                    "type": "human",
                    "text": second_turn_answer,
                },
                {"type": "model", "text": response2},
            ],
        )["text"]

        time.sleep(10)

        results = {}
        results['image_path'] = img_path
        #results['instruction_conditioned_caption'] = instruction_conditioned_caption
        results['first_turn_instruction'] = first_turn_instruction
        results['first_turn_answer'] = response1
        results['second_turn_instruction'] = second_turn_instruction
        results['second_turn_answer'] = response2
        results['third_turn_instruction'] = third_turn_instruction
        results['third_turn_answer'] = response3
        # results["first_tokens"] = first_token
        # results["second_tokens"] = second_token
        # results["third_tokens"] = third_token

        if save_path:
            with open(result_path, 'w') as f:
                yaml.dump(results, f)

def parse():
    parser = argparse.ArgumentParser(description='GPT Args.')
    parser.add_argument('--data_root', type=str, default="/nvme/share/liushuo/dataset/visit_bench/", 
                        help='root path to the dataset')
    parser.add_argument('--save_root', type=str, default='./evaluation_result/exp_result/gpt4v/', 
                        help='root path for saving results')
    parser.add_argument('--model', type=str, default='chatgpt', choices=['chatgpt', 'gpt4'],
                        help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
    parser.add_argument('--openai_key', type=str,  default='', 
                        help='OpenAI Key for GPT-3.5/4 API')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--dataset', type=str, default='visit_gpt4',
                        help='Names of the dataset to use in the experiment. Valid datasets include vcr_val, ve_dev. Default is vcr_val')    
    args = parser.parse_args()
    return args
    
    
def main(args):
    # Set OpenAI
    #OPENAI_API_KEY = args.openai_key
    #openai.api_key = OPENAI_API_KEY
    random.seed(args.seed)

    dataset = DatasetSampler(dataset_root="/mnt/lustre/liushuo/VLMEvalKit/mte")
    print('Finish loading data')
    
    save_path = args.save_root

    GPT(dataset, dataset.ids,)
    

if __name__ == '__main__':
    args = parse()
    main(args)
