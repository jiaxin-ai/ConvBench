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
# from chat import VCRConversationTwoAgent
# from chat import VEConversationTwoAgent
import random
import time
import re
from PIL import Image
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ["OPENAI_API_KEY"] = "sb-fb43570969a6107c4dc146c41841f9c342f9c94befc8fd29"
#os.environ["OPENAI_API_KEY"] = "sb-21a8b17645c06e2fc402f20a5ad2a833af150f8f86125a73"
#os.environ["OPENAI_API_KEY"] = "sb-719c6650f5e094abe3a9ad901640b55b847f66e6684b6eee"
os.environ["OPENAI_API_KEY"] = "sk-whqQs9OWYxRLmNYo6e1fC0D8Ce00481196F2150eB0579417"
os.environ["OPENAI_BASE_URL"] = 'https://chatapi.onechat.fun/v1/'

from vlmeval.utils import track_progress_rich

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
            response = client.chat.completions.create(model="claude-3-opus-20240229", messages = chatgpt_messages,max_tokens=512)
            #print("chatgpt_messages",chatgpt_messages)
            reply = response.choices[0].message.content
            print("reply", reply)
            # total_tokens = response.usage.total_tokens
            success = True
            return reply, 0
        except Exception as e:
            print('[Worker] an exception occured: %s (%s). retrying in 3 minutes.' % (type(e), str(e)))
            # time.sleep(30)
            pass
def parse_final_answer(gpt_response):
    # ===== Parse the paragraph starting with analysis. =====
    try:
        analysis_result = re.search('Response:(.*)', gpt_response)
        analysis_string = analysis_result.group(1).strip()
        return analysis_string
    except Exception as e:
        print("Can not parse analysis")
        return None


def call_chat(data_id, dataset, save_path='work_dirs/claude/1'):
        image_id, image_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption, first_turn_category, second_turn_category, third_turn_category, first_turn_human_answer, second_turn_human_answer, third_turn_human_answer, third_turn_demands  = dataset.fetch_data(data_id)
        # image_id, image_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption  = dataset.fetch_data(data_id)
        #image_split = image_path.split("/")
        #image_name = image_split[-1]
        #image_name_split = image_name.split(".")
        result_path = os.path.join(save_path, '{}.yaml'.format(image_id))
        if os.path.isfile(result_path):
            return

        base64_image = encode_image_to_base64(Image.open(image_path),768)

        human = "user"
        model_name = "assistant"
        history_chat = []
        #prompt_fp = "./prompts/gpt4-response.txt"
        system_prompt = "Based on the image, please answer the questions.\n"
        #system_prompt +="Image Caption: " + instruction_conditioned_caption
        history_chat.append(["system", system_prompt])

        gpt4v_prompt = "Question: "
        first_turn_content = [{"type":"text","text":gpt4v_prompt+first_turn_instruction},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{base64_image}","image_detail": "low"}},]
        history_chat.append([human, first_turn_content])
        history_chat.append([model_name, None])
        history_chat[-1][1] = first_turn_human_answer

        second_turn_content = [{"type":"text","text":gpt4v_prompt+second_turn_instruction},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{base64_image}","image_detail": "low"}},]
        history_chat.append([human, second_turn_content])
        history_chat.append([model_name, None])
        second_turn_answer, second_token = call_gpt(history_chat)
        history_chat[-1][1] = second_turn_answer        

        third_turn_content = [{"type":"text","text":gpt4v_prompt+third_turn_instruction},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{base64_image}","image_detail": "low"}},]
        history_chat.append([human, third_turn_content])
        history_chat.append([model_name, None])
        third_turn_answer, third_token = call_gpt(history_chat)
        history_chat[-1][1] = third_turn_answer        


        results = {}
        results['image_path'] = image_path
        results['instruction_conditioned_caption'] = instruction_conditioned_caption
        results['first_turn_instruction'] = first_turn_instruction
        results['first_turn_human_answer'] = first_turn_human_answer
        results['first_turn_category'] = first_turn_category


        results['second_turn_instruction'] = second_turn_instruction
        results['second_turn_answer'] = second_turn_answer
        results['second_turn_human_answer'] = second_turn_human_answer
        results['second_turn_category'] = second_turn_category        

        results['third_turn_instruction'] = third_turn_instruction
        results['third_turn_answer'] = third_turn_answer
        results['third_turn_human_answer'] = second_turn_human_answer
        results['third_turn_category'] = third_turn_category     
        results['third_turn_demands'] = third_turn_demands   

        if save_path:
            with open(result_path, 'w') as f:
                yaml.dump(results, f)


def GPT(dataset, data_ids,  save_path='/mnt/lustre/liushuo/VLMEvalKit/work_dirs/claude/1'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    job_list = [{"data_id": data_id, "dataset": dataset} for data_id in data_ids]
    track_progress_rich(call_chat, job_list, nproc=10, chunksize=10)


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

    # load the dataset
    dataset = DatasetSampler('/mnt/lustre/liushuo/VLMEvalKit/mte')
    print('Finish loading data')
    question_model = args.model

    # preparing the folder to save results
    save_path = args.save_root
    #if not os.path.exists(save_path):
        #os.makedirs(os.path.join(save_path, 'result'))


    # start Conversation
    GPT(dataset, dataset.ids)
    

if __name__ == '__main__':
    args = parse()
    main(args)