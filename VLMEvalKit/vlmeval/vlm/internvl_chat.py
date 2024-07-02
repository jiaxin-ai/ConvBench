import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, CLIPImageProcessor
import warnings
import os.path as osp
from vlmeval.smp import isimg
import re
from PIL import Image
from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt
import pandas as pd
import string


class InternVLChat(CustomPrompt):

    INSTALL_REQ = False

    def __init__(self, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-1', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)                                                  
        self.image_size = self.model.config.vision_config.image_size

        if 'V1-1' in model_path:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=5, max_length=4096)         
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1, max_length=4096)            
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
    
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question
    
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question
    
        if len(options):
            prompt += "\n请直接回答选项字母。" if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."
    
        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                   question=prompt, generation_config=self.kwargs)
        return response
    
    def multi_turn_generate(self, inputs, history):
        image_path = inputs["image_path"]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        
        if len(history) == 0:
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])
            # turn 1
            response1 = self.chat(self.tokenizer, pixel_values=pixel_values,
                            input_list=input_list, history_list=history_list, generation_config=self.kwargs)
            
            input_list.append(inputs['input2'])
            history_list.append(response1)

            # turn 2
            response2 = self.chat(self.tokenizer, pixel_values=pixel_values,
                input_list=input_list, history_list=history_list, generation_config=self.kwargs)
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            response3 = self.chat(self.tokenizer, pixel_values=pixel_values,
                input_list=input_list, history_list=history_list, generation_config=self.kwargs)

        elif len(history) == 1:
            response1 = history[0]
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])

            input_list.append(inputs['input2'])
            history_list.append(response1)

            # turn 2
            response2 = self.chat(self.tokenizer, pixel_values=pixel_values,
                input_list=input_list, history_list=history_list, generation_config=self.kwargs)
            # truncate input_ids in generate_ids and then decode to text
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            response3 = self.chat(self.tokenizer, pixel_values=pixel_values,
                input_list=input_list, history_list=history_list, generation_config=self.kwargs)

        else:
            response1 = history[0]
            response2 = history[1]
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])

            input_list.append(inputs['input2'])
            history_list.append(response1)
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            response3 = self.chat(self.tokenizer, pixel_values=pixel_values,
                input_list=input_list, history_list=history_list, generation_config=self.kwargs)
        
        return response1, response2, response3
    
    def chat(self, tokenizer, pixel_values, input_list, history_list, generation_config,
            IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        from .conversation import get_conv_template
        template = get_conv_template(self.model.template)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token + IMG_END_TOKEN

        template.append_message(template.roles[0], image_tokens + '\n' + input_list[0])
        for i, h in enumerate(history_list):
            template.append_message(template.roles[1], h)
            template.append_message(template.roles[0], input_list[i + 1])
        # template.append_message(template.roles[0], image_tokens + '\n' + question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()

        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        query_to_print = query.replace(image_tokens, '<image>')
        print(query_to_print, response)
        return response