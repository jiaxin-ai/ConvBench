import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os.path as osp
from vlmeval.smp import isimg, listinstr
import re

class QwenVL:

    INSTALL_REQ = False

    def __init__(self, model_path='Qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)

        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        vl_list = [{'image': img} for img in image_paths] + [{'text': prompt}]
        query = self.tokenizer.from_list_format(vl_list)

        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    def interleave_generate(self, ti_list, dataset=None):
        vl_list = [{'image': s} if isimg(s) else {'text': s} for s in ti_list]
        query = self.tokenizer.from_list_format(vl_list)

        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(query)[1].split('<|endoftext|>')[0]
        return response
    
class QwenVLChat:

    INSTALL_REQ = False

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        vl_list = [{'image': img} for img in image_paths] + [{'text': prompt}]
        query = self.tokenizer.from_list_format(vl_list)   
         
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
    
    def interleave_generate(self, ti_list, dataset=None):
        vl_list = [{'image': s} if isimg(s) else {'text': s} for s in ti_list]
        query = self.tokenizer.from_list_format(vl_list)

        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
    
    def from_list_format(self, list_format):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}: '
                text += "<img>" + ele['image'] + "</img>"
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text
    
    def multi_turn_generate(self, inputs, history):
        # list_format = [{'image': image_path}, {'text': prompt}]
        # query = self.tokenizer.from_list_format(list_format)
        # response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        
        image_path = inputs["image_path"]
        
        if len(history) == 0:
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant:'})
            # turn 1
            query = self.tokenizer.from_list_format(list_format)
            response1, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)

            # turn 2
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant: {response1}\nUser: {inputs["input2"]}\nAssistant:'})
            query = self.tokenizer.from_list_format(list_format)
            response2, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)

            # turn 3
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant: {response1}\nUser: {inputs["input2"]}\nAssistant: {response2}\nUser: {inputs["input3"]}\nAssistant:'})
            query = self.tokenizer.from_list_format(list_format)
            response3, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)

        elif len(history) == 1:
            response1 = history[0]
            # turn 2
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant: {response1}\nUser: {inputs["input2"]}\nAssistant:'})
            query = self.tokenizer.from_list_format(list_format)
            response2, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)

            # turn 3
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant: {response1}\nUser: {inputs["input2"]}\nAssistant: {response2}\nUser: {inputs["input3"]}\nAssistant:'})
            query = self.tokenizer.from_list_format(list_format)
            response3, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)

        else:
            response1 = history[0]
            response2 = history[1]
            # turn 3
            list_format = [{'image': image_path}]
            list_format.append({"text": f'User: {inputs["input1"]}\nAssistant: {response1}\nUser: {inputs["input2"]}\nAssistant: {response2}\nUser: {inputs["input3"]}\nAssistant:'})
            query = self.tokenizer.from_list_format(list_format)
            response3, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        
        return response1, response2, response3
