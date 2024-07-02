import warnings
from ..smp import *
from PIL import Image
import torch

# pip install accelerate
import torch
from PIL import Image

import cv2
import torch
import sys
from PIL import Image
sys.path.append('/mnt/lustre/liushuo/VLMEvalKit/model_zoo/LLaMA-Adapter/llama_adapter_v2_multimodal7b')
import llama


class llama_adapter:
    INSTALL_REQ = False

    def __init__(self, model_path="/mnt/lustre/liushuo/VLMEvalKit/checkpoints/llama_model_weights", **kwargs):

        assert model_path is not None
        self.model_path = model_path

        # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
        self.model, self.preprocess = llama.load("LORA-BIAS-7B-v21", model_path, llama_type="7B", device="cuda")
        self.model.eval()

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        prompt_input = "Below is an instruction that describes a task, paired with an input that provides further context. "\
            "Write a response that appropriately completes the request.\n\n"  \
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

        prompt = llama.format_prompt(prompt)
        img = Image.fromarray(cv2.imread(image_path))
        img = self.preprocess(img).unsqueeze(0).to("cuda")

        output = self.model.generate(img, [prompt])[0]

        return output
    
    def multi_turn_generate(self, inputs, history):
        
        image_path = inputs["image_path"]
        img = Image.fromarray(cv2.imread(image_path))
        img = self.preprocess(img).unsqueeze(0).to("cuda")
        
        if len(history) == 0:
            # turn 1
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response1 = self.model.generate(img, [prompt])[0]

            # turn 2
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response: {response1}\n\n###Instruction:\n{inputs['input2']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response2 = self.model.generate(img, [prompt])[0]

            # turn 3
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response: {response1}\n\n###Instruction:\n{inputs['input2']}\n\n### Response:{response2}\n\n###Instruction:\n{inputs['input3']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response3 = self.model.generate(img, [prompt])[0]

        elif len(history) == 1:
            response1 = history[0]
            # turn 2
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response: {response1}\n\n###Instruction:\n{inputs['input2']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response2 = self.model.generate(img, [prompt])[0]

            # turn 3
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response: {response1}\n\n###Instruction:\n{inputs['input2']}\n\n### Response:{response2}\n\n###Instruction:\n{inputs['input3']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response3 = self.model.generate(img, [prompt])[0]

        else:
            response1 = history[0]
            response2 = history[1]
            # turn 3
            prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. " 
                "Write a response that appropriately completes the request.\n\n"  
                f"### Instruction:\n{inputs['input1']}\n\n### Response: {response1}\n\n###Instruction:\n{inputs['input2']}\n\n### Response:{response2}\n\n###Instruction:\n{inputs['input3']}\n\n### Response:")
            prompt = llama.format_prompt(prompt)
            img = Image.fromarray(cv2.imread(image_path))
            img = self.preprocess(img).unsqueeze(0).to("cuda")
            response3 = self.model.generate(img, [prompt])[0]
        
        return response1, response2, response3
    