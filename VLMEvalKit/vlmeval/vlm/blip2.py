import os.path as osp
import warnings
from ..smp import *
from PIL import Image
import torch

# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2:
    INSTALL_REQ = False

    def __init__(self, model_path="Salesforce/blip2-flan-t5-xxl", **kwargs):
        assert model_path is not None
        self.model_path = model_path

        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        raw_image = Image.open(image_path).convert('RGB')

        inputs = self.processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.float16)

        out = self.model.generate(**inputs,
                                do_sample=False,
                                num_beams=1,
                                max_length=1024,
                                min_length=1,
                                top_p=0.9,
                                repetition_penalty=1.5,
                                length_penalty=1.0,
                                temperature=1,)
        output = self.processor.decode(out[0], skip_special_tokens=True)
        
        return output
    
    def multi_turn_generate(self, inputs, history):
        # return response
        image_path = inputs['image_path']
        raw_image = Image.open(image_path).convert('RGB')
        if len(history) == 0:
            # turn1
            cur_prompt = f"Question: {inputs['input1']} Answer: "
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response1 = self.processor.decode(out[0], skip_special_tokens=True)

            # turn2
            cur_prompt = f"Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:"
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response2 = self.processor.decode(out[0], skip_special_tokens=True)

            # turn3
            cur_prompt = f"Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response3 = self.processor.decode(out[0], skip_special_tokens=True)

        elif len(history) == 1:
            response1 = history[0]
            # turn2
            cur_prompt = f"Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:"
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response2 = self.processor.decode(out[0], skip_special_tokens=True)

            # turn3
            cur_prompt = f"Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response3 = self.processor.decode(out[0], skip_special_tokens=True)

        else:
            response1 = history[0]
            response2 = history[1]
            # turn3
            cur_prompt = f"Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            inputs_ = self.processor(raw_image, cur_prompt, return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(**inputs_,
                                    do_sample=False,
                                    num_beams=1,
                                    max_length=1024,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=0.2,
                                    temperature=1,)
            response3 = self.processor.decode(out[0], skip_special_tokens=True)
        return response1, response2, response3
    
