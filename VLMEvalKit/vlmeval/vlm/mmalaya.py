import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os.path as osp
from vlmeval.smp import isimg
import re
from PIL import Image

from .mm_utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .mm_utils import conv_templates, KeywordsStoppingCriteria


class MMAlaya:

    INSTALL_REQ = False

    def __init__(self, model_path='DataCanvas/MMAlaya', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
        # need initialize tokenizer
        model.initialize_tokenizer(self.tokenizer)
        self.model = model.cuda()
        
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate(self, image_path, prompt, dataset=None):
        # read image
        image = Image.open(image_path).convert("RGB")
        # tokenize prompt, and proprecess image
        input_ids, image_tensor, stopping_criteria = self.model.prepare_for_inference(
            prompt, 
            self.tokenizer, 
            image,
            return_tensors='pt')
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids.cuda(),
                images=image_tensor.cuda(),
                do_sample=False,
                max_new_tokens=1024,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
        return response
    
    def prepare_for_inference_multi_turn(
        self, 
        inputs,
        history,
        tokenizer, 
        image,
        image_token_index=IMAGE_TOKEN_INDEX, 
        return_tensors=None
        ):
        # 加载对话模板
        conv = conv_templates["mmalaya_llama"].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inputs[0]

        conv.append_message(conv.roles[0], inp)
        for i, h in enumerate(history):
            conv.append_message(conv.roles[1], h)
            conv.append_message(conv.roles[0], inputs[i + 1])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        # 加载generate stop条件
        stopping_criteria = KeywordsStoppingCriteria(
            [conv.sep2], 
            tokenizer, 
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            )
        # 加载图像
        image_tensor = self.model.get_vision_tower().image_processor(
            image, return_tensors='pt')['pixel_values'].half()

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0), image_tensor, stopping_criteria
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        
        return input_ids, image_tensor, stopping_criteria
    
    def multi_turn_generate(self, inputs, history):
        print(history)
        image_path = inputs["image_path"]
        # read image
        image = Image.open(image_path).convert("RGB")

        if len(history) == 0:
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])
            # turn 1
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response1 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
            
            input_list.append(inputs['input2'])
            history_list.append(response1)

            # turn 2
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response2 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response3 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()

        elif len(history) == 1:
            response1 = history[0]
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])

            input_list.append(inputs['input2'])
            history_list.append(response1[: 1000])

            # turn 2
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response2 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response3 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
            

        else:
            response1 = history[0]
            response2 = history[1]
            input_list = []
            history_list = []
            input_list.append(inputs["input1"])

            input_list.append(inputs['input2'])
            history_list.append(response1[:1000])
            
            input_list.append(inputs['input3'])
            history_list.append(response2)

            # turn 3
            input_ids, image_tensor, stopping_criteria = self.prepare_for_inference_multi_turn(
                input_list,
                history_list,
                self.tokenizer, 
                image,
                return_tensors='pt')
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response3 = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0].strip()
        
        return response1, response2, response3


if __name__ == "__main__":
    model = MMAlaya()
    response = model.generate(
        image_path='./assets/apple.jpg',
        prompt='请详细描述一下这张图片。',
        )
    print(response)

"""
export PYTHONPATH=$PYTHONPATH:/tmp/VLMEvalKit
CUDA_VISIBLE_DEVICES=0 python vlmeval/vlm/mmalaya.py
"""
