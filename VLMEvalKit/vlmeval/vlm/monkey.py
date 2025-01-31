import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os.path as osp
from vlmeval.smp import isimg
from ..utils import DATASET_TYPE, CustomPrompt

class Monkey:

    INSTALL_REQ = False

    def __init__(self, model_path='echo840/Monkey', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
        self.model = model.cuda()
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_vanilla(self, image_path, prompt):
        cur_prompt = f'<img>{image_path}</img> {prompt} Answer: '
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response

    def generate_multichoice(self, image_path, prompt):
        cur_prompt = f'<img>{image_path}</img> \n {prompt} Answer: '
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=10,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response
    
    def generate(self, image_path, prompt, dataset=None):
        if dataset is None:
            return self.generate_vanilla(image_path, prompt)
        assert isinstance(dataset, str)
        if DATASET_TYPE(dataset) == 'multi-choice' or DATASET_TYPE(dataset) == 'Y/N' or dataset=="HallusionBench":
            return self.generate_multichoice(image_path, prompt)
        else:
            return self.generate_vanilla(image_path, prompt)
    


class MonkeyChat:

    INSTALL_REQ = False

    def __init__(self, model_path='echo840/Monkey-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
        self.model = model.cuda()
        self.kwargs = kwargs
        
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
    
    def generate_vanilla(self, image_path, prompt):
        cur_prompt = f'<img>{image_path}</img> {prompt} Answer: '
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response

    def generate_multichoice(self, image_path, prompt):
        cur_prompt = f'<img>{image_path}</img> \n {prompt} Answer: '
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=10,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response
    
    def generate(self, image_path, prompt, dataset=None):
        if dataset is None:
            return self.generate_vanilla(image_path, prompt)
        assert isinstance(dataset, str)
        if DATASET_TYPE(dataset) == 'multi-choice' or DATASET_TYPE(dataset) == 'Y/N' or dataset=="HallusionBench":
            return self.generate_multichoice(image_path, prompt)
        else:
            return self.generate_vanilla(image_path, prompt)
    
    def generate_vanilla(self, image_path, prompt):
        cur_prompt = f'<img>{image_path}</img> {prompt} Answer: '
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response

    def multi_turn_generate(self, inputs, history):
        # return response
        image_path = inputs['image_path']
        if len(history) == 0:
            # turn1
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']} Answer: "
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response1 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

            # turn2
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:"
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response2 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

            # turn3
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response3 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

        elif len(history) == 1:
            response1 = history[0]
            # turn2
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:"
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response2 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

            # turn3
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response3 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

        else:
            response1 = history[0]
            response2 = history[1]
            # turn3
            cur_prompt = f"<img>{image_path}</img> Question: {inputs['input1']}\nAnswer: {response1}\nQuestion: {inputs['input2']}\n Answer:{response2}\nQuestion: {inputs['input3']}\n Answer:"
            input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
            output_ids = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                )
            response3 = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return response1, response2, response3
