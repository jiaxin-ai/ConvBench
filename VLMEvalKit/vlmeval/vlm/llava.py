import torch
from PIL import Image
from abc import abstractproperty
import os, sys
import os.path as osp
from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt

class LLaVA(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self, 
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs): 
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn("Please install llava before using LLaVA")
            sys.exit(-1)
            
        assert osp.exists(model_pth) or splitlen(model_pth) == 2
        
        if model_pth == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_pth == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth, 
                model_base=None, 
                model_name=model_name, 
                device='cpu', 
                device_map='cpu'
            )
        except:
            if 'ShareGPT4V' in model_pth:
                import llava
                warnings.warn(
                    f'Please manually remove the encoder type check in {llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)
        
        self.model = self.model.cuda()
        self.conv_mode =  'llava_v1'

        kwargs_default = dict(do_sample=True, temperature=0.2, max_new_tokens=1024, top_p=None, num_beams=1)
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
            prompt += "\n请直接回答选项字母。" if cn_string(prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        return output
    
    def multi_turn_generate(self, inputs, history):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle

        image_path = inputs["image_path"]
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        # self.model.chat(text=inputs['input1'], image=inputs["image_path"], history=None)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inputs["input1"]
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inputs["input1"]
        if len(history) == 0:
            # turn 1
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response1 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]

            # turn 2
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], response1)
            conv.append_message(conv.roles[0], inputs['input2'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response2 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]

            # turn 3
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], response1)
            conv.append_message(conv.roles[0], inputs['input2'])
            conv.append_message(conv.roles[1], response2)
            conv.append_message(conv.roles[0], inputs['input3'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response3 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]

        elif len(history) == 1:
            response1 = history[0]
            # turn 2
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], response1)
            conv.append_message(conv.roles[0], inputs['input2'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response2 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]

            # turn 3
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], response1)
            conv.append_message(conv.roles[0], inputs['input2'])
            conv.append_message(conv.roles[1], response2)
            conv.append_message(conv.roles[0], inputs['input3'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response3 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]

        else:
            response1 = history[0]
            response2 = history[1]
            # turn 3
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], response1)
            conv.append_message(conv.roles[0], inputs['input2'])
            conv.append_message(conv.roles[1], response2)
            conv.append_message(conv.roles[0], inputs['input3'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            response3 = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        
        return response1, response2, response3