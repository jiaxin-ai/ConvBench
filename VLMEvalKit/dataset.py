import os
from tqdm import tqdm
import json
from copy import deepcopy
import yaml
import pandas as pd


class DatasetSampler():
    def __init__(self, dataset_root):
        self.ann = {}
        dataset_anno_dir = os.path.join(dataset_root, 'ConvBench.xlsx')
        df = pd.read_excel(dataset_anno_dir)
        for i in range(len(df)):
            cur_id = df['ID'][i]
            cur_image_id = df['image_id'][i]
            cur_first_turn_instruction = df['The_first_turn_instruction'][i]
            cur_second_turn_instruction =  df['The_second_turn_instruction'][i]
            cur_third_turn_instruction = df['The_third_turn_instruction'][i]
            cur_instruction_conditioned_caption = df['instruction-conditioned-caption'][i]
            cur_first_turn_category = df['First_turn_instruction_category'][i]
            cur_first_turn_answer = df['first_turn_answer'][i]
            cur_second_turn_category = df['Second_turn_instruction_category'][i]
            cur_second_turn_answer = df['second_turn_answer'][i]
            cur_third_turn_category = df['Third_turn_instruction_category'][i]
            cur_third_turn_answer = df['third_turn_answer'][i]
            cur_third_turn_demands = df['third_turn_demands'][i]



            img_path = os.path.join(dataset_root, 'visit_bench_images', cur_image_id)

            self.ann[i] = { 'ID': cur_id,
                            'img_path': img_path,
                            'first_turn_instruction': cur_first_turn_instruction, 
                            'second_turn_instruction': cur_second_turn_instruction,
                            'third_turn_instruction': cur_third_turn_instruction,
                            'instruction_conditioned_caption': cur_instruction_conditioned_caption,
                            'first_turn_category': cur_first_turn_category,
                            'second_turn_category': cur_second_turn_category,
                            'third_turn_category': cur_third_turn_category,
                            'first_turn_answer': cur_first_turn_answer,
                            'second_turn_answer': cur_second_turn_answer,
                            'third_turn_answer': cur_third_turn_answer,
                            'third_turn_demands' : cur_third_turn_demands,
                            }

        self._ids = list(self.ann.keys())
        
    @property
    def ids(self):
        return deepcopy(self._ids)

    def fetch_data(self, id):
        ann = self.ann[id]
        img_id = ann['ID']
        img_path = ann['img_path']
        first_turn_instruction = ann['first_turn_instruction']
        second_turn_instruction = ann['second_turn_instruction']
        third_turn_instruction = ann['third_turn_instruction']
        instruction_conditioned_caption = ann['instruction_conditioned_caption']
        first_turn_category = ann['first_turn_category']
        second_turn_category = ann['second_turn_category']
        third_turn_category = ann['third_turn_category']
        first_turn_answer = ann['first_turn_answer']
        second_turn_answer = ann['second_turn_answer']        
        third_turn_answer = ann['third_turn_answer']
        third_turn_demands = ann['third_turn_demands']

        return img_id, img_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption, first_turn_category, second_turn_category, third_turn_category, first_turn_answer, second_turn_answer, third_turn_answer, third_turn_demands  

