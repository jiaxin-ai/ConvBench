import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.evaluate import COCO_eval, YOrN_eval, MMVet_eval, multiple_choice_eval, VQAEval, MathVista_eval, LLaVABench_eval
from vlmeval.config import supported_VLM
from vlmeval.utils import dataset_URLs, DATASET_TYPE, abbr2full
from dataset import DatasetSampler
import yaml

import os

os.environ["GOOGLE_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["DASHSCOPE_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mte')
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--work-dir", type=str, default='./work_dirs', help="select the output directory")
    parser.add_argument("--mode", type=str, default='all', choices=['all', 'infer'])
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument("--ignore", action='store_true', help="Ignore failed indices. ")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--prefetch", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    return args



# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset_name, index_set, api_nproc=4):
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    is_api = getattr(model, 'is_api', False)
    assert is_api

    lt, indices = len(data), list(data['index'])
    structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]
    
    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        res = {k: v for k, v in res.items() if FAIL_MSG not in v}
    
    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]
    
    gen_func = None
    if listinstr(['MMMU'], dataset_name):
        assert hasattr(model, 'interleave_generate')
        gen_func = model.interleave_generate
        structs = [dict(ti_list=split_MMMU(struct), dataset=dataset_name) for struct in structs]
    elif listinstr(['CORE_MM'], dataset_name):
        assert hasattr(model, 'multi_generate')
        gen_func = model.multi_generate
        structs = [dict(image_paths=struct['image'], prompt=struct['text'], dataset=dataset_name) for struct in structs]
    else:
        gen_func = model.generate
        structs = [dict(image_path=struct['image'], prompt=struct['text'], dataset=dataset_name) for struct in structs]

    inference_results = track_progress_rich(
        gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)
    
    res = load(out_file)
    for idx, text in zip(indices, inference_results):
        assert (res[idx] == text if idx in res else True)
        res[idx] = text
    return res


def infer_data(model_name, work_dir, dataset_name, out_file, verbose=False, api_nproc=4):
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    data_sampler = DatasetSampler("../")

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, data_ids = len(data_sampler.ids), data_sampler.ids
        
        for data_id in tqdm(data_ids):
            img_id, img_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption, first_turn_category, second_turn_category, third_turn_category, first_turn_answer, second_turn_answer, third_turn_answer, third_turn_demands  = data_sampler.fetch_data(data_id)

            base64_image = None
            # 3次不同的实验
            for i in range(3):
                result_path = os.path.join(f"{work_dir}/{i}", '{}.yaml'.format(img_id))
                if os.path.isfile(result_path):
                    continue
                history_dict = list()
                if i == 0:
                    # 全部重新推理p
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 2:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            print(img_path)
                        retry += 1
                    
                elif i == 1:
                    history_dict = [first_turn_answer]
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 2:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            print(img_path)
                        retry += 1

                elif i == 2:
                    history_dict = [first_turn_answer, second_turn_answer]
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 2:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            print(img_path)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                        retry += 1
                
                if retry == 2:
                    continue

                results = {}
                results['image_path'] = img_path
                results['first_turn_instruction'] = first_turn_instruction
                results['first_turn_answer'] = r1
                results['second_turn_instruction'] = second_turn_instruction
                results['second_turn_answer'] = r2
                results['third_turn_instruction'] = third_turn_instruction
                results['third_turn_answer'] = r3

                if not os.path.exists(f"{work_dir}/{i}"):
                    os.makedirs(f"{work_dir}/{i}")

                if True:
                    with open(result_path, 'w') as f:
                        yaml.dump(results, f)

        return model_name
    else:
        lt, data_ids = len(data_sampler.ids), data_sampler.ids
        
        for data_id in tqdm(data_ids):
            img_id, img_path, first_turn_instruction, second_turn_instruction, third_turn_instruction, instruction_conditioned_caption, first_turn_category, second_turn_category, third_turn_category, first_turn_answer, second_turn_answer, third_turn_answer, third_turn_demands  = data_sampler.fetch_data(data_id)

            base64_image = None
            # 3次不同的实验
            for i in range(3):
                result_path = os.path.join(f"{work_dir}/{i}", '{}.yaml'.format(img_id))
                print("result_path",result_path,flush=True)
                if os.path.isfile(result_path):
                    continue
                history_dict = list()
                if i == 0:
                    # 全部重新推理
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 10:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                        retry += 1
                    
                elif i == 1:
                    history_dict = [first_turn_answer]
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 10:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                        retry += 1

                elif i == 2:
                    history_dict = [first_turn_answer, second_turn_answer]
                    input = {"input1": first_turn_instruction, 
                             "input2": second_turn_instruction, 
                             "input3": third_turn_instruction, 
                             "image_path": img_path, "image64": base64_image}
                    retry = 0

                    while retry < 10:
                        try:
                            r1, r2, r3 = model.multi_turn_generate(input, history=history_dict)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                        retry += 1
                
                if retry == 10:
                    continue

                results = {}
                results['image_path'] = img_path
                #results['instruction_conditioned_caption'] = instruction_conditioned_caption
                results['first_turn_instruction'] = first_turn_instruction
                results['first_turn_answer'] = r1
                results['second_turn_instruction'] = second_turn_instruction
                results['second_turn_answer'] = r2
                results['third_turn_instruction'] = third_turn_instruction
                results['third_turn_answer'] = r3

                if not os.path.exists(f"{work_dir}/{i}"):
                    os.makedirs(f"{work_dir}/{i}")

                if True:
                    with open(result_path, 'w') as f:
                        yaml.dump(results, f)

        return model_name


def infer_data_job(model, work_dir, model_name, dataset_name, verbose=False, api_nproc=4, ignore_failed=False):
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
    rank, world_size = get_rank_and_world_size()   
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    if not osp.exists(result_file):
        model = infer_data(model, work_dir=work_dir, dataset_name=dataset_name, out_file=out_file, verbose=verbose)
        return model

def main():
    logger = get_logger('RUN')

    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    if args.debug:
        # ****************************************************************************
        # ****************************************************************************
        import debugpy;import socket
        address = socket.gethostbyname(socket.gethostname())
        port = int(os.environ['SLURM_PROCID']) + 5683
        print(f'===========================address is: {address}=============================', flush=True)
        print(f'===========================port is: {port}=============================', flush=True)
        debugpy.listen(('0.0.0.0', port))
        debugpy.wait_for_client()
        # ****************************************************************************
        # ****************************************************************************
    
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    for _, model_name in enumerate(args.model):
        model = None

        pred_root = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root, exist_ok=True)
        print("pred_root", pred_root, flush=True)

        for i, dataset_name in enumerate(args.data):
    
            if model is None:
                model = model_name # which is only a name
            
            # model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
            pass

            model = infer_data_job(model, work_dir=pred_root, model_name=model_name, dataset_name=dataset_name, verbose=args.verbose, api_nproc=args.nproc, ignore_failed=args.ignore)
            
if __name__ == '__main__':
    main()
