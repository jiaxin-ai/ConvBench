To create the enviroment.

```shell
pip install requirements.txt
```

To generate the multi-turn responses for each sample in ConvBench via each Large Vision-Language Model.

```shell
python ./Close-Source-Model/gpt4v_answer.py
python ./Close-Source-Model/reka_answer.py
python ./Close-Source-Model/claude_answer.py
python ./run_multi_turn.py --data "mte" --model "blip2-flan-t5-xxl"
python ./run_multi_turn.py --data "mte" --model "blip2-flan-t5-xxl"
python ./run_multi_turn.py--data "mte" --model "InternVL-Chat-V1-2"
python ./run_multi_turn.py --data "mte" --model "llama_adapter_v2_multimodal7b"
python ./run_multi_turn.py --data "mte" --model "llava_v1.5_7b"
python ./run_multi_turn.py --data "mte" --model "llava_v1.5_13b"
python ./run_multi_turn.py --data "mte" --model "MMAlaya"
python ./run_multi_turn.py --data "mte" --model "monkey-chat"
python ./run_multi_turn.py --data "mte" --model "mPLUG-Owl2"
python ./run_multi_turn.py --data "mte" --model "qwen_chat"
python ./run_multi_turn.py --data "mte" --model "sharegpt4v_7b"
python ./run_multi_turn.py --data "mte" --model "sharegpt4v_13b"
python ./run_multi_turn.py --data "mte" --model "XComposer"
python ./run_multi_turn.py --data "mte" --model "XComposer2"
```

