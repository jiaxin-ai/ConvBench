## Evaluation

To calculate the scores, please prepare the model responses in yaml format, like this [example](../VLMEvalKit/work_dirs/InternVL-Chat-V1-2/0/1.yaml). Then you can place all yaml files in a work_dir folder and execute our script [convbencheval.py](convbencheval.py) to get the scores.

```shell
python convbencheval.py --vqa_model="InternVL-Chat-V1-2" 
```

To conduct the hierarchical ablation evaluation with perfect perception ability, execute our script [convbencheval_with_perfect_perception.py](convbencheval_with_perfect_perception.py) to get the scores.

```shell
python convbencheval_with_perfect_perception.py --vqa_model="InternVL-Chat-V1-2" 
```

To conduct the hierarchical ablation evaluation with perfect perception and reasoning abilities, execute our script [convbencheval_with_perfect_perception_and_reasoning.py](convbencheval_with_perfect_perception_and_reasoning.py) to get the scores.

```shell
python convbencheval_with_perfect_perception_and_reasoning.py --vqa_model="InternVL-Chat-V1-2" 
```

