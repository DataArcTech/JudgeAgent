# JudgeAgent
This is the source code of the paper "JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer".



## How to Run

### 1. Download Datasets

In our experiments, we use [MedQA](https://github.com/jind11/MedQA), [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/), and [QuALITY](https://github.com/nyu-mll/quality).

Please download the dataset you use and put the dataset folder under `./data`

### 2. Process Data

Run following code to process the data into formats JudgeAgent needs:

```shell
python process_data.py --data [MedQA | MultiHopRAG | QuALITY]
```

- `--data`  is the name of the dataset folder.

After running this code, you can see a new folder under the project's directory: `./processed_data/[data_name]`, and processed chunks`./corpus_chunks.jsonl` and questions `./questions.json` in this directory.

### 3.  Preprocess

JudgeAgent constructs context graphs, extracts entities of questions, and obtains embeddings before the formal evaluation process.

Before you run these codes, you **must** modify the `MODEL_PARAMS` in `./JudgeAgent/llm_params.py` by adding your own LLM information, including API_KEY, base_url, model_name, etc.

Please run the code in the following steps.

#### step1. construct graphs and extract entities

```shell
python construct_graph.py --data [MedQA | ...] --model gpt
python get_question_entity.py --ta [MedQA | ...] --model gpt
```

- `--data`  is the name of the dataset folder.
- `--model` is the LLM used to extract entities from text

You can run these two codes at the same time.

#### step2. obtain embeddings

```shell
python get_embeddings.py --data [MedQA | ...] --model qwen3 --dim 1024
```

- `--model` is the text embedding model. We use the embedding service of Qwen to get the embeddings in our experiments.
- `--dim` is the dimension of embeddings

#### step3. match the extracted entities with the entities on graph

```shell
python get_similar_entity_on_graph.py --data [MedQA | ...]
```

This code match the extracted entities with the entities on context graph by the similarity calculated from the embeddings obtained in step2.

### 4. Evaluation

For efficiency, JudgeAgent synthetises questions before evaluating the target LLM.

Please run the codes in the following steps.

#### step1. synthetise questions

```shell
python synthetise_questions.py \
--data [MedQA |...] \
--model gpt \
--bs 3 \
--sample_hop 2 \
--max_extend_round 3 \
--no_graph \ # Optional
--fix_difficulty ["" | easy | medium | hard]
```

- `--bs` is the batch size in Benchmark Grading stage
- `--sample_hop` is the number of hop in sampling knowledge paths on context graph
- `--max_extend_round` is the max number of extension round in Interactive Extension stage

The following two arguments are for ablation study.

- `--no_graph`: When add this argument, JudgeAgent will synthetise questions with the texts randomly sampled from `corpus_chunks.jsonl`  instead of paths sampled on context graph
- `--fix_difficulty`: When this argument is not empty string "", JudgeAgent will synthetise questions under the `fix_difficuly`, refering to the ablation study "*w/o difficulty-adaptive*". When this argument is "", JudgeAgent will synthetise questions for each difficulty level before evaluation, ensuring the choice in dynamic evaluaiton.

#### step2. evaluate the target and get feedback

```shell
python evaluate_target.py \
--data [MedQA |...] \
--target XXX \
--model gpt \
--bs 3 \
--sample_hop 2 \
--max_extend_round 3 \
# following are ablation settings
--eval_all_rounds \ # Optional
--no_graph \ # Optional
--no_extension \ # Optional
--fix_difficulty ["" | easy | medium | hard]
```

- `--target` is the target LLM to be evaluated, and `--model` is the LLM used in JudgeAgent for evaluation.

The following arguments are for ablation study about extension stage.

- `--eval_all_rounds`: When add this argument, JudgeAgent will evalaute the target's performance until current rounds, provide suggetions, and validate the effectiveness after each round of extension.
- `--no_extension`: When add this argument, JudgeAgent will skip Interactive Extension stage, and only evaluate the target's performance on base questions.



