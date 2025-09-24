import os
from typing import List, Dict

from JudgeAgent import *




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--bs", type=int, default=3, help="the batch size in Benchmark Grading")
    parser.add_argument("--sample_hop", type=int, default=2, help="the max hop in sampling.")
    parser.add_argument("--max_extend_round", type=int, default=3, help="the max rounds for expanding")
    # ablation settings
    parser.add_argument("--eval_all_rounds", action="store_true", help="evaluate after all extension rounds")
    parser.add_argument("--no_graph", action="store_true", help="the ablation setting of no context graph")
    parser.add_argument("--no_extension", action="store_true", help="the ablation setting of no extension")
    parser.add_argument("--fix_difficulty", type=str, default="", choices=["", "easy", "medium", "hard"], help="the ablation setting of no difficulty-adaptive")
    args = parser.parse_args()

    # basic parameters
    data_name: str = args.data
    batch_size: int = args.bs
    sample_hop: int = args.sample_hop
    max_extend_rounds: int = args.max_extend_rounds
    eval_all_rounds: bool = args.eval_all_rounds
    no_graph: bool = args.no_graph
    no_extension: bool = args.no_extension
    fix_difficulty: str = args.fix_difficulty

    if data_name in ["MedQA"]:
        question_type = QTYPE.KMC
    elif data_name in ["MultiHopRAG"]:
        question_type = QTYPE.KQA
    elif data_name in ["QuALITY"]:
        question_type = QTYPE.RMC
    else:
        raise ValueError(f"Dataset {data_name} is not supported now.")

    # file paths
    data_dir = os.path.join("processed_data", data_name)
    chunk_path = os.path.join(data_dir, "corpus_chunks.jsonl")
    questions_path = os.path.join(data_dir, "questions_for_eval.json")

    postfix = ""
    if no_graph:
        postfix += "_ng"
    else:
        postfix += f"_hop{sample_hop}"
    if fix_difficulty:
        postfix += f"_fd-{fix_difficulty}"
    syn_questions_path = os.path.join(data_dir, f"syn_questions_{data_name}_rnd{max_extend_rounds}_bs{batch_size}{postfix}.json")

    if no_extension:
        postfix += "_ne"
    statistic_path = os.path.join(data_dir, f"statistic_{data_name}_rnd{max_extend_rounds}_bs{batch_size}{postfix}.json")
    eval_result_path = os.path.join(data_dir, f"eval_results_{data_name}_rnd{max_extend_rounds}_bs{batch_size}{postfix}.json")

    # load frameworks
    target = LLMClient(MODEL_PARAMS[args.target])
    client = LLMClient(MODEL_PARAMS[args.model])

    graph = Graph(data_dir)
    embedding_client = EmbeddingClient(
        save_dir=os.path.join(data_dir, "embeddings"), 
        params=MODEL_PARAMS[args.model + "-embedding"]
    )
    sampler = GraphPathSampler(
        input_data=chunk_path, 
        save_dir=data_dir, 
        embedding_client=embedding_client, 
        graph=graph, 
        sample_hop=sample_hop
    )

    judge_agent = JudgeAgent(
        llm_client=client, 
        sampler=sampler, 
        save_dir=data_dir, 
        result_path=eval_result_path, 
        max_extend_rounds=3, 
        benchmarking_batch_size=batch_size
    )

    # load datas
    questions_for_eval = load_json(questions_path)
    
    # stage1. Benchmark Grading
    judge_agent.stage_benchmark_grading(
        target=target,
        questions=questions_for_eval, 
        question_type=question_type, 
        load_record=True
    )

    # stage2. Interactive Extension
    if not no_extension:
        judge_agent.stage_interactive_extension(
            target=target, 
            question_type=question_type, 
            syn_question_path=syn_questions_path, 
            load_record=True, 
            fix_difficulty=fix_difficulty
        )
    
    # stage3. Evaluation and Feedback
    judge_agent.stage_evaluation_feedback(
        target=target, 
        question_type=question_type, 
        statistic_path=statistic_path, 
        eval_after_all_extend_rounds=eval_all_rounds, 
        eval_no_extension=no_extension, 
        load_record=True
    )