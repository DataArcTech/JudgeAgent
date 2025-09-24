import os
import random
from typing import List, Dict, Any

from JudgeAgent import *





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--bs", type=int, default=3, help="the batch size in Benchmark Grading")
    parser.add_argument("--sample_hop", type=int, default=2, help="the max hop in sampling.")
    parser.add_argument("--max_extend_round", type=int, default=3, help="the max rounds for expanding")
    parser.add_argument("--no_graph", action="store_true", help="the ablation setting of no context graph")
    parser.add_argument("--fix_difficulty", type=str, default="", choices=["", "easy", "medium", "hard"], help="the ablation setting of no difficulty-adaptive")
    args = parser.parse_args()
    data_name: str = args.data
    batch_size: int = args.bs
    sample_hop: int = args.sample_hop
    max_extend_rounds: int = args.max_extend_rounds
    no_graph: bool = args.no_graph
    fix_difficulty: str = args.fix_difficulty

    if data_name in ["MedQA"]:
        question_type = QTYPE.KMC
    elif data_name in ["MultiHopRAG"]:
        question_type = QTYPE.KQA
    elif data_name in ["QuALITY"]:
        question_type = QTYPE.RMC
    else:
        raise ValueError(f"Dataset {data_name} is not supported now.")

    # load chunks
    data_dir = os.path.join("processed_data", data_name)
    chunk_path = os.path.join(data_dir, "corpus_chunks.jsonl")
    if no_graph:
        raw_chunks = load_jsonl(os.path.join(data_dir, "corpus_chunks.jsonl"))
        chunks: List[str] = []
        for data in raw_chunks:
            chunks.extend(data["chunks"])

    # load framework
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
        max_extend_rounds=3, 
        benchmarking_batch_size=batch_size
    )

    # load seed questions
    questions_for_eval = load_json(os.path.join(data_dir, "questions_for_eval.json"))
    if data_name in ["QuALITY"]:
        question_batches: List[List[Dict]] = [qdata["questions"] for qdata in questions_for_eval]
    else:
        question_batches: List[List[Dict]] = split_into_batches(questions_for_eval, batch_size)

    # load record
    save_postfix = ""
    if no_graph:
        save_postfix += "_ng"
    else:
        save_postfix += f"_hop{sample_hop}"
    if fix_difficulty:
        save_postfix += "_fd"
    save_path = os.path.join(data_dir, f"syn_questions_{data_name}_rnd{max_extend_rounds}_bs{batch_size}{save_postfix}.json")
    syn_questions: List[List[Dict[str, Dict]]] = load_json(save_path) if os.path.exists(save_path) else []
    

    from tqdm import tqdm
    index = len(syn_questions)
    with tqdm(total=len(question_batches), desc=f"syn_rnd{max_extend_rounds}_bs{batch_size}_{save_postfix}") as pbar:
        pbar.update(index)

        for batch in question_batches[index : ]:
            # get example for question generation
            if data_name in ["QuALITY"]:
                example = judge_agent.generate_example_text(batch, fix_difficulty)
            else:
                example: str = None

            questions: List[Dict[str, Dict]] = []
            if no_graph:
                # if no context graph, randomly sample texts from the original knowlege base
                for _ in max_extend_rounds:
                    text_fragments = random.sample(chunks, batch_size)
                    fragments = [f"Text Fragment {i+1}: {text}" for i, text in enumerate(text_fragments)]
                    context = "\n".join(fragments)
                    round_questions = judge_agent.synthetize_questions_with_context(
                        context=context, 
                        question_type=question_type, 
                        example=example, 
                        difficulty=fix_difficulty
                    )
                    if fix_difficulty:
                        temp_questions: Dict[str, Dict] = {d: None for d in ["easy", "medium", "hard"]}
                        temp_questions[fix_difficulty] = round_questions
                    else:
                        temp_questions: Dict[str, Dict] = round_questions
                    questions.append(temp_questions)
            else:
                questions = judge_agent.synthetize_questions_multi_rounds(
                    seed_questions=batch, 
                    question_type=question_type, 
                    example=example, 
                    difficulty=fix_difficulty
                )
            
            # temporarily save synthetized questions
            syn_questions.append(questions)
            dump_json(syn_questions, save_path)
            pbar.update(1)

    