import os
import math
from tqdm import tqdm

from JudgeAgent import *
from JudgeAgent.embedding import EmbeddingClient



def get_embeddings(
    batches: List[List[str]], 
    client: EmbeddingClient, 
    index: Dict, 
    progress_path: str, 
    field: str
):
    with tqdm(total=len(batches), desc=f"embed {field}") as pbar:
        idx = index[field]
        pbar.update(idx)
        for batch in batches[idx : ]:
            client.get_embeddings(batch, use_cache=True, save_new=True)

            index[field] += 1
            dump_json(index, progress_path)
            pbar.update(1)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--bs", type=int, default=50)
    args = parser.parse_args()
    data_name: str = args.data
    model_name: str = args.model + "-embedding"
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"The embedding function of {args.model} is not supported in this program temporarily.")

    data_dir = os.path.join("processed_data", data_name)
    save_dir = os.path.join(data_dir, "embeddings")
    client = EmbeddingClient(
        params=MODEL_PARAMS[model_name], 
        save_dir=save_dir, 
        dimension=args.dim
    )

    progress_path = os.path.join("temp_progress", data_name, "embedding_progress_index.json")
    index = load_json(progress_path) if os.path.exists(progress_path) else {"corpus": 0, "entity": 0, "graph": 0, "bs": args.bs}
    batch_size = index["bs"]
    # corpus
    chunks = load_jsonl(os.path.join(data_dir, "corpus_chunks.jsonl"))
    batches: List[List[str]] = chunk_to_batch_no_position(chunks, batch_size)
    get_embeddings(batches, client, index, progress_path, "corpus")

    # entity of questions
    question_with_entities: List[Dict] = load_json(os.path.join(data_dir, "question_with_entities.json"))
    entities: List[str] = []
    for qdata in question_with_entities:
        q_entities: List[str] = [e["name"] for e in qdata["entities"]]
        entities.extend(q_entities)
    
    batches: List[List[str]] = split_into_batches(entities, batch_size)
    get_embeddings(batches, client, index, progress_path, "entity")

    # node name in graph
    graph = Graph(data_dir)
    graph.load_graph()
    node_list = graph.get_node_list()
    batches: List[List[str]] = split_into_batches(node_list, batch_size)
    get_embeddings(batches, client, index, progress_path, "graph")