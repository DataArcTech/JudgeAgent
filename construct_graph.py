from typing import List, Dict, Set
from tqdm import tqdm

from JudgeAgent.label_entity import Entity, label_entity_for_batch
from JudgeAgent import *



def load_entity_dict(path: str) -> Dict[int, Dict[int, Set[Entity]]]:
    entity_dict: Dict[int, Dict[int, Set[Entity]]] = {}

    if os.path.exists(path):
        data: Dict[int, Dict[int, Set[Dict]]] = load_json(path)
        
        for pid, pdict in data.items():
            entity_dict[pid] = {}
            for sid, sentities in pdict.items():
                entity_dict[pid][sid] = set([Entity.from_dict(se) for se in sentities])

    return entity_dict

def save_entity_dict(entity_dict: Dict[int, Dict[int, Set[Entity]]], path: str):
    save_dict: Dict[int, Dict[int, List[Dict]]] = {}
    for pid, pdict in entity_dict.items():
        save_dict[pid] = {}
        for sid, sentities in pdict.items():
            save_dict[pid][sid] = [se.to_dict() for se in sentities]
    dump_json(save_dict, path)

def refresh_entity_dict(base_dict: Dict[int, Dict[int, Set[Entity]]], add_dict: Dict[int, Dict[int, List[Entity]]]) -> Dict[int, Dict[int, Set[Entity]]]:
    for pid, pdict in add_dict.items():
        if pid not in base_dict:
            base_dict[pid] = {}
        for sid, sentities in pdict.items():
            if sid not in base_dict[pid]:
                base_dict[pid][sid] = set(sentities)
            else:
                base_dict[pid][sid].update(sentities)
    return base_dict





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--bs", type=int, default=10)
    args = parser.parse_args()
    data_name: str = args.data
    batch_size: int = args.bs

    data_dir = os.path.join("processed_data", data_name)
    graph = Graph(data_dir)

    chunks = load_jsonl(os.path.join(data_dir, "corpus_chunks.jsonl"))
    batches = chunk_to_batch(chunks, batch_size=batch_size)

    progress_path = os.path.join("temp_progress", data_name, "label_entity_progress_index.json")
    index = load_json(progress_path) if os.path.exists(progress_path) else {"index": 0}
    entity_dict_path = os.path.join(data_dir, "entity_dict.json")
    entity_dict: Dict[int, Dict[int, Set[Entity]]] = load_entity_dict(entity_dict_path)


    client = LLMClient(MODEL_PARAMS[args.model])

    with tqdm(total=len(batches), desc="label entity") as pbar:
        pbar.update(index["index"])

        for batch_data in batches[index["index"] : ]:
            batch, area = batch_data["batch"], batch_data["area"]

            temp_entity_dict = label_entity_for_batch(batch, client, area, batch_size=batch_size)
            entity_dict = refresh_entity_dict(entity_dict, temp_entity_dict)

            save_entity_dict(entity_dict, entity_dict_path)
            index["index"] += 1
            dump_json(index, progress_path)
            pbar.update(1)

    graph.construct_graph_from_entity_dict(entity_dict)
    graph.save_graph()