import os
import json
from typing import List, Dict, Tuple




# load and save files
### json
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data

def dump_json(obj, path: str):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=4)

### json line (jsonl)
def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            if line:
                data.append(json.loads(line.strip()))
    return data

def dump_jsonl(obj: list, path: str):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fw:
        for line in obj:
            fw.write(json.dumps(line, ensure_ascii=False) + "\n")



# process background knowledge texts
def chunk_to_batch(chunks: List[Dict], batch_size: int = 10) -> List[Dict]:
    batches: List[Dict] = []
    for paragraph_id, paragraph_data in enumerate(chunks):
        paragraph: List[str] = paragraph_data["chunks"]
        paragraph_area: str = paragraph_data["area"]
        batch = []
        for sentence_id, sentence in enumerate(paragraph):
            batch.append((sentence, (paragraph_id, sentence_id)))
        
            if len(batch) == batch_size:
                batches.append({"batch": batch, "area": paragraph_area})
                batch = []
        if batch:
            batches.append({"batch": batch, "area": paragraph_area})
    
    return batches


def chunk_to_batch_no_position(chunks: List[Dict], batch_size: int = 10) -> List[List[str]]:
    batches: List[List[str]] = []
    batch: List[str] = []
    for paragraph_data in chunks:
        paragraph: List[str] = paragraph_data["chunks"]
        for sentence in paragraph:
            batch.append(sentence)
            if len(batch) == batch_size:
                batches.append(batch)
                batch: List[str] = []
    if batch:
        batches.append(batch)

    return batches


def split_into_batches(arrays: List, batch_size: int = 10) -> List[List]:
    batches: List[List] = [arrays[idx : idx+batch_size] for idx in range(0, len(arrays), batch_size)]
    return batches


# class of corpus
class Corpus:
    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = corpus_path
        self.corpus = self._load_data(corpus_path)

    def _load_data(self, corpus_path: str) -> List[List[str]]:
        jsonl_data = load_jsonl(corpus_path)
        corpus_data = [d["chunks"] for d in jsonl_data]
        return corpus_data

    def idx_to_chunk(self, chunk_id: Tuple[int, int]) -> str:
        return self.corpus[chunk_id[0]][chunk_id[1]]