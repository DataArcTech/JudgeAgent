import os
import re
from typing import List

from JudgeAgent.utils import *



# process MedQA
def split_chunks_MedQA(data_dir: str, save_dir: str):
    def load_text(path: str):
        datas = []
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if len(line) > 0:
                    datas.append(line)
        return datas

    all_corpus = []
    text_dir = os.path.join(data_dir, "textbooks", "en")
    for root, dirs, files in os.walk(text_dir):
        for f in files:
            area = f.replace(".txt", "").replace("_", " ").lower()
            paragraphs = load_text(os.path.join(text_dir))
            chunks: List[str] = []
            for paragraph in paragraphs:
                if len(paragraph.split()) > 5:
                    chunks.append(paragraph)
                
            if len(chunks) > 0:
                all_corpus.append({"chunks": chunks, "area": area})
    
    dump_jsonl(all_corpus, os.path.join(save_dir, "corpus_chunks.jsonl"))

def process_questions_MedQA(question_path: str, save_dir: str):    
    raw_questions: List[Dict] = load_jsonl(question_path)
    questions: List[Dict] = []
    for q in raw_questions:
        questions.append({
            "question": q["question"], 
            "options": q["options"], 
            "answer": q["answer_idx"], 
            "meta_info": q["meta_info"], 
            "area": None
        })
    
    dump_json(questions, os.path.join(save_dir, "questions.json"))


# process MultiHopRAG
def split_chunks_MultiHopRAG(data_dir: str, save_dir: str):
    all_corpus = []
    raw_corpus: List[Dict] = load_json(os.path.join(data_dir, "corpus.json"))
    for data in raw_corpus:
        area = data.pop("category")
        body: str = data.pop("body")
        chunks = [s for s in body.split("\n\n") if s]
        all_corpus.append({"chunks": chunks, "area": area})

    dump_jsonl(all_corpus, os.path.join(save_dir, "corpus_chunks.jsonl"))

def process_questions_MultiHopRAG(question_path: str, save_dir: str):
    if not os.path.exists(question_path):
        raise Exception(f"{question_path} doesn't exist.")

    raw_questions: List[Dict] = load_jsonl(question_path)
    questions: List[Dict] = []
    for q in raw_questions:
        evidence_list = q["evidence_list"]
        area = evidence_list[0]["category"] if len(evidence_list) > 0 else None
        meta_info = "\n".join([e["fact"] for e in evidence_list])

        questions.append({
            "question": q["query"], 
            "answer": q["answer"], 
            "meta_info": meta_info, 
            "area": area
        })
    
    dump_json(questions, os.path.join(save_dir, "questions.json"))


# process QuALITY
def process_data_QuALITY(data_dir: str, save_dir: str, benchmarking_batch_size: int = 3):
    """
    Notice: "benchmarking_batch_size" is related to Benchmark Grading stage of JudgeAgent.
    For QuALITY, we only sample a batch for each article.
    """
    datas = load_json(os.path.join(data_dir, "data", "dev.json"))
    all_corpus = []
    all_questions = []
    for d in datas:
        area: str = d["topic"]
        # split chunks
        article: str = d["article"]
        article = article.replace("\n", "##SPLIT##", 2).replace("\n\n\n", "##SPLIT##").replace("\n", " ")
        article = re.sub(r"\s{2,}", " ", article)
        chunks = []
        for p in article.split("##SPLIT##")[2:]:
            p = p.strip()
            if len(p.split()) >= 3:
                chunks.append(p)
        all_corpus.append({"chunks": chunks, "area": area})

        # process questions
        article: str = d["article"]
        article = article.replace("\n\n\n ", "##SPLIT##").replace("\nBy ", "##By##", 1).replace("\n", " ")
        article = re.sub(r"\s{2,}", " ", article)
        article = article.replace("##SPLIT##", "\n").replace("##By##", "\nBy ")

        total_question_num = 0
        difficulty_questions_dict: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}
        for q in d["questions"]:
            new_q = {"question": q["question"], "options": q["options"]}
            gold_label = q["gold_label"]
            acc_num = 0
            for v in q["speed_validation"]:
                if v["speed_answer"] == gold_label:
                    acc_num += 1
            if acc_num <= 1:
                difficulty = "hard"
            elif acc_num <= 3:
                difficulty = "medium"
            else:
                difficulty = "easy"
            new_q = {**new_q, **{"answer": gold_label-1, "difficulty": difficulty}}
            difficulty_questions_dict[difficulty].append({
                "question": q["question"], 
                "options": {chr(ord("A")+i): o for i, o in enumerate(q["options"])}, 
                "answer": chr(ord("A")+gold_label-1), 
                "difficulty": difficulty, 
                "area": area
            })
            total_question_num += 1
        
        questions = []
        if total_question_num < benchmarking_batch_size:
            for d_questions in difficulty_questions_dict.values():
                questions.extend(d_questions)
        else:
            qnum_in_each_difficulty = [benchmarking_batch_size // 3] * 3
            for i in range(benchmarking_batch_size - 3*qnum_in_each_difficulty[0]):
                qnum_in_each_difficulty[i] += 1
            for i, difficulty in enumerate(["hard", "medium", "easy"]):
                qnum = qnum_in_each_difficulty[i]
                d_questions = difficulty_questions_dict[difficulty]
                if qnum > len(d_questions):
                    questions.extend(d_questions)
                    if i < 2:
                        qnum_in_each_difficulty[i+1] += qnum - len(d_questions)
                else:
                    random.shuffle(d_questions)
                    questions.extend(d_questions[:qnum])

        all_questions.append({
            "questions": questions, 
            "article": article
        })

    dump_jsonl(all_corpus, os.path.join(save_dir, "corpus_chunks.jsonl"))
    dump_json(questions, os.path.join(save_dir, "questions.json"))





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=3, help="the batch size in Benchmark Grading, only useful for QuALITY in this code.")
    args = parser.parse_args()
    data_name: str = args.data
    set_random_seed(args.random_seed)

    data_dir = os.path.join("data", data_name)
    save_dir = os.path.join("processed_data", data_name)
    if data_name.lower() == "medqa":
        split_chunks_MedQA(data_dir, save_dir)
        process_questions_MedQA(os.path.join(data_dir, "questions", "US", "4_options", "phrases_no_exclude_test.jsonl"), save_dir)
    elif data_name.lower() == "multihoprag":
        split_chunks_MultiHopRAG(data_dir, save_dir)
        process_questions_MultiHopRAG(os.path.join(data_dir, "MultiHopRAG.json"), save_dir)
    elif data_name.lower() == "quality":
        process_data_QuALITY(data_dir, save_dir, args.bs)
    else:
        raise ValueError(f"The process of {data_name} is not supported now, please write the code yourself first.")