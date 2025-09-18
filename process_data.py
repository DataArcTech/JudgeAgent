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
            chunks = []
            for paragraph in paragraphs:
                if len(paragraph.split()) > 5:
                    chunks.append(paragraph)
                
            if len(chunks) > 0:
                all_corpus.append({"chunks": chunks, "area": area})
    
    dump_jsonl(all_corpus, os.path.join(save_dir, "corpus_chunks.jsonl"))

def process_questions_MedQA(question_path: str, save_dir: str):
    if not os.path.exists(question_path):
        raise Exception(f"{question_path} doesn't exist.")
    
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
def process_data_QuALITY(data_dir: str, save_dir: str):
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
        questions = []
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
            questions.append({
                "question": q["question"], 
                "options": {chr(ord("A")+i): o for i, o in enumerate(q["options"])}, 
                "answer": chr(ord("A")+gold_label-1), 
                "difficulty": difficulty, 
                "area": area
            })
        all_questions.append({
            "questions": questions, 
            "article": article
        })

    dump_jsonl(all_corpus, os.path.join(save_dir, "corpus_chunks.jsonl"))
    dump_json(questions, os.path.join(save_dir, "questions.json"))


if __name__ == "__main__":
    pass