import os
from tqdm import tqdm
from typing import List, Dict

from JudgeAgent import *
from JudgeAgent.label_entity import label_entity_for_texts





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--lang", type=str, default="English")
    args = parser.parse_args()
    data_name: str = args.data
    language: str = args.lang

    data_dir = os.path.join("processed_data", data_name)
    save_path = os.path.join(data_dir, "question_with_entities.json")
    questions = load_json(os.path.join(data_dir, "questions.json"))
    question_with_entities: List[Dict] = load_json(save_path) if os.path.exists(save_path) else []

    client = LLMClient(MODEL_PARAMS[args.model])
    
    with tqdm(total=len(questions), desc="label entity on question") as pbar:
        index = len(question_with_entities)
        pbar.update(index)

        if data_name.lower() == "quality":
            for qdata in questions[index : ]:
                new_questions = []
                for q in qdata["questions"]:
                    question, area = q["question"], q["area"]
                    labeled_entities = label_entity_for_texts([question], client, area, language=language)
                    entities = labeled_entities[0]

                    new_questions.append({**q, **{"entities": entities}})
                
                question_with_entities.append({
                    "quetions": new_questions, 
                    "article": qdata["article"]
                })
                index += 1
                dump_json(question_with_entities, save_path)
                pbar.update(1)
        else:
            for qdata in questions[index : ]:
                question, area = qdata["question"], qdata["area"]
                labeled_entities = label_entity_for_texts([question], client, area, language=language)
                entities = labeled_entities[0]

                question_with_entities.append({**qdata, **{"entities": entities}})

                index += 1
                dump_json(question_with_entities, save_path)
                pbar.update(1)
