import os
import time
import json
from typing import List, Dict, Tuple

from .client import LLMClient
from .prompt import LABEL_PROMPT



DEFAULT_BATCH_SIZE = 10


class Entity:
    def __init__(self, name, type):
        self.name:str = name
        self.type:str = type
    
    def __eq__(self, other: "Entity"):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"Entity({self.name}, {self.type})"

    def to_dict(self):
        return {
            "name": self.name, 
            "type": self.type
        }

    @staticmethod
    def from_dict(dict):
        entity = Entity(name=dict["name"], type=dict["type"])
        return entity


def label_entity_for_texts(
    texts: List[str], 
    client: LLMClient, 
    area: str = None, 
    batch_size: int = DEFAULT_BATCH_SIZE, 
    language: str = "English"
) -> List[List[Dict]]: 
    if language == "Chinese":
        system_message = "你是一个智能的NER系统。给定特定的实体类型和一些文本，你需要识别并标注这些文本中的实体。"
        system_message = system_message + f"\n领域：{area}" if area else system_message
    elif language == "English":
        system_message = "You are an intelligent Named Entity Recognition system. Please extract all entities that are important for solving the text.\nDon't include the whole article as a sentence in the return json format."
        system_message = system_message + f"\nArea: {area}" if area else system_message
    else:
        raise ValueError(f"Language {language} is not supported temporarily.")

    labeled_entities_all: List[List[Dict]] = []
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx : idx+batch_size]
        if language == "Chinese":
            batch_sentences = "\n\n".join([f"句子{si+1}: {txt}" for si, txt in enumerate(batch_texts)])
            annotation = "\n（其中，\"专有名词\"指的是" + area + "领域有可能用上的所有专有名词/概念。）" if area else ""
        elif language == "English":
            batch_sentences = "\n\n".join([f"Sentence{si+1}: {txt}" for si, txt in enumerate(batch_texts)])
            annotation = "\nPlease label the Named Entities refer to all proper nouns/concepts that may be used in the " + area + " field." if area else ""

        prompt = LABEL_PROMPT[language].format(annotation=annotation, sentences=batch_sentences)
        messages = [
            {"role": "system", "content": system_message}, 
            {"role": "user", "content": prompt}
        ]

        labeled_text = client.chat(messages=messages, json_format=True, temperature=0.01)
        try:
            if "labeled_data" in labeled_text:
                labeled_text = json.loads(labeled_text)["labeled_data"]
            else:
                if not labeled_text.startswith("["):
                    labeled_text = "[" + labeled_text
                if not labeled_text.endswith("]"):
                    labeled_text = labeled_text + "]"
                labeled_text = json.loads(labeled_text)
            
            for d in labeled_text:
                if "entity_list" in d:
                    entities: List[Dict] = []
                    for e in d["entity_list"]:
                        if "entity_text" in e:
                            ename = e["entity_text"]
                            etype = e["entity_type"] if "entity_type" in e else ""
                            entities.append({"name": ename, "type": etype})
                    labeled_entities_all.append(entities)
        except Exception as e:
            print(f"Exception during json.loads: {e}")
    
    return labeled_entities_all


def label_entity_for_batch(
    batch: List[Tuple[str, Tuple[int, int]]], 
    client: LLMClient, 
    area: str = None, 
    batch_size: int = DEFAULT_BATCH_SIZE, 
    language: str = "English"
):
    '''
    args:
        batch: a batch of sentences, each element is in the form of (sentence, (paragraph_id, sentence_id))
    '''
    sentences = [sentence for sentence, _ in batch]
    labeled_entities = label_entity_for_texts(sentences, client, area, batch_size=batch_size, language=language)
    
    entity_dict: Dict[int, Dict[int, List[Entity]]] = {}
    for entities, (_, (paragraph_id, sentence_id)) in zip(labeled_entities, batch):
        if paragraph_id not in entity_dict:
            entity_dict[paragraph_id] = {}
        entity_dict[paragraph_id][sentence_id] = entities

    return entity_dict