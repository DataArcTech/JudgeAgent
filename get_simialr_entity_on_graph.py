import os
import numpy as np
from typing import List, Set, Dict

from JudgeAgent import *
from JudgeAgent.embedding import EmbeddingClient




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MedQA")
    args = parser.parse_args()
    data_name: str = args.data

    # load data
    data_dir = os.path.join("processed_data", data_name)
    graph = Graph(data_dir)

    question_path = os.path.join(data_dir, "question_with_entities.json")
    save_path = os.path.join(data_dir, "questions_for_eval.json")
    question_with_entities: List[Dict] = load_json(question_path)

    client = EmbeddingClient(save_dir=os.path.join(data_dir, "embeddings"))
    print("# Finish Load Graph and Embeddings")

    # find the most similar entities on graph
    entity_set: Set[str] = set()
    if data_name.lower() == "quality":
        for qdata in question_with_entities:
            for q in qdata["questions"]:
                entity_set.update([e["name"] for e in q["entities"]])
    else:
        for qdata in question_with_entities:
            entity_set.update([e["name"] for e in qdata["entities"]])
    entities: List[str] = list(entity_set)
    entity_embeddings = client.get_embeddings(entities)

    nodes = graph.get_node_list()
    node_embeddings = client.get_embeddings(nodes)

    sims: np.ndarray = np.dot(entity_embeddings, node_embeddings.T)
    most_similar_node_ids = np.argmax(sims, axis=-1)
    most_similar_entity_on_grpah: Dict[str, str] = {}
    for ent, nid in zip(entities, most_similar_node_ids):
        most_similar_entity_on_grpah[ent] = nodes[nid]
    print("# Finish align entities on graph.")

    # save entities
    question_with_entities_on_graph: List[Dict] = []
    if data_name.lower() == "quality":
        for qdata in question_with_entities:
            new_questions = []
            for q in qdata["questions"]:
                entities_on_graph = []
                for e in q["entities"]:
                    node = graph[most_similar_entity_on_grpah[e["name"]]]
                    entities_on_graph.append({"name": node.name, "type": node.type})
                q["entities"] = entities_on_graph
                new_questions.append(q)
            question_with_entities_on_graph.append({"questions": new_questions, "article": qdata["article"]})
    else:
        for qdata in question_with_entities:
            entities_on_graph = []
            for e in qdata["entities"]:
                node = graph[most_similar_entity_on_grpah[e["name"]]]
                entities_on_graph.append({"name": node.name, "type": node.type})
            qdata["entities"] = entities_on_graph
            question_with_entities_on_graph.append(qdata)
    dump_json(question_with_entities_on_graph, save_path)