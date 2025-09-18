'''
File for the classes and functions related to context graph.
'''
import os
import json
from typing import List, Set, Dict, Union

from .label_entity import Entity



class Position:
    '''
    Position of an entity, including the IDs of article, paragraph, and sentence.
    '''
    def __init__(self):
        self.paragraph_id: int = 0
        self.sentence_id: int = 0

    @staticmethod
    def from_id(paragraph_id: int, sentence_id: int):
        position = Position()
        position.paragraph_id = int(paragraph_id)
        position.sentence_id = int(sentence_id)
        return position

    @staticmethod
    def from_dict(dict):
        position = Position()
        position.sentence_id = int(dict["sen_id"])
        position.paragraph_id = int(dict["par_id"])
        return position

    def to_dict(self):
        return {
            "par_id": self.paragraph_id,
            "sen_id": self.sentence_id
        }

    def to_tuple(self):
        return (self.paragraph_id, self.sentence_id)

    def __eq__(self, other: "Position"):
        return self.paragraph_id == other.paragraph_id and self.sentence_id == other.sentence_id

    def __str__(self):
        return f"Position({self.paragraph_id}, {self.sentence_id})"

    def __hash__(self):
        return hash((self.paragraph_id, self.sentence_id))


class Node:
    def __init__(self, name=None):
        self.name = name
        self.id = None
        self.type = set() 
        self.positions: Set[Position] = set()
        self.sen_childs: Set[str] = set() 
        self.par_childs: Set[str] = set()

    @staticmethod
    def reset_id():
        Node.id = 0

    @staticmethod
    def from_entity(entity: Entity):
        node = Node(name=entity.name)
        node.type.add(entity.type)
        return node
    
    @staticmethod
    def from_dict(dict):
        node = Node()
        node.name = dict["name"]
        node.id = dict["id"]
        node.type = set(dict["type"])
        node.positions = set([Position.from_dict(position) for position in dict["positions"]])
        node.sen_childs = set(dict["sen_childs"])
        node.par_childs = set(dict["par_childs"])
        return node

    def to_dict(self):
        return {
            "name": self.name,
            "id": self.id,
            "type": list(self.type),
            "positions": [position.to_dict() for position in self.positions],
            "sen_childs": list(self.sen_childs),
            "par_childs": list(self.par_childs),
        }

    def refresh_node(self, node: "Node"):
        self.type.update(node.type)
        self.positions.update(node.positions)
        self.sen_childs.update(node.sen_childs)
        self.par_childs.update(node.par_childs)

    def refresh_sen_child(self, entitys: Union[List[Entity], Set[Entity]]):
        for entity in entitys:
            if entity.name == self.name:
                continue
            self.sen_childs.add(entity.name)
            self.par_childs.discard(entity.name)
    
    def refresh_par_child(self, entitys: Union[List[Entity], Set[Entity]]): 
        for entity in entitys:
            if entity.name == self.name or entity.name in self.sen_childs:
                continue
            self.par_childs.add(entity.name)
    
    def refresh_position(self, position: Position):
        self.positions.add(position)    
    
    def __str__(self) -> str:
        return (
            f"Node(ID: {self.id}, Name: {self.name}, Type: {', '.join(self.type)}, "
            f"Positions: {', '.join(str(pos) for pos in self.positions)}, "
            f"Sen_Childs: {', '.join(self.sen_childs)}, "
            f"Par_Childs: {', '.join(self.par_childs)})"
        )



class Graph:
    def __init__(self, save_dir: str) -> None:
        self.graph: Dict[str, Node] = {}

        self.save_dir = save_dir
        if save_dir is None:
            raise Exception("The directory for saving graph is None.")
        os.makedirs(save_dir, exist_ok=True)

        self.load_graph()

    def __getitem__(self, name: str) -> Node:
        return self.graph[name] if name in self.graph else None

    def get_node(self, name: str) -> Node:
        return self.graph[name] if name in self.graph else None

    def in_graph(self, name: str) -> bool:
        return name in self.graph
    
    def get_node_list(self) -> List[str]:
        return list(self.graph.keys())

    def assign_ids(self):
        current_id = 0
        for node in self.graph.values():
            node.id = current_id
            current_id += 1
    
    def load_graph(self, save_path: str = None):
        if len(self.graph) == 0:
            save_path = os.path.join(self.save_dir, "graph.jsonl") if save_path is None else save_path
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as fr:
                    for line in fr:
                        if line:
                            data = json.loads(line.strip())
                            name = data["name"]
                            node = Node.from_dict(data)
                            self.graph[name] = node

    def save_graph(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.save_dir, "graph.jsonl")
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as fw:
            for node in self.graph.values():
                fw.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

    def construct_graph_from_entity_dict(self, 
        entity_dict: Dict[int, Dict[int, List[Entity]]]
    ):
        for paragraph_id, paragraph_dict in entity_dict.items():
            paragraph_entities: Set[Entity] = set()

            for sentence_id, sentence_entities in paragraph_dict.items():
                for entity in sentence_entities:
                    ename = entity.name
                    if ename not in self.graph:
                        self.graph[ename] = Node.from_entity(entity)
                    node = self.graph[ename]
                    node.refresh_sen_child(sentence_entities)
                    node.refresh_position(Position.from_id(paragraph_id, sentence_id))
                paragraph_entities.update(sentence_entities)

            for entity in paragraph_entities:
                self.graph[entity.name].refresh_par_child(paragraph_entities)
    
        
