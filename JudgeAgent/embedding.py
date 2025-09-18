import os
import math
import numpy as np
from typing import List, Dict, Union

from .client import LLMClient
from .llm_params import LLMParams
from .utils import *



DEFAULT_BATCH_SIZE = 10
DEFAULT_EMBEDDING_DIMENSION = 1024


class EmbeddingClient:
    def __init__(self, 
        save_dir: str, 
        params: LLMParams = None,  
        dimension: int = DEFAULT_EMBEDDING_DIMENSION, 
    ) -> None:
        self.dimension = dimension
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if params is not None:
            self.client = LLMClient(params)
        else:
            self.client = None

        self.embeddings: np.ndarray = None
        self.name_dict: Dict[str, int] = {}
        self.embeddings_num = 0

        self.load_embeddings()

    def get_llm_params(self):
        return self.client.get_llm_params() if self.client is not None else None
    
    def load_embeddings(self, save_dir: str = None):
        if self.embeddings is None:
            save_dir = self.save_dir if save_dir is None else save_dir
            if save_dir is not None and os.path.exists(save_dir):
                embedding_path = os.path.join(save_dir, "embeddings.npy")
                if os.path.exists(embedding_path):
                    self.embeddings: np.ndarray = np.load(embedding_path)
                    self.name_dict: Dict[str, int] = load_json(os.path.join(save_dir, "name_dict.json"))
                    self.dimension = self.embeddings.shape[-1]
                    self.embeddings_num: int = self.embeddings.shape[0]
            print(f"# Load Embeddings: Num={self.embeddings_num} , Dimension={self.dimension}")
        return self.embeddings_num

    def save_embeddings(self, save_dir: str = None):
        save_dir = self.save_dir if save_dir is None else save_dir
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        dump_json(self.name_dict, os.path.join(save_dir, "name_dict.json"))

    def get_embeddings(self, inputs: Union[str, List], batch_size: int = DEFAULT_BATCH_SIZE, use_cache: bool = True, save_new: bool = True) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = [inputs]
        else:
            if not isinstance(inputs, List):
                raise Exception(f"Get Embedding not support input of type {type(inputs)}.")
            inputs = list(set(inputs))

        embeddings = np.zeros((len(inputs), self.dimension))
        old_idxs: List[int] = []
        old_embed_idxs: List[int] = []
        new_idxs: List[int] = []
        new_texts: List[str] = []
        if use_cache:
            for i, text in enumerate(inputs):
                if text in self.name_dict:
                    old_idxs.append(i)
                    old_embed_idxs.append(self.name_dict[text])
                else:
                    new_idxs.append(i)
                    new_texts.append(text)
        else:
            new_idxs = list(range(len(inputs)))
            new_texts = inputs
        
        if old_idxs and self.embeddings is not None:
            embeddings[old_idxs] = self.embeddings[old_embed_idxs]
        
        if new_idxs:
            if self.client is None:
                raise Exception("Please provide parameters of LLMs before using EmbedddingClient to get new embeddings.")
            float_embeddings: List[List[float]] = []
            for idx in range(0, len(inputs), batch_size):
                batch = new_texts[idx : idx+batch_size]
                batch_embeddings = self.client.get_embeddings(batch, self.dimension, encoding_format="float")
                float_embeddings.extend(batch_embeddings)
            float_embeddings = np.array(float_embeddings)
            float_embeddings = float_embeddings / np.linalg.norm(float_embeddings, axis=-1, keepdims=True)
            embeddings[new_idxs] = float_embeddings

            eidx = self.embeddings_num
            for i, text in enumerate(new_texts):
                self.name_dict[text] = eidx + i
            if self.embeddings is not None:
                self.embeddings = np.concatenate([self.embeddings, float_embeddings], axis=0)
            else:
                self.embeddings = float_embeddings
            self.embeddings_num = self.embeddings.shape[0]

            if save_new:
                self.save_embeddings()

        return embeddings

    def text_similarities(self, query: str, candidates: List[str], use_cache: bool = True) -> List[float]:
        query_embeddings = self.get_embeddings(query, use_cache)
        cand_embeddings = self.get_embeddings(candidates, use_cache)
        sims: np.ndarray = np.dot(query_embeddings, cand_embeddings.T)
        sims = sims.reshape(len(candidates))
        return sims.tolist()
