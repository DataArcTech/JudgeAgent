"""
File for sampling functions on graph
"""
import time
import random
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Set, Dict, Tuple, Optional, Union
import numpy as np
import pickle
import atexit

from .graph import *
from .embedding import EmbeddingClient
from .utils import *



CHUNK_SIM_CACHE_SIZE = 10000    # cache size of LRU
BATCH_SAVE_INTERVAL = 50        # intervals of batch saving 
MAX_WORKERS = 4                 # max num of parallel threadings
SIMILARITY_THRESHOLD = 0.1      # threshold of similarity for early stopping
MAX_RETRIES = 3                 # max num of retries

# parameters of controlling accuracy
ACCURACY_MODE = "balanced"  # "fast", "balanced", "accurate"
ACCURACY_CONFIGS = {
    "fast": {
        "sample_max": 2,
        "consider_top_entities": 3,
        "consider_top_chunks": 3,
        "best_choice_probability": 0.5
    },
    "balanced": {
        "sample_max": 3,
        "consider_top_entities": 5,
        "consider_top_chunks": 5,
        "best_choice_probability": 0.7
    },
    "accurate": {
        "sample_max": 5,
        "consider_top_entities": 8,
        "consider_top_chunks": 8,
        "best_choice_probability": 0.9
    }
}


class TreeNode:
    """
    Class of tree node for constructing search tree
    Attributes:
        node: str, name of node / entity
        chunk_id: Tuple[int, int], id of text chunk (paragraph_id, sentence_id)
        children: List[TreeNode], list of children nodes
        parent: Optional[TreeNode], reference of parent node
    """

    def __init__(self, 
        node: str, 
        chunk_id: Tuple[int, int], 
        parent: Optional["TreeNode"] = None
    ) -> None:
        self.node = node
        self.chunk_id = chunk_id
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = parent
        self._depth = 0 if parent is None else parent._depth + 1

    def add_child(self, child_node: "TreeNode") -> None:
        child_node.parent = self
        child_node._depth = self._depth + 1
        self.children.append(child_node)

    def get_leaf_paths(self, 
        current_path: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None, 
        paths: Optional[List[List[Tuple[str, Tuple[int, int, int]]]]] = None, 
        unique_paths: Optional[Set[Tuple]] = None
    ) -> List[List[Tuple[str, Tuple[int, int, int]]]]:
        """
        get paths from the root node to all leaf nodes
        """
        if current_path is None:
            current_path = []
        if paths is None:
            paths = []
        if unique_paths is None:
            unique_paths = set()
            
        current_path.append((self.node, self.chunk_id))

        if not self.children:  # leaf node
            path_tuple = tuple(current_path)
            if path_tuple not in unique_paths:
                paths.append(current_path.copy())
                unique_paths.add(path_tuple)
        else:
            for child in self.children:
                child.get_leaf_paths(current_path, paths, unique_paths)

        current_path.pop()
        return paths

    @property
    def depth(self) -> int:
        return self._depth
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class GraphPathSampler:
    """
    Main functions:
    1. sampling paths based on the similarity between entities
    2. supporting parallel processing and cache optimization
    3. preprocess matrix of similarity between entities
    4. LRU store similarity between text chunks

    Attributes:
        input_data: str, path of corpus 
        embedding_client: EmbeddingClient, client for get embeddings
        sample_max: int, max num of positions for each entity
        sample_hop: int, max num of hop on graph
        save_dir: str, directory for saving data
        enable_parallel: bool, whether to allow parallel processing
        enable_cache: bool, whether to use cache
        similarity_threshold: float, the threshold of similarity for early stopping
        accuracy_mode: str, mode of accuracy--fast / balanced / accurate
    """
    def __init__(self, 
        input_data: str, 
        save_dir: str, 
        embedding_client: EmbeddingClient, 
        sample_max: int = 3, 
        sample_hop: int = 2, 
        enable_parallel: bool = True,
        enable_cache: bool = True,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        accuracy_mode: str = "balanced"  # 新增：准确率模式选择
    ) -> None:
        # register delete functions
        atexit.register(self.del_func)

        # base attributes
        self.input_data = input_data
        self.corpus = Corpus(input_data)
        self.graph: Graph = None
        self.embedding_client = embedding_client
        self.sample_hop = max(1, sample_hop)
        self.similarity_threshold = similarity_threshold

        # set accuracy mode
        if accuracy_mode not in ACCURACY_CONFIGS:
            accuracy_mode = "balanced"
        self.accuracy_mode = accuracy_mode
        self.accuracy_config = ACCURACY_CONFIGS[accuracy_mode]

        if sample_max == 3: # if use default value, adjust the value based on the accuracy mode
            self.sample_max = self.accuracy_config["sample_max"]
        else:
            self.sample_max = max(1, sample_max)

        # optimization configs
        self.enable_parallel = enable_parallel
        self.enable_cache = enable_cache

        # paths of saving
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.sample_neighbors_save_path = os.path.join(save_dir, "similar_neighbors.json")
        self.entity_sim_matrix_path = os.path.join(save_dir, "entity_sim_matrix.pkl")

        # cache and status
        self.most_similar_neighbors: Dict[str, List[str]] = {}
        self.entity_sim_matrix: Dict[str, Dict[str, float]] = {}
        self.chunk_sim_cache: Dict[str, float] = {}

        # threading safety
        self._lock = threading.RLock()
        self._save_counter = 0

        # load data
        

    def load_graph(self, graph: Graph) -> None:
        self.graph = graph
        
        if self.enable_cache and not self.entity_sim_matrix:
            entities = list(graph.get_node_list())
            if len(entities) <= 1000:   # only preprocess for graph of small/medium scale
                self.precompute_entity_similarities(entities)
        
    def load_similar_neighbors(self) -> None:
        try:
            if os.path.exists(self.sample_neighbors_save_path):
                self.most_similar_neighbors = load_json(self.sample_neighbors_save_path)
            else:
                self.most_similar_neighbors = {}
        except Exception as e:
            self.most_similar_neighbors = {}

    def save_similar_neighbors(self, force: bool = False) -> None:
        with self._lock:
            self._save_counter += 1
            if force or self._save_counter >= BATCH_SAVE_INTERVAL:
                try:
                    dump_json(self.most_similar_neighbors, self.sample_neighbors_save_path)
                    self._save_counter = 0
                except Exception as e:
                    pass
    
    def load_entity_similarity_matrix(self) -> None:
        try:
            if os.path.exists(self.entity_sim_matrix_path):
                with open(self.entity_sim_matrix_path, "rb") as fr:
                    self.entity_sim_matrix = pickle.load(fr)
            else:
                self.entity_sim_matrix = {}
        except:
            self.entity_sim_matrix = {}

    def save_entity_similarity_matrix(self) -> None:
        try:
            with open(self.entity_sim_matrix_path, "wb") as fw:
                pickle.dump(self.entity_sim_matrix, fw)
        except Exception as e:
            pass
    
    def precompute_entity_similarities(self, entities: Optional[List[str]] = None) -> None:
        if entities is None:
            entities = list(self.graph.get_node_list())

        embeddings = self.embedding_client.get_embeddings(entities, use_cache=True)
        sim_matrix = np.dot(embeddings, embeddings.T)

        for i, ent_i in enumerate(entities):
            if ent_i not in self.entity_sim_matrix:
                self.entity_sim_matrix[ent_i] = {}
            for j, ent_j in enumerate(entities):
                self.entity_sim_matrix[ent_i][ent_j] = float(sim_matrix[i, j])
        
        self.save_entity_similarity_matrix()
    
    @lru_cache(maxsize=CHUNK_SIM_CACHE_SIZE)
    def _chunk_similarity_cached(self, 
        query_id_str: str, 
        candidate_ids_str: str, 
        use_cache: bool = False
    ) -> Tuple[float, ...]:
        try:
            query_id = eval(query_id_str)
            candidate_ids = eval(candidate_ids_str)
            
            query_text = self.corpus.idx_to_chunk(query_id)
            candidate_texts = [self.corpus.idx_to_chunk(cid) for cid in candidate_ids]
            
            sims = self.embedding_client.text_similarities(query_text, candidate_texts, use_cache)
            
            return tuple(sims)
        except Exception as e:
            return tuple([0.0] * len(eval(candidate_ids_str)))

    def chunk_similarity(self, 
        query_id: Tuple[int, int, int], 
        candidate_ids: List[Tuple[int, int, int]], 
        use_cache: bool = False
    ) -> List[float]:
        if not candidate_ids:
            return []

        cache_key = f"{query_id}_{tuple(candidate_ids)}"
        
        if self.enable_cache:
            if cache_key in self.chunk_sim_cache:
                return self.chunk_sim_cache[cache_key]

        try:
            # 使用LRU缓存计算
            query_id_str = str(query_id)
            candidate_ids_str = str(candidate_ids)
            
            sims = list(self._chunk_similarity_cached(query_id_str, candidate_ids_str, use_cache))
            
            # 存储到缓存
            if self.enable_cache:
                self.chunk_sim_cache[cache_key] = sims
            
            return sims
            
        except Exception as e:
            return [0.0] * len(candidate_ids)

    def get_name_on_graph(self, entity: str) -> str:
        if self.graph is None:
            raise ValueError("Graph hasn't been loaded")

        if self.graph.in_graph(entity):
            return entity
        
        try:
            nodes = list(self.graph.get_node_list())
            if not nodes:
                raise ValueError("There is no node in graph")

            if entity in self.entity_sim_matrix:
                similarities = [(self.entity_sim_matrix[entity].get(node, 0.0), node) for node in nodes]
                best_sim, best_node = max(similarities, key=lambda x: x[0])
                if best_sim > self.similarity_threshold:
                    return best_node

            sims = self.embedding_client.text_similarities(entity, nodes, use_cache=True)
            sim_idxs = sorted(
                [(sim, i) for i, sim in enumerate(sims)], 
                key=lambda x: x[0], reverse=True
            )
            
            best_sim, best_idx = sim_idxs[0]
            best_node = nodes[best_idx]
            return best_node

        except Exception as e:
            nodes = list(self.graph.get_node_list())
            return nodes[0] if nodes else entity

    def _get_similar_entities(self, current_entity: str, visited_nodes: List[str]) -> List[str]:
        if current_entity in self.most_similar_neighbors:
            selected_entities = self.most_similar_neighbors[current_entity]
        else:
            # get neighbor entities
            current_node = self.graph[current_entity]
            neighbors = [n for n in current_node.sen_childs if n not in visited_nodes] if current_node else []
            
            if not neighbors:
                return []
            
            # calculate similarities
            try:
                if current_entity in self.entity_sim_matrix:
                    entity_sims = [
                        (neighbor, self.entity_sim_matrix[current_entity].get(neighbor, 0.0))
                        for neighbor in neighbors
                    ]
                else:
                    sims = self.embedding_client.text_similarities(current_entity, neighbors, use_cache=True)
                    entity_sims = [(neighbors[i], sims[i]) for i in range(len(neighbors))]
                
                # sort and filter
                entity_sims = sorted(entity_sims, key=lambda x: x[1], reverse=True)
                selected_entities = [e for e, s in entity_sims if s > self.similarity_threshold]
                
                # save result
                self.most_similar_neighbors[current_entity] = selected_entities
                self.save_similar_neighbors()
                
            except Exception as e:
                selected_entities = neighbors[:5]  # fallback
        
        # 过滤已访问的节点
        return [e for e in selected_entities if e not in visited_nodes]

    def _select_next_entity(self, selected_entities: List[str]) -> str:
        if not selected_entities:
            raise ValueError("No entities to be selected")
        
        # Strategy：ensure the first and second best options have the highest weight, others random.
        if len(selected_entities) == 1:
            return selected_entities[0]
        elif len(selected_entities) <= 3:
            weights = [0.6, 0.3, 0.1][:len(selected_entities)]
        else:
            if random.random() < 0.7:
                return random.choice(selected_entities[:2])
            else:
                candidates = selected_entities[2 : min(8, len(selected_entities))]
                return random.choice(candidates) if candidates else selected_entities[0]
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        selected_idx = np.random.choice(len(selected_entities), p=weights)
        return selected_entities[selected_idx]

    def _get_available_chunks(self, entity: Node, mask_ids_shared: Set[str]) -> List[Tuple[int, int]]:
        entity_chunk_idxs = []
        for position in entity.positions:
            cid = position.to_tuple()
            if str(cid) not in mask_ids_shared:
                entity_chunk_idxs.append(cid)
        return entity_chunk_idxs

    def _select_best_chunk(self, 
        query_chunk_idx: Tuple[int, int], 
        candidate_chunks: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if not candidate_chunks:
            return None
        
        try:
            sims = self.chunk_similarity(query_chunk_idx, candidate_chunks, use_cache=True)
            
            # sort and select
            chunk_sims = sorted(
                [(candidate_chunks[i], sims[i]) for i in range(len(sims))], 
                key=lambda x: x[1], reverse=True
            )
            
            # More conservative strategy: prioritize high-quality chunks
            high_quality_chunks = [c for c, s in chunk_sims if s > self.similarity_threshold]
            
            if len(high_quality_chunks) >= 3:
                # 80% select the best, 20% select from top-3.
                if random.random() < 0.8:
                    return chunk_sims[0][0]  # select the best
                else:
                    return random.choice([c for c, s in chunk_sims[:3]])
            elif high_quality_chunks:
                # 60% select the best when high-quality chunks are few.
                if random.random() < 0.6:
                    return chunk_sims[0][0]
                else:
                    return random.choice(high_quality_chunks)
            else:
                # select the bese directy when no chunk is high-quality
                return chunk_sims[0][0] if chunk_sims else None
            
        except Exception as e:
            return random.choice(candidate_chunks) if candidate_chunks else None

    def _process_single_position_path(self, 
        args: Tuple[Position, str, Set[str]]
    ) -> Optional[List[Tuple[str, str]]]:
        poi, node_name, mask_ids_shared = args
        
        try:
            chunk_idx = poi.to_tuple()
            path: List[Tuple[str, str]] = [(node_name, self.corpus.idx_to_chunk(chunk_idx))]
            visited_nodes = [node_name]
            current_node = self.graph[node_name]
            
            for hop in range(self.sample_hop):
                selected_entities = self._get_similar_entities(current_node.name, visited_nodes)
                
                if not selected_entities:
                    break
                
                entity_name = self._select_next_entity(selected_entities)
                entity = self.graph[entity_name]
                
                entity_chunk_idxs = self._get_available_chunks(entity, mask_ids_shared)
                
                if not entity_chunk_idxs:
                    continue
                
                selected_chunk = self._select_best_chunk(chunk_idx, entity_chunk_idxs)
                
                if selected_chunk is None:
                    continue
                
                # update the path
                with self._lock:
                    mask_ids_shared.add(str(selected_chunk))
                
                path.append((entity_name, self.corpus.idx_to_chunk(selected_chunk)))
                visited_nodes.append(entity_name)
                current_node = entity
                
                # 质量检查：如果相似度过低，提前结束
                # quality check: if the similarity is too low, stop early.
                if len(path) >= 2:
                    chunk_text = self.corpus.idx_to_chunk(selected_chunk)
                    if len(chunk_text.strip()) < 10:  # the text is too short
                        break
            
            return path if len(path) > 1 else None
            
        except Exception as e:
            return None

    def _sample_paths_parallel(self, 
        selected_positions: List[Position], 
        node: Node, 
        mask_ids: Set[str]
    ) -> List[List[Tuple[str, str]]]:
        paths = []
        
        task_args = [(poi, node.name, mask_ids) for poi in selected_positions]
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(selected_positions))) as executor:
            try:
                future_to_poi = {executor.submit(self._process_single_position_path, args): args[0] for args in task_args}
                
                for future in as_completed(future_to_poi, timeout=30): 
                    try:
                        path = future.result()
                        if path:
                            paths.append(path)
                    except Exception as e:
                        pass
                        
            except Exception as e:
                # fallback到串行处理
                return self._sample_paths_sequential(selected_positions, node, mask_ids)
        
        return paths

    def _sample_paths_sequential(self, 
        selected_positions: List[Position], 
        node: Node, 
        mask_ids: Set[str]
    ) -> List[List[Tuple[str, str]]]:
        paths = []
        
        for poi in selected_positions:
            try:
                args = (poi, node.name, mask_ids)
                path = self._process_single_position_path(args)
                if path:
                    paths.append(path)
            except Exception as e:
                continue
        
        return paths

    def sample_paths_with_entity_sims(self, 
        name: str, 
        max_retries: int = MAX_RETRIES
    ) -> List[List[Tuple[str, str]]]:
        if self.graph is None:
            raise ValueError("Graph hasn't been loaded")
        
        try:
            name = self.get_name_on_graph(name)
            node = self.graph[name]
            if node is None:
                raise KeyError(f"There is no node named {name} in the graph.")
        except Exception as e:
            raise e

        # select positions
        pois = list(node.positions)
        if len(pois) < self.sample_max:
            selected_positions = pois
        else:
            selected_positions = random.sample(pois, self.sample_max)
        
        # initialize shared status
        mask_ids = {str(poi.to_tuple()) for poi in selected_positions}
        paths: List[List[Tuple[str, str]]] = []
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                if self.enable_parallel and len(selected_positions) > 1:
                    paths = self._sample_paths_parallel(selected_positions, node, mask_ids)
                else:
                    paths = self._sample_paths_sequential(selected_positions, node, mask_ids)
                
                # filter to get valid paths
                valid_paths = [p for p in paths if p and len(p) > 1]
                
                if valid_paths:
                    return valid_paths
                
                retry_count += 1
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(0.1)
        
        return []

    def sample_single_path_with_entity_sims(self, 
        name: str, 
        max_retries: int = MAX_RETRIES
    ) -> List[Tuple[str, str]]:
        if self.graph is None:
            raise ValueError("Graph hasn't been loaded")
        
        try:
            name = self.get_name_on_graph(name)
            node = self.graph[name]
            if node is None:
                raise KeyError(f"There is no node named {name} in the graph.")
        except Exception as e:
            raise e

        retry_count = 0
        while retry_count < max_retries:
            try:
                # select the start position
                pois = list(node.positions)
                if len(pois) < self.sample_max:
                    selected_positions = pois
                else:
                    selected_positions = random.sample(pois, self.sample_max)
                
                mask_ids = {str(poi.to_tuple()) for poi in selected_positions}
                
                # randomly select a position as the start node.
                poi = random.choice(selected_positions)
                chunk_idx = poi.to_tuple()
                
                # initialize the path
                path: List[Tuple[str, str]] = [(node.name, self.corpus.idx_to_chunk(chunk_idx))]
                visited_nodes = [node.name]
                current_node = node
                
                # extend the path
                for hop in range(self.sample_hop):
                    selected_entities = self._get_similar_entities(current_node.name, visited_nodes)
                    
                    if not selected_entities:
                        break
                    
                    entity_name = self._select_next_entity(selected_entities)
                    entity = self.graph[entity_name]
                    
                    entity_chunk_idxs = self._get_available_chunks(entity, mask_ids)
                    
                    if not entity_chunk_idxs:
                        continue
                    
                    selected_chunk = self._select_best_chunk(chunk_idx, entity_chunk_idxs)
                    
                    if selected_chunk is None:
                        continue
                    
                    mask_ids.add(str(selected_chunk))
                    path.append((entity_name, self.corpus.idx_to_chunk(selected_chunk)))
                    visited_nodes.append(entity_name)
                    current_node = entity
                    
                    # update the query chunk for the similary calculation in next hop.
                    chunk_idx = selected_chunk
                
                # return valid path
                if len(path) > 1:
                    return path
                
                retry_count += 1
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise e
                time.sleep(0.1)
        
        # if all retries are failed, return the path with only the start node.
        try:
            poi = random.choice(list(node.positions))
            chunk_idx = poi.to_tuple()
            return [(node.name, self.corpus.idx_to_chunk(chunk_idx))]
        except Exception as e:
            return [(node.name, "[ERROR]: Cannot get text chunk")]

    def sample_paths_with_chunk_sims(self, 
        name: str,
        max_retries: int = MAX_RETRIES
    ) -> List[List[Tuple[str, str]]]:        
        if self.graph is None:
            raise ValueError("Graph hasn't been loaded")
        
        try:
            name = self.get_name_on_graph(name)
            node = self.graph[name]
            if node is None:
                raise KeyError(f"There is no node named {name} in the graph.")
        except Exception as e:
            raise e
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # select positions
                pois = list(node.positions)
                if len(pois) < self.sample_max:
                    selected_positions = pois
                else:
                    selected_positions = random.sample(pois, self.sample_max)
                
                paths: List[List[Tuple[str, str]]] = []
                
                # build path tree for each position
                for poi_idx, poi in enumerate(selected_positions):
                    try:
                        query_chunk_idx = poi.to_tuple()
                        root = TreeNode(name, query_chunk_idx)
                        mask_ids = {str(query_chunk_idx)}
                        current_nodes = [root]
                        
                        # build multi-layer path tree
                        for hop in range(self.sample_hop):
                            next_nodes = []
                            
                            for current_node in current_nodes:
                                try:
                                    # get node and neighbors on graph
                                    graph_node = self.graph[current_node.node]
                                    if not graph_node:
                                        continue
                                        
                                    neighbors = set(graph_node.sen_childs)
                                    chunks_with_neighbors = []
                                    
                                    # collect all available text chunks of neighbors
                                    for neighbor in neighbors:
                                        # avoid revisiting parent node
                                        if current_node.parent is None or neighbor != current_node.parent.node:
                                            neighbor_node = self.graph[neighbor]
                                            if neighbor_node:
                                                for position in neighbor_node.positions:
                                                    chunk_id = position.to_tuple()
                                                    if str(chunk_id) not in mask_ids:
                                                        chunks_with_neighbors.append((neighbor, chunk_id))
                                    
                                    if not chunks_with_neighbors:
                                        continue
                                    
                                    chunk_ids = [chunk_id for _, chunk_id in chunks_with_neighbors]
                                    sims = self.chunk_similarity(query_chunk_idx, chunk_ids, use_cache=True)
                                    
                                    # sort and select the most similar text chunk
                                    similarity_scores = sorted(
                                        [(neighbor, chunk_id, sims[i]) for i, (neighbor, chunk_id) in enumerate(chunks_with_neighbors)], 
                                        key=lambda x: x[2], reverse=True
                                    )
                                    
                                    # select top-3, and avoid repeated neighbors
                                    added_neighbors = set()
                                    for neighbor, chunk_id, sim_score in similarity_scores:
                                        if len(next_nodes) >= 3:
                                            break
                                        if neighbor not in added_neighbors and sim_score > self.similarity_threshold:
                                            child_node = TreeNode(neighbor, chunk_id, parent=current_node)
                                            current_node.add_child(child_node)
                                            next_nodes.append(child_node)
                                            mask_ids.add(str(chunk_id))
                                            added_neighbors.add(neighbor)
                                    
                                except Exception as e:
                                    continue
                            
                            current_nodes = next_nodes
                            if not current_nodes:  # no more nodes to extend
                                break
                        
                        # extract all leaf paths
                        try:
                            pois_paths = root.get_leaf_paths()
                            for path in pois_paths:
                                if len(path) > 1:
                                    path_with_chunks = [
                                        (node_name, self.corpus.idx_to_chunk(chunk_idx)) 
                                        for node_name, chunk_idx in path
                                    ]
                                    paths.append(path_with_chunks)
                        except Exception as e:
                            continue
                            
                    except Exception as e:
                        continue
                
                if paths:
                    return paths
                
                retry_count += 1
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(0.1)
        
        return []

    def del_func(self):
        """
        被注册的析构函数，确保数据保存
        """
        try:
            self.save_similar_neighbors(force=True)
        except:
            pass