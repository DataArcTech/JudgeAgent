import os
from typing import List, Dict, Tuple

from .utils import *
from .prompt import *
from .client import LLMClient
from .sample import GraphPathSampler



class JudgeAgent:
    def __init__(self, 
        llm_client: LLMClient, 
        sampler: GraphPathSampler, 
        max_extend_rounds: int = 3, 
        benchmarking_batch_size: int = 3
    ) -> None:
        self.llm_client = llm_client
        self.sampler = sampler
        self.benchmarking_batch_size = benchmarking_batch_size
        self.max_extend_rounds = max_extend_rounds
        # optimization for graph sampling in extension stage
        self.sampler.sample_hop = self.sampler.sample_hop + max_extend_rounds - 1

    def _split_benchmark_eval_batch(self):
        pass