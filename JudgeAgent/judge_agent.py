import os
from tqdm import tqdm
from typing import List, Dict, Tuple

from .utils import *
from .prompt import *
from .client import LLMClient
from .sample import GraphPathSampler



class JudgeAgent:
    def __init__(self, 
        llm_client: LLMClient, 
        sampler: GraphPathSampler, 
        save_dir: str, 
        max_extend_rounds: int = 3, 
        benchmarking_batch_size: int = 3
    ) -> None:
        self.llm_client = llm_client
        self.sampler = sampler
        self.benchmarking_batch_size = benchmarking_batch_size
        self.max_extend_rounds = max_extend_rounds

        # optimization for graph sampling in extension stage
        self.sampler.sample_hop = self.sampler.sample_hop + max_extend_rounds - 1

        # settings for saving files
        self.save_dir = save_dir
        self.stage1_path = os.path.join(save_dir, "stage1_benchmark_grading.json")
        self.stage2_path = os.path.join(save_dir, "stage2_interactive_extension.json")
        self.stage3_path = os.path.join(save_dir, "stage3_evaluation_feedback.json")

    def _process_question_data(self, questions: List[Dict]) -> Tuple[str, List[Tuple[str, str]]]:
        text = ""
        answers: List[str] = []
        for q in questions:
            question: str = q["question"]
            options: Dict[str, str] = q.get("options", None)
            answer: str = q["answer"]
            difficulty: str = q.get("difficulty", None)

            answers.append([(answer, difficulty)])
            single_text = f"\n\t{{\"question\": \"{question.strip()}\""
            if options is not None:
                option_text = {o:c.strip() for o, c in options.items()}
                single_text += f", \"options\": {option_text}"
            single_text += "}}"
            text += single_text
        return text, answers

    def _process_llm_answers(self, response: str) -> List[str]:
        answers: List[str] = []
        try:
            json_answers: List[Dict] = json.loads(response)["response"]
            for ans in json_answers:
                answers.append(ans.get("answer", None))
        except:
            answers = []
        return answers

    def stage_benchmark_grading(self, target: LLMClient, questions: List[Dict], question_type: QTYPE, load_record: bool = True) -> List[List[Dict]]:
        # preprocess questions in benchmark to the format which can be filled by PromptFiller
        datas: List[Dict] = []
        if question_type == QTYPE.RMC:
            for qdata in questions:
                batch_questions, batch_answers = self._process_question_data(qdata["questions"])
                datas.append({"questions": batch_questions, "article": qdata["article"], "answers": batch_answers})
        else:
            batches: List[List[Dict]] = split_into_batches(questions, self.benchmarking_batch_size)
            for qdata in batches:
                batch_questions, batch_answers = self._process_question_data(qdata)
                datas.append({"questions": batch_questions, "answers": batch_answers})

        results: List[Dict] = load_json(self.stage1_path) if os.path.exists(self.stage1_path) and load_record else []
        index = len(results)

        print("# Start Stage 1: Benchmark Grading.")
        with tqdm(total=len(datas), desc="stage 1") as pbar:
            pbar.update(index)
            for data in datas[index : ]:
                prompt = PROMPT_FILLER.fill(PTYPE.ANSWER, question_type, data)
                response = target.chat(messages=[{"role": "user", "content": prompt}])
                
                llm_answers = self._process_llm_answers(response)
                correct_answers = data["answers"]

