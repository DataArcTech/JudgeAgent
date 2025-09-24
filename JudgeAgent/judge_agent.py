import os
from tqdm import tqdm
from typing import List, Dict, Set, Any, Tuple 

from .utils import *
from .prompt import *
from .client import LLMClient
from .sample import GraphPathSampler
from .label_entity import label_entity_for_texts



class DifficultyController:
    def __init__(self) -> None:
        self.score = 0.0
        self.total = 0
        self.thresholds: List[float] = [0.5, 1.0]
        self.difficulties: List[str] = ["easy", "medium", "hard"]
        self.score_map: Dict[bool, Dict[str, float]] = {
            True: {"easy": 1, "medium": 1.5, "hard": 2},  # when correct 
            False: {"easy": 0, "medium": 0, "hard": -0.2}  # when incorrect
        }

    def _get_difficulty(self, score: float) -> str:
        idx = 0
        while idx < 2 and score > self.thresholds[idx]:
            idx += 1
        return self.difficulties[idx]
    
    def update(self, is_correct: bool, difficulty: str = None):
        self.total += 1
        difficulty = difficulty if difficulty else "medium"
        self.score += self.score_map[is_correct][difficulty]
        return self._get_difficulty(self.score / self.total)

    def get_score(self) -> Tuple[float, int]:
        return self.score, self.total
    
    def load_score(self, score: float, total: int):
        self.score = score
        self.total = total

    def to_dict(self) -> Dict:
        return {
            "score": self.score, 
            "total": self.total
        }
    
    def load_dict(self, d: Dict):
        self.score: float = d.get("score", 0.0)
        self.total: int = d.get("total", 0)

    @staticmethod
    def from_dict(d: Dict) -> "DifficultyController":
        controller = DifficultyController()
        controller.load_dict(d)
        return controller


class Evaluation:
    def __init__(self) -> None:
        self.lack_knowledge: str = ""
        self.lack_ability: str = ""
        self.performance: str = ""
        self.suggestions: str = ""
    
    def load_dict(self, d: Dict[str, str]):
        self.lack_knowledge = d.get("lack_of_knowledge", "")
        self.lack_ability = d.get("lack_of_ability", "")
        self.performance = d.get("comprehensive_performance", "")
        self.suggestions = d.get("suggestions", "")

    def to_dict(self) -> Dict[str, str]:
        return {
            "lack_of_knowledge": self.lack_knowledge, 
            "lack_of_ability": self.lack_ability, 
            "comprehensive_performance": self.performance, 
            "suggestions": self.suggestions
        }
    
    @staticmethod
    def from_dict(d: Dict[str, str]) -> "Evaluation":
        evaluation = Evaluation()
        evaluation.load_dict(d)
        return evaluation

    def __str__(self) -> str:
        return json.dumps(self.to_dict, ensure_ascii=False, indent=4)


class EvalUnit:
    """
    Class of the entire evaluation unit to save in files
    """
    def __init__(self, 
        question_datas: List[Dict] = [],  # basic information of questions in a batch 
        ground_answers: List[str] = [],  # ground-truth answers in a batch
        max_rounds: int = 0,            # the max rounds of Interactive Extension
    ) -> None:
        # base questions
        # basic information
        self.questions: List[Dict] = question_datas
        self.answers: List[str] = ground_answers
        # llm answers of each round 
        # the 0-th element is the answers before evaluation
        # the i-th element is the answers after the i-th extension evaluation)
        self.llm_answers: List[List[str]] = [[] for _ in range(max_rounds+1)]
        self.corrects: List[List[bool]] = [[] for _ in range(max_rounds+1)]

        # extend questions
        self.extend_questions: List[Dict] = []
        self.extend_answers: List[str] = []
        self.extend_llm_answers: List[str] = []
        self.extend_corrects: List[bool] = []

        # evaluation result
        self.evaluation = Evaluation()

        # difficulty controller
        self.difficulty_controller = DifficultyController()
        self.difficulty_record = {
            "scores": [0.0 for _ in range(max_rounds+1)], 
            "totals": [0 for _ in range(max_rounds+1)]
        }

    def add_extend_question(self, question: Dict):
        self.extend_questions.append(question)

    def add_extend_answer(self, answer: str):
        self.extend_answers.append(answer)

    def add_extend_llm_answer(self, answer: str, correct: bool, difficulty: str = None):
        self.extend_llm_answers.append(answer)
        self.extend_corrects.append(correct)

        self.difficulty_controller.update(correct, difficulty)
        score, total = self.difficulty_controller.get_score()
        round_idx = len(self.extend_llm_answers)
        self.difficulty_record["scores"][round_idx] = score
        self.difficulty_record["total"][round_idx] = total

    def add_llm_answers(self, answers: List[str], corrects: List[bool], difficulties: List[str], round: int):
        self.llm_answers[round] = answers
        self.corrects[round] = corrects

        if round == 0:
            # only adjust the difficulty for round 0, which is the result before evaluation feedback
            for c, d in zip(corrects, difficulties):
                self.difficulty_controller.update(c, d)
            score, total = self.difficulty_controller.get_score()
            self.difficulty_record["scores"][0] = score
            self.difficulty_record["total"][0] = total

    def add_evaluation(self, evaluation: Dict[str, str]):
        self.evaluation.load_dict(evaluation)
    
    def load_dict(self, d: Dict):
        # load base questions
        base = d["base"]
        self.questions = base["questions"]
        self.answers = base["answers"]
        self.llm_answers = base["llm_answers"]
        self.corrects = base["corrects"]
        # extend
        extend = d["extend"]
        self.extend_questions = extend["questions"]
        self.extend_answers = extend["answers"]
        self.extend_llm_answers = extend["llm_answers"]
        self.extend_corrects = extend["corrects"]
        # evaluation
        self.evaluation.load_dict(d["evaluation"])
        # difficulty controller
        round_idx = len(self.extend_llm_answers)
        difficulty_scores = d["difficulty scores"]
        self.difficulty_record = difficulty_scores
        self.difficulty_controller.load_score(difficulty_scores["scores"][round_idx], difficulty_scores["totals"][round_idx])

    def to_dict(self) -> Dict:
        return {
            "base": {
                "questions": self.questions, 
                "answers": self.answers, 
                "llm_answers": self.llm_answers, 
                "corrects": self.corrects
            }, 
            "extend": {
                "questions": self.extend_questions, 
                "answers": self.extend_answers, 
                "llm_answers": self.extend_llm_answers, 
                "corrects": self.extend_corrects
            }, 
            "evaluation": self.evaluation.to_dict(), 
            "difficulty scores": self.difficulty_record
        }

    @staticmethod
    def from_dict(d: Dict) -> "EvalUnit":
        unit = EvalUnit()
        unit.load_dict(d)
        return unit

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


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
        self.sample_hop = self.sampler.sample_hop
        self.sampler.sample_hop = self.sampler.sample_hop + max_extend_rounds - 1

        # settings for saving files
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, "evaluation_results.json")


    # functions for process the response from llms
    def _process_question_data(self, questions: List[Dict]) -> Tuple[str, List[Dict], List[str]]:
        text = ""
        qdatas: List[Dict] = []
        answers: List[str] = []
        for q in questions:
            question: str = q["question"]
            options: Dict[str, str] = q.get("options", None)
            answer: str = q.pop("answer")
            qdatas.append(q)

            answers.append(answer)
            single_text = f"\n\t{{\"question\": \"{question.strip()}\""
            if options is not None:
                option_text = {o:c.strip() for o, c in options.items()}
                single_text += f", \"options\": {option_text}"
            single_text += "}}\n"
            text += single_text
        return text, qdatas, answers

    def _process_llm_answers(self, response: str, ans_num: int) -> List[str]:
        answers: List[str] = []
        try:
            json_answers: List[Dict] = json.loads(response)["response"]
            for ans in json_answers:
                answers.append(ans.get("answer", None))
        except:
            answers = []
        if len(answers) < ans_num:
            answers = answers + [None for _ in range(ans_num - len(answers))]
        return answers

    def _process_sog_response(self, response: str) -> Dict[str, Any]:
        try:
            questions: Dict[str, Any] = json.loads(response)["generated_question"]
        except:
            questions: Dict[str, Any] = {}
        return questions


    # functions of tools
    def load_eval_units(self, path: str):
        units_json_format: List[Dict] = load_json(path) if os.path.exists(path) else []
        units = [EvalUnit.from_dict(u) for u in units_json_format]
        return units

    def save_eval_units(self, units: List[EvalUnit], path: str):
        units_json_format: List[Dict] = [u.to_dict() for u in units]
        dump_json(units_json_format, path)

    def generate_example_text(self, questions: List[Dict], difficulty: str = None) -> str:
        question_dict: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}
        for q in questions:
            question_dict[q["difficulty"]].append({"question": q["question"], "options": q["options"], "answer": q["answer"]})
        examples = {}
        for d, sub_questions in question_dict.items():
            if len(sub_questions) > 0:
                examples[d] = random.choice(sub_questions)

        if difficulty:
            example = examples.get(difficulty, "")
            if example:
                example_text = "{{\n\t\"generated_question\": {{"
                example_text += "\n\t\t{{\"question\": {q}, \"options\": {o}, \"answer\": {a}}}".format(d=d, q=example["question"], o=example["options"], a=example["answer"])
                example_text += "\n}}"
            else:
                example_text = ""
        else:
            example_text = "{{\n\t\"generated_question\": {{"
            for d, example in examples.items():
                example_text += "\n\t\t\"{d}\": [{{\"question\": {q}, \"options\": {o}, \"answer\": {a}}}]".format(d=d, q=example["question"], o=example["options"], a=example["answer"])
            example_text += "\n}}"

        return example_text
        
    def validate_answers_correct(self, correct_answers: List[str], llm_answers: List[str], question_type: QTYPE) -> List[bool]:
        """
        functions for validate whether the llm answers are correct according to the correct answers 
        """
        if question_type in [QTYPE.KMC, QTYPE.RMC]:
            # validate answers for multiple-choice questions
            corrects = [a1.lower() == a2.lower() for a1, a2 in zip(correct_answers, llm_answers)]
        else:
            # validate answers for phrase Q&A questions
            ans_num: int = len(correct_answers)
            corrects: List[bool] = [False for _ in range(ans_num)]   # initialize
            # find out those llm answers which are the same as the correct answers to reduce the call of LLMs
            idx_need_llm: List[int] = []
            for i, a1, a2 in zip(range(ans_num), correct_answers, llm_answers):
                if a1.lower() == a2.lower():
                    corrects[i] = True
                else:
                    idx_need_llm.append(i)
            # call LLMs to validate the answers
            if idx_need_llm:
                answer_list: List[Tuple[str, str]] = [(correct_answers[i], llm_answers[i]) for i in idx_need_llm]
                prompt = VALIDATE_PROMPT.format(answer_list=answer_list)
                response = self.llm_client.chat(messages=[{"role": "user", "content": prompt}])
                try:
                    judges: List[str] = json.loads(response)["judge"]
                    judges = [j.lower() for j in judges]
                except:
                    judges: List[str] = []
                # padding
                if len(judges) < len(idx_need_llm):
                    judges = judges + ["no" for _ in range(len(idx_need_llm) - len(judges))]
                # fill the corrects array
                for idx, judge in zip(idx_need_llm, judges):
                    corrects[idx] = "yes" in judge
        return corrects


    # functions for synthetizing questions
    def synthetize_questions_with_context(self, 
        context: str, 
        question_type: QTYPE, 
        example: str = None, 
        difficulty: str = None
    ) -> Dict[str, Any]:
        """
        synthetize questions with context.
        """
        # get prompt
        prompt_type = PTYPE.SOG if difficulty else PTYPE.SOG_FULL_DIFFICULTY
        if example:
            data = {"example": example, "context": context}
        else:
            data = {"context": context}
        prompt = PROMPT_FILLER.fill(prompt_type, question_type, data)
        # synthetize questions
        response = self.llm_client.chat(messages=[{"role": "user", "content": prompt}])
        questions = self._process_sog_response(response)
        return questions

    def synthetize_questions_with_path(self, 
        path: List[Tuple[str, str]], 
        question_type: QTYPE, 
        example: str = None, 
        difficulty: str = None
    ) -> Dict[str, Any]:
        """
        synthetize questions with the sampled path.
        Args:
            path: the sampled knowledge path
            question_type: the type of question
            difficulty: the difficulty of question
            full_difficulty: whether to use prompt: SOG_FULL_DIFFICULTY_PROMPT_X, which means that generates a question for each level at the same time (totally 3 questions).
        """
        # get context from path
        path_context: List[str] = []
        for idx, (_, node_chunk_text) in enumerate(path, start=1):  # exclude the root node, since this node is the entity of seed question
            path_context.append(f"Text Fragment {idx+1}: {node_chunk_text}")
        context = "\n".join(path_context)
        return self.synthetize_questions_with_context(context, question_type, example, difficulty)

    def synthetize_questions_multi_rounds(self, 
        seed_questions: List[Dict], 
        question_type: QTYPE, 
        example: str = None, 
        difficulty: str = None, 
        max_num_set: int = 1
    ):
        """
        synthetize evaluation questions based on seed questions and context graph.
        Args:
        - seed_questions: the seed questions for expanding
        - question_type: the type of question
        - example: the example for question generation
        - difficulty: the difficulty level to be fixed. If None, means all difficulties are OK.
        - max_num_set: the max number of question sets. Each set is a list in length of the expand rounds. 
        """
        # collect entities
        raw_entities: List[str] = []
        no_entity_questions: Dict[str, List[Dict]] = {}
        for sq in seed_questions:
            if "entities" not in sq:
                area = sq.get("area", "")
                if area not in no_entity_questions:
                    no_entity_questions[area] = [sq]
                else:
                    no_entity_questions[area].append(sq)
            else:
                raw_entities.extend([e["name"] for e in sq["entities"]])
        if len(no_entity_questions) > 0:
            # label entities for the questions without entities
            for area, s_questions in no_entity_questions.items():
                labeled_entities = label_entity_for_texts(s_questions, self.llm_client, area)
                for ent_list in labeled_entities:
                    for ent in ent_list:
                        e = self.sampler.get_name_on_graph(ent["name"])
                        raw_entities.append(e)
        entity_num_dict: Dict[str, int] = {}
        for ent in raw_entities:
            if ent not in entity_num_dict:
                entity_num_dict[ent] = 1
            else:
                entity_num_dict[ent] += 1
        entities_with_num: List[Tuple[str, int]] = [(e, n) for e, n in entity_num_dict.items()]
        entities_with_num = sorted(entities_with_num, key=lambda x:x[1], reverse=True)
        entities: List[str] = [e for e, _ in entities_with_num]

        # synthetize questions based on extracted entities
        # new_questions: [{"easy": q1-easy, "medium": q1-medium, "hard": q1-hard}, {"easy": q2-easy, ...}, ...].
        # The length of new_questions is the number of extend rounds
        new_questions: List[Dict[str, Any]] = []
        for entity in entities[ : max_num_set]:
            # sample path with entity as root on the context graph
            sampled_path = self.sampler.sample_single_path_with_entity_sims(entity)
            paths: List[List[Tuple[str, str]]] = []
            if len(sampled_path) > self.sample_hop:
                for idx in range(min(self.max_extend_rounds, len(sampled_path) - self.sample_hop + 1)):
                    path = [sampled_path[pi] for pi in range(idx, idx+self.sample_hop)]
                    paths.append(path)
            # synthetize questions based on paths
            for path in enumerate(paths):
                questions = self.synthetize_questions_with_path(path, question_type, example, difficulty)
                if questions:
                    if difficulty:
                        temp_questions = {d: {} for d in ["easy", "medium", "hard"]}
                        temp_questions[difficulty] = questions
                        questions = temp_questions
                    new_questions.append(questions)

        return new_questions
            

    # functions for the pipeline of 3 stages
    def stage_benchmark_grading(self, 
        target: LLMClient, 
        questions: List[Dict], 
        question_type: QTYPE, 
        load_record: bool = True
    ) -> List[List[Dict]]:
        """
        pipeline for Stage1. Benchmark Grading 
        """
        # preprocess questions in benchmark to the format which can be filled by PromptFiller
        # step1. batch
        datas: List[Dict] = []
        if question_type == QTYPE.RMC:
            for qdata in questions:
                batch_questions, batch_qdatas, batch_answers = self._process_question_data(qdata["questions"])
                datas.append({"questions": batch_questions, "article": qdata["article"], "answers": batch_answers, "qdatas": batch_qdatas})
        else:
            batches: List[List[Dict]] = split_into_batches(questions, self.benchmarking_batch_size)
            for qdata in batches:
                batch_questions, batch_qdatas, batch_answers = self._process_question_data(qdata)
                datas.append({"questions": batch_questions, "answers": batch_answers, "qdatas": batch_qdatas})

        # load the results obtained from the last interrupted experiment
        results: List[EvalUnit] = self.load_eval_units(self.save_path) if load_record else []
        index = len(results)

        # step2. testing and tierings
        print("# Start Stage 1: Benchmark Grading.")
        with tqdm(total=len(datas), desc="stage 1") as pbar:
            pbar.update(index)
            for data in datas[index : ]:
                # get response from the target LLM
                prompt = PROMPT_FILLER.fill(PTYPE.ANSWER, question_type, data)
                response = target.chat(messages=[{"role": "user", "content": prompt}])
                
                # validate the target's answers whether are correct
                correct_answers: List[str] = data["answers"]
                llm_answers: List[str] = self._process_llm_answers(response, len(correct_answers))
                corrects = self.validate_answers_correct(correct_answers, llm_answers, question_type)

                # tiering: determine the initial difficulty for extended questions
                result = EvalUnit(data["qdatas"], correct_answers, self.max_extend_rounds)
                difficulties = [d.get("difficulty", None) for d in data["qdatas"]]
                result.add_llm_answers(llm_answers, corrects, difficulties, 0)
                results.append(result)

                # temporarily save results
                self.save_eval_units(results, self.save_path)
                pbar.update(1)

        return results

    def stage_interactive_extension(self, 
        target: LLMClient, 
        questions: List[Dict], 
        question_type: QTYPE, 
        load_record: bool = True
    ) -> List[List[Dict]]:
        """
        pipeline for Stage2. Interactive Extension
        """

    def stage_evaluation_feedback(self, 
        target: LLMClient, 
        questions: List[Dict], 
        question_type: QTYPE, 
        load_record: bool = True
    ) -> List[List[Dict]]:
        """
        pipeline for Stage3. Evaluation and Feedback
        """