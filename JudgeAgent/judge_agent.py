import os
import time
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union

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
        self.current_difficulty: str = None

    def _get_difficulty(self, score: float) -> str:
        idx = 0
        while idx < 2 and score > self.thresholds[idx]:
            idx += 1
        return self.difficulties[idx]
    
    def update(self, is_correct: bool, difficulty: str = None):
        self.total += 1
        difficulty = difficulty if difficulty else "medium"
        self.score += self.score_map[is_correct][difficulty]
        self.current_difficulty = self._get_difficulty(self.score / self.total)
        return self.current_difficulty

    def get_score(self) -> Tuple[float, int]:
        return self.score, self.total
    
    def load_score(self, score: float, total: int):
        self.score = score
        self.total = total

    def to_dict(self) -> Dict:
        return {
            "score": self.score, 
            "total": self.total, 
            "difficulty": self.current_difficulty
        }
    
    def load_dict(self, d: Dict):
        self.score: float = d.get("score", 0.0)
        self.total: int = d.get("total", 0)
        self.current_difficulty: str = d.get("difficulty", None)

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

    def is_empty(self) -> bool:
        return self.suggestions == ""

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
        article: str = ""               # the article of questions, only useful for QuALITY
    ) -> None:
        # base questions
        # basic information
        self.article: str = article      # QuALITY's article
        self.questions: List[Dict] = question_datas
        self.answers: List[str] = ground_answers
        # llm answers of base questions
        self.llm_answers: List[str] = []
        self.corrects: List[bool] = []
        # llm answers of base questions after evaluation of each round
        # the 0-th element is the answers after evaluation on the performance on base questions, which is the ablation setting of no extension
        # the i-th element is the answers after the i-th extension evaluation)
        self.llm_answers_after_eval: List[List[str]] = [[] for _ in range(max_rounds+1)]
        self.corrects_after_eval: List[List[bool]] = [[] for _ in range(max_rounds+1)]

        # extend questions
        self.extend_questions: List[Dict] = []
        self.extend_answers: List[str] = []
        self.extend_llm_answers: List[str] = []
        self.extend_corrects: List[bool] = []

        # evaluations after each extension round
        # the 0-th element is the evaluation of performance on base questions, which is the ablation setting of no extension
        self.evaluation: List[Evaluation] = [Evaluation() for _ in range(max_rounds+1)]

        # difficulty controller
        self.difficulty_controller = DifficultyController()
        self.difficulty_record: List[Dict[str, Any]] = [{"score": 0.0, "total": 0, "difficulty": None} for _ in range(max_rounds+1)]

        # time cost
        self.time_cost: List[float] = [0.0 for _ in range(max_rounds+1)]

    def get_accuracy(self, K: int) -> Dict:
        """
        get the statistic information of base + K-th round extension
        """
        corrects = self.corrects[0] + self.extend_corrects[ : K]
        if len(corrects) > 0:
            return sum(corrects) / len(corrects)
        else:
            return -1

    def add_extend_question(self, question: Dict):
        self.extend_questions.append(question)

    def add_extend_answer(self, answer: str):
        self.extend_answers.append(answer)

    def add_extend_llm_answer(self, answer: str, correct: bool, difficulty: str = None, fix_difficulty: str = None):
        self.extend_llm_answers.append(answer)
        self.extend_corrects.append(correct)

        self.difficulty_controller.update(correct, difficulty)
        if fix_difficulty:
            self.difficulty_controller.current_difficulty = fix_difficulty
        round_idx = len(self.extend_llm_answers)
        self.difficulty_record[round_idx] = self.difficulty_controller.to_dict()

    def add_llm_answers(self, answers: List[str], corrects: List[bool], difficulties: List[str] = None):
        self.llm_answers = answers
        self.corrects = corrects

        # adjust the difficulty for Benchmark Grading
        if difficulties is None:
            difficulties: List[str] = [None for _ in range(len(corrects))]
        for c, d in zip(corrects, difficulties):
            self.difficulty_controller.update(c, d)
        self.difficulty_record[0] = self.difficulty_controller.to_dict()
    
    def add_llm_answers_after_eval(self, answers: List[str], corrects: List[bool], round_idx: int):
        self.llm_answers_after_eval[round_idx] = answers
        self.corrects_after_eval[round_idx] = corrects

    def add_evaluation(self, evaluation: Union[Evaluation, Dict], round: int):
        if isinstance(evaluation, Dict):
            self.evaluation[round].load_dict(evaluation)
        elif isinstance(evaluation, Evaluation):
            self.evaluation[round] = evaluation
        else:
            raise ValueError(f"Type of {type(evaluation)} is not supported to add to the evaluation of EvalUnit.")

    def add_time_cost(self, time_cost: float, round_idx: int = -1):
        if round_idx < 0:
            for i in range(len(self.time_cost)):
                self.time_cost[i] += time_cost
        else:
            self.time_cost[round_idx] += time_cost
    
    def load_dict(self, d: Dict):
        # load base questions
        base = d["base"]
        self.questions: List[Dict] = base["questions"]
        self.answers: List[str] = base["answers"]
        self.llm_answers: List[str] = base["llm_answers"]
        self.corrects: List[bool] = base["corrects"]
        self.llm_answers_after_eval: List[List[str]] = base["llm_answers_after_eval"]
        self.corrects_after_eval: List[List[bool]] = base["corrects_after_eval"]
        # extend
        extend = d["extend"]
        self.extend_questions: List[Dict] = extend["questions"]
        self.extend_answers: List[str] = extend["answers"]
        self.extend_llm_answers: List[str] = extend["llm_answers"]
        self.extend_corrects: List[bool] = extend["corrects"]
        # evaluation
        for i, evaluation in enumerate(d["evaluations"]):
            self.evaluation[i].load_dict(evaluation)
        # difficulty controller
        round_idx = len(self.extend_llm_answers)
        self.difficulty_record = d["difficulty_record"]
        self.difficulty_controller.load_dict(self.difficulty_record[round_idx])
        # time cost
        self.time_cost: List[float] = d["time_cost"]

    def to_dict(self) -> Dict:
        return {
            "base": {
                "questions": self.questions, 
                "answers": self.answers, 
                "llm_answers": self.llm_answers, 
                "corrects": self.corrects, 
                "llm_answers_after_eval": self.llm_answers_after_eval, 
                "corrects_after_eval": self.corrects_after_eval
            }, 
            "extend": {
                "questions": self.extend_questions, 
                "answers": self.extend_answers, 
                "llm_answers": self.extend_llm_answers, 
                "corrects": self.extend_corrects
            }, 
            "evaluation": [e.to_dict() for e in self.evaluation], 
            "difficulty_record": self.difficulty_record, 
            "time_cost": self.time_cost
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

    def _process_eval_response(self, response: str) -> Evaluation:
        try:
            json_eval: Dict[str, str] = json.loads(response)
        except:
            json_eval: Dict[str, str] = {}
        
        return Evaluation.from_dict(json_eval)


    # functions of tools
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

    def _form_performance_to_text(self, qdatas: List[Dict], llm_answers: List[str], corrects: List[bool]) -> str:
        text_list: List[str] = ""
        for qdata, llm_answer, correct in zip(qdatas, llm_answers, corrects):
            question: str = qdata["question"]
            question_text = question

            options: Dict[str, str] = qdata.get("options", None)
            if options is not None:
                options_text = " ".join(f"{o}.{c}" for o, c in options.items())
                question_text = f"{question_text} {options_text}"

            difficulty: str = qdata.get("difficulty", None)
            if difficulty is not None:
                question_text = f"[{difficulty}-level] {question_text}"

            text = f"\n\t{{\"question\": {question_text}, \"llm_answer\": {llm_answer}, \"is_llm_correct\": {correct}}}"
            text_list.append(text)

        return "".join(text_list) + "\n"

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

    def get_statistic(self, corrects1: List[bool], corrects2: List[bool]) -> Dict[str, Dict]:
        statistic: Dict[str, Dict] = {
            "accuracy_before_eval": {"num": 0, "total": 0}, 
            "accuracy_after_eval": {"num": 0, "total": 0}, 
            "correction_rate": {"num": 0, "total": 0}, 
            "false_correction_rate": {"num": 0, "total": 0}, 
        }
        for c1, c2 in zip(corrects1, corrects2):
            if c1:
                statistic["accuracy_before_eval"]["num"] += 1
            if c2:
                statistic["accuracy_after_eval"]["num"] += 1
            if not c1 and c2:
                statistic["correction_rate"]["num"] += 1
            if c1 and not c2:
                statistic["false_correction_rate"]["num"] += 1

        for k in statistic.keys():
            statistic[k]["total"] = len(corrects1)

        return statistic


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
    ) -> List[Dict[str, Dict]]:
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
        new_questions: List[Dict[str, Dict]] = []
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
                    else:
                        temp_questions: Dict[str, Dict] = questions
                    new_questions.append(temp_questions)

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
                start_time = time.time()
                # get response from the target LLM
                prompt = PROMPT_FILLER.fill(PTYPE.ANSWER, question_type, data)
                response = target.chat(messages=[{"role": "user", "content": prompt}])
                
                # validate the target's answers whether are correct
                correct_answers: List[str] = data["answers"]
                llm_answers: List[str] = self._process_llm_answers(response, len(correct_answers))
                corrects = self.validate_answers_correct(correct_answers, llm_answers, question_type)

                # tiering: determine the initial difficulty for extended questions
                time_cost = time.time() - start_time()
                result = EvalUnit(data["qdatas"], correct_answers, self.max_extend_rounds, data.get("article", ""))
                difficulties = [d.get("difficulty", None) for d in data["qdatas"]]
                result.add_llm_answers(llm_answers, corrects, difficulties)
                result.add_time_cost(time_cost)
                results.append(result)

                # temporarily save results
                self.save_eval_units(results, self.save_path)
                pbar.update(1)

        return results

    def stage_interactive_extension(self, 
        target: LLMClient, 
        question_type: QTYPE, 
        syn_question_path: str, 
        load_record: bool = True, 
        fix_difficulty: str = None, 
    ) -> List[List[Dict]]:
        """
        pipeline for Stage2. Interactive Extension
        """
        # load record
        datas: List[EvalUnit] = self.load_eval_units(self.save_path)
        progress_path = os.path.join(self.save_dir, "progress_stage2.json")
        index_dict: Dict[str, int] = load_json(progress_path) if load_record and os.path.exists(progress_path) else {"extend": 0}

        # load synthetized questions
        syn_questions: List[List[Dict[str, Dict]]] = load_json(syn_question_path)

        # start extend questions and collect the target's responses
        step_name = "eval"
        index = index_dict[step_name]
        with tqdm(total=len(datas), desc=f"stage 2: {step_name}") as pbar:
            pbar.update(index)
            for data in datas[index : ]:
                extend_questions: List[Dict[str, Dict]] = syn_questions[index]  # length is the max extension rounds
                if fix_difficulty:
                    # ablation setting of no difficulty-adaptive
                    difficulty = fix_difficulty
                else:
                    difficulty = data.difficulty_controller.current_difficulty  # get the initial difficulty obtained in Benchmark Grading
                for ridx, extend_question_dict in enumerate(extend_questions):
                    start_time = time.time()
                    # get prompt
                    extend_question = extend_question_dict[difficulty]
                    questions_text, qdatas, answers = self._process_question_data([extend_question])
                    if question_type == QTYPE.RMC:
                        prompt_data = {"questions": questions_text, "article": data.article, "answers": answers}
                    else:
                        prompt_data = {"questions": questions_text, "answers": answers}
                    prompt = PROMPT_FILLER.fill(PTYPE.ANSWER, question_type, prompt_data)
                    # collect answer from the target llm
                    response = target.chat(messages=[{"role": "user", "content": prompt}])
                    llm_answers: List[str] = self._process_llm_answers(response, len(answers))
                    # validate the correctness
                    corrects = self.validate_answers_correct(answers, llm_answers, question_type)
                    # update extend question data and update the difficulty
                    datas[index].add_extend_question(qdatas[0])
                    datas[index].add_extend_answer(answers[0])
                    datas[index].add_extend_llm_answer(llm_answers[0], corrects[0], difficulty)
                    time_cost = time.time() - start_time
                    datas[index].add_time_cost(time_cost, ridx+1)

                # temporarily save results
                self.save_eval_units(datas, self.save_path)
                index += 1    
                index_dict[step_name] = index
                dump_json(index_dict, progress_path)
                
                pbar.update(1)


    def stage_evaluation_feedback(self, 
        target: LLMClient, 
        question_type: QTYPE, 
        statistic_path: str, 
        eval_after_all_extend_rounds: bool = False, 
        eval_no_extension: bool = False, 
        load_record: bool = True, 
    ) -> List[List[Dict]]:
        """
        pipeline for Stage3. Evaluation and Feedback
        """
        # load record
        datas: List[EvalUnit] = self.load_eval_units(self.save_path)
        progress_path = os.path.join(self.save_dir, "progress_stage3.json")
        index_dict: Dict[str, int] = load_json(progress_path) if load_record and os.path.exists(progress_path) else {"eval": 0, "requery": 0}

        # setting for whether to evaluate after all extend rounds
        eval_rounds: List[int] = []
        if eval_no_extension:   # ablation setting of no extension
            eval_rounds.append(0)
        else:
            if eval_after_all_extend_rounds:
                eval_rounds.extend([i+1 for i in range(self.max_extend_rounds)])
            else:
                eval_rounds.append(self.max_extend_rounds)

        # start evaluate the performance of target and generate suggestions
        step_name = "eval"
        index = index_dict[step_name]
        with tqdm(total=len(datas), desc=f"stage 3: {step_name}") as pbar:
            pbar.update(index)
            for data in datas[index : ]:
                update: bool = False
                for ridx in eval_rounds:
                    if data.get_accuracy(ridx) < 1 and data.evaluation[ridx].is_empty():
                        start_time = time.time()
                        # transform the data into formated text
                        base_questions_text = self._form_performance_to_text(data.questions, data.llm_answers[0], data.corrects[0])
                        addition_questions_text = self._form_performance_to_text(data.extend_questions[ : ridx], data.extend_llm_answers[ : ridx], data.extend_corrects[ : ridx])
                        # get prompt
                        prompt = EVALUATE_PROMPT.format(base_questions=base_questions_text, addition_questions=addition_questions_text)
                        # get evaluation
                        response = self.llm_client.chat(messages=[{"role": "system", "content": DEFAULT_EVAL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
                        evaluation = self._process_eval_response(response)
                        # save evaluation
                        time_cost = time.time() - start_time
                        datas[index].add_evaluation(evaluation, ridx)
                        datas[index].add_time_cost(time_cost, ridx)
                        update = True
                
                # temporarily save results
                if update:
                    self.save_eval_units(datas, self.save_path)
                index += 1
                index_dict[step_name] = index
                dump_json(index_dict, progress_path)

                pbar.update(1)

        # start requery the target to validate the effectiveness of suggestions
        step_name = "requery"
        index = index_dict[step_name]
        with tqdm(total=len(datas), desc=f"stage 3: {step_name}") as pbar:
            pbar.update(index)
            for data in datas[index : ]:
                for ridx in eval_rounds:
                    start_time = time.time()
                    if data.evaluation[ridx].is_empty():
                        # there is no suggestions
                        llm_answers = data.llm_answers[0]
                        corrects = data.corrects[0]
                    else:
                        # get prompt
                        batch_questions, _, batch_answers = self._process_question_data(data.questions)
                        if question_type == QTYPE.RMC:
                            prompt_data = {"questions": batch_questions, "article": data.article, "answers": batch_answers, "suggestions": data.evaluation[ridx].suggestions}
                        else:
                            prompt_data = {"questions": batch_questions, "answers": batch_answers, "suggestions": data.evaluation[ridx].suggestions}
                        prompt = PROMPT_FILLER.fill(PTYPE.REANSWER, question_type, prompt_data)
                        # collect answer from the target llm
                        response = target.chat(messages=[{"role": "user", "content": prompt}])
                        llm_answers: List[str] = self._process_llm_answers(response, len(batch_answers))
                        # validate the correctness
                        corrects = self.validate_answers_correct(batch_answers, llm_answers, question_type)

                    # save answers
                    time_cost = time.time() - start_time
                    datas[index].add_llm_answers_after_eval(llm_answers, corrects, ridx)
                    datas[index].add_time_cost(time_cost, ridx)

                # temporarily save results
                self.save_eval_units(datas, self.save_path)
                index += 1
                index_dict[step_name] = index
                dump_json(index_dict, progress_path)

                pbar.update(1)

        # statistic the results
        statistic = {
            ridx: {
                "accuracy_before_eval": {"num": 0, "total": 0, "rate": 0}, 
                "accuracy_after_eval": {"num": 0, "total": 0, "rate": 0}, 
                "correction_rate": {"num": 0, "total": 0, "rate": 0}, 
                "false_correction_rate": {"num": 0, "total": 0, "rate": 0}, 
                "time_cost": {"total": 0.0, "average": 0.0}
            } for ridx in eval_rounds
        } if not os.path.exists(statistic_path) else load_json(statistic_path)
        for data in datas:
            for ridx in eval_rounds:
                temp_statistic = self.get_statistic(data.corrects, data.corrects_after_eval[ridx])
                for k in temp_statistic.keys():
                    statistic[ridx][k]["num"] += temp_statistic[k]["num"]
                    statistic[ridx][k]["total"] += temp_statistic[k]["total"]
                statistic[ridx]["time_cost"]["total"] += data.time_cost[ridx]  
        # calculate the rate
        for ridx in eval_rounds:
            for k in statistic[ridx].keys():
                if k == "time_cost":
                    statistic[ridx][k]["average"] = statistic[ridx][k]["total"] / statistic[ridx]["accuracy_before_eval"]["total"]
                else:
                    statistic[ridx][k]["rate"] = statistic[ridx][k]["num"] / statistic[ridx][k]["total"]
        # save statistic results
        dump_json(statistic, statistic_path)
