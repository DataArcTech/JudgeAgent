# Entity Labeling
# args: annotation, sentences
LABEL_PROMPT = """
Please identify and label the entities in the following multiple sentences, and return the entity labeling results for each sentence.
The results for each sentence should be independent, in JSON format, containing the sentence text, and the list of recognized entities (including entity text, type, and position).
{annotation}
Return format is a dictionary, with only one key 'labeled_data', and the value is a list, each element is a dictionary containing the sentence text and the entity list.
{{
    'labeled_data':             
    [
        {{"text":"Sentence 1", "entity_list": [{{"entity_text": "", "entity_type": ""}}]}}, 
        {{"text":"Sentence 2", "entity_list": [{{"entity_text": "", "entity_type": ""}}]}}, 
        ...
    ]
}}
Sentence list:
{sentences}
""".strip()


# Question Generation
DIFFICULTY = {  # the rules for different difficulty levels
    "easy": "**Encourage Knowledge Memorization**: Design questions that assess whether respondents have memorized relevant knowledge. Create questions by directly extracting and blanking out content from the given passage.", 
    "medium": "**Encourage Knowledge Comprehension**: Design questions that prompt respondents to dissect and comprehend concepts involved in the topic. Avoid assessing only superficial knowledge retention", 
    "hard": "**Encourage Knowledge Deep Analysis**: Design questions that prompt respondents to engage in deep thinking and analysis. Avoid merely testing knowledge recall or conceptual comprehension; do not simply extract fragments from the given passage to create fill-in-the-blank items. Encourage respondents to focus on entities within the question and employ logical skills for complex reasoning."
}

# default example for multiple-choice questions generation
DEFAULT_EXAMPLE_MC = """
- **Text Fragments**: 
Text Fragment1: Respiratory burst Also called oxidative burst. Involves the activation of the phagocyte NADPH oxidase complex (eg, in neutrophils, monocytes), which utilizes O2 as a substrate. Plays an important role in the immune response  rapid release of reactive oxygen species (ROS). NADPH plays a role in both the creation and neutralization of ROS. Myeloperoxidase contains a blue-green, heme-containing pigment that gives sputum its color.
- **Generated Questions**:
{{
    "generated_question": {{
        "question": "A 5-year-old female suffers from recurrent infections by Aspergillus species, Pseudomonas species, and Staphylococcus aureus. The patient's neutrophils are examined in the laboratory and they fail to react during the nitroblue tetrazolium test. Which of the following is most likely dysfunctional in this patient?", 
        "options": {{"A": "Immunoglobulin class switching", "B": "Superoxide dismutase", "C": "Myeloperoxidase", "D": "Respiratory burst"}}, 
        "answer": "D"
    }}
}}
""".strip()

# default example for phrase question-answer questions generation
DEFAULT_EXAMPLE_QA = """
- **Text Fragments**: 
Text Fragment1: Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president.
Text Fragment2: The prosecution argues that was to mask a drop in the value of one of his other properties.
- **Generated Questions**:
{{
    "generated_question": {{
        "question": "Which individual is implicated in both inflating the value of a Manhattan apartment to a figure not yet achieved in New York City's real estate history, and is also accused of adjusting this apartment's valuation to compensate for a loss in another asset's worth?", 
        "answer": "Donald Trump"
    }}
}}
""".strip()


# prompt for synthetising multiple-choice (MC) questions on graph.
# args: difficulty (rules from DIFFICULTY), example, context
SOG_PROMPT_MC = """
As an interviewer, you are tasked with designing multiple-choice style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Plausibly Related Options**: Based on the question you design, the options must include one correct choice aligned with the given passage content. Additionally, ensure all other provided options are incorrect alternatives of the same category but differing content, while randomizing the position of the correct answer.
{difficulty}
**Example**: 
{example}

Please generate a multiple-choice question according to the above requirements, referring to the sample format, and based on the given text fragments, output in the following specified JSON format:
- **Text Fragments**: 
{context}
- **Generated Questions**:
{{
    "generated_question": {{
        "question": "Generated Question", 
        "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}}, 
        "answer": "Correct Option Index"
    }}
}}
""".strip()

# prompt for synthetising phrase question-answer (QA) questions on graph
# args: difficulty (rules from DIFFICULTY), example, context
SOG_PROMPT_QA = """
As an interviewer, you are tasked with designing phrase-based Q&A style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Ensure Accuracy and Conciseness of Answers**: Verify that the provided answer is both correct for your designed question within the context and logic of the given text fragments, and ensure the answer is sufficiently concise—presented as a word or phrase, avoiding redundancy.
{difficulty}
**Example**: 
{example}

Please generate a phrase-based Q&A question according to the above requirements, referring to the sample format, and based on the given text fragments, output in the following specified JSON format:
- **Text Fragments**: 
{context}
- **Generated Questions**:
{{
    "generated_question": {{
        "question": "Generated Question", 
        "answer": "Correct Answer"
    }}
}}
""".strip()


# prompt for synthetising multiple-choice (MC) questions on graph.
# allowing for synthetising all questions with all difficulty levels.
# args: example, context
SOG_FULL_DIFFICULTY_PROMPT_MC = """
As an interviewer, you are tasked with designing multiple-choice style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Plausibly Related Options**: Based on the question you design, the options must include one correct choice aligned with the given passage content. Additionally, ensure all other provided options are incorrect alternatives of the same category but differing content, while randomizing the position of the correct answer.
3. **Conform to difficulty requirements**: You need to design a question each for the [easy], [medium], and [hard] difficulty levels, with specific requirements as follows:
    (1). [easy]: **Encourage Knowledge Memorization**: Design questions that assess whether respondents have memorized relevant knowledge. Create questions by directly extracting and blanking out content from the given passage. 
    (2). [medium]: **Encourage Knowledge Comprehension**: Design questions that prompt respondents to dissect and comprehend concepts involved in the topic. Avoid assessing only superficial knowledge retention.
    (3). [hard]: "**Encourage Knowledge Deep Analysis**: Design questions that prompt respondents to engage in deep thinking and analysis. Avoid merely testing knowledge recall or conceptual comprehension; do not simply extract fragments from the given passage to create fill-in-the-blank items. Encourage respondents to focus on entities within the question and employ logical skills for complex reasoning.

**Example**: 
{example}

Please generate a multiple-choice question according to the above requirements, referring to the sample format, and based on the given text fragments, output in the following specified JSON format:
- **Text Fragments**: 
{context}
- **Generated Questions**:
{{
    "generated_question": {{
        "easy": {{
            "question": "Generated question-easy level", 
            "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}}, 
            "answer": "Correct Option Index"
        }}, 
        "medium": {{
            "question": "Generated question-medium level", 
            ...
        }}, 
        "hard": ...
    }}
}}
""".strip()

# prompt for synthetising phrase question-answer (QA) questions on graph
# allowing for synthetising all questions with all difficulty levels.
# args: example, context
SOG_FULL_DIFFICULTY_PROMPT_QA = """
As an interviewer, you are tasked with designing phrase-based Q&A style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Ensure Accuracy and Conciseness of Answers**: Verify that the provided answer is both correct for your designed question within the context and logic of the given text fragments, and ensure the answer is sufficiently concise—presented as a word or phrase, avoiding redundancy.
3. **Conform to difficulty requirements**: You need to design a question each for the [easy], [medium], and [hard] difficulty levels, with specific requirements as follows:
    (1). [easy]: **Encourage Knowledge Memorization**: Design questions that assess whether respondents have memorized relevant knowledge. Create questions by directly extracting and blanking out content from the given passage. 
    (2). [medium]: **Encourage Knowledge Comprehension**: Design questions that prompt respondents to dissect and comprehend concepts involved in the topic. Avoid assessing only superficial knowledge retention.
    (3). [hard]: "**Encourage Knowledge Deep Analysis**: Design questions that prompt respondents to engage in deep thinking and analysis. Avoid merely testing knowledge recall or conceptual comprehension; do not simply extract fragments from the given passage to create fill-in-the-blank items. Encourage respondents to focus on entities within the question and employ logical skills for complex reasoning.

**Example**: 
{example}

Please generate a phrase-based Q&A question according to the above requirements, referring to the sample format, and based on the given text fragments, output in the following specified JSON format:
- **Text Fragments**: 
{context}
- **Generated Questions**:
{{
    "generated_question": {{
        "question": "Generated Question", 
        "answer": "Correct Answer"
    }}
}}
""".strip()

# Answer Question
# prompt for answering multiple-choice questions
# args: questions
ANSWER_PROMPT_MC = """
Please complete the following multiple-choice question:
[questions]: [
{questions}
]

The question has only one correct option. Please ONLY output the index (A/B/C/D) of the correct answer for each question in [questions].
Please output in the following JSON format:
{{
    "response": [
        {{"answer": "choice of question 1"}}, 
        {{"answer": "choice of question 2"}}, 
        ...
    ]
}}
""".strip()

# prompt for answering phrase Q&A questions
# args: questions
ANSWER_PROMPT_QA = """
Please complete the following phrase Q&A questions:
[questions]: [
{questions}
]

Please answer each question in [questions] as minimally as possible using one word or phrase.
Please output in the following JSON format:
{{
    "response": [
        {{"answer": "answer of question 1"}}, 
        {{"answer": "answer of question 2"}}, 
        ...
    ]
}}
""".strip()

# prompt for answering multiple-choice questions with backgroun article (mainly assess the reasoning and comphrension capability)
# args: article, questions
ANSWER_PROMPT_RMC = """
Please answer some multiple-choice style questions based on the following provided article:

{article}

Please answer the following multiple-choice questions based on the above article: [
{questions}
]

Each question has only one correct option. Please ONLY output the index (A/B/C/D) of the correct answer for each question in the following JSON format:
{{
    "response": [
        {{"answer": "choice of question 1"}}, 
        {{"answer": "choice of question 2"}}, 
        ...
    ]
}}
""".strip()


# Re-answer questions with suggestions from JudgeAgent
# prompt for re-answer multiple-choice questions
# args: questions, suggestions
REANSWER_PROMPT_MC = """
Please complete the following multiple-choice questions:
[questions]: [
{questions}
]

In your previous responses to the question, the interviewer has provided the following suggestions for you to help you answer better:
[suggestions]: {suggestions}

The question has only one correct option. Please consider the above suggestions, and ONLY output the index (A/B/C/D) of the correct answer for each question in [questions].
Please output in the following JSON format:
{{
    "response": [
        {{"answer": "choice of question 1"}}, 
        {{"answer": "choice of question 2"}}, 
        ...
    ]
}}
""".strip()

# prompt for re-answer phrase Q&A questions
# args: questions, suggestions
REANSWER_PROMPT_QA = """
Please complete the following phrase Q&A questions:
[questions]: [
{questions}
]

In previous responses to this question and related derivative questions, the interviewer has provided the following suggestions for you:
[suggestions]: {suggestions}

Please consider the above [suggestions], and answer each question in [questions] as minimally as possible using one word or phrase.
Please output in the following JSON format:
{{
    "response": [
        {{"answer": "answer of question 1"}}, 
        {{"answer": "answer of question 2"}}, 
        ...
    ]
}}
""".strip()

# prompt for re-answer multiple-choice questions with backgroun article (mainly assess the reasoning and comphrension capability)
# args: article, questions, suggestions
REANSWER_PROMPT_RMC = """
Please answer some multiple-choice style questions based on the following provided article:

{article}

Please answer the following multiple-choice questions based on the above article: [
{questions}
]

In your previous responses to these questions, the interviewer has provided the following suggestions for you to help you answer better:
[suggestions]: {suggestions}

Each question has only one correct option. Please consider the above suggestions, optimize your reasoning and analysis process, and output the index (A/B/C/D) of the correct answer for each question in the following JSON format:
{{
    "response": [
        {{"answer": "choice of question 1"}}, 
        {{"answer": "choice of question 2"}}, 
        ...
    ]
}}
""".strip()

# prompt for validating the phrase answer's correctness
VALIDATE_PROMPT = """
Given a list of answer pairs in format (correct_answer, candidate_answer):
{answer_list}
For each pair, does candidate_answer represent the same meaning with correct_answer?
For example, If input:
[("no", "No significant change"), ("OpenAI", "Meta AI")]
then output:
{{"judge": [YES, NO]}} 
Please output only YES or NO in the following JSON format:
{{
    "judge": [<YES/NO for pair 1>, <YES/NO for pair 2>, ...]
}}
""".strip()


# Evaluation && Suggestions
# default system prompt of JudgeAgent evaluator
DEFAULT_EVAL_SYSTEM_PROMPT = "You are an examiner who needs to assess the student's knowledge level and ability tier based on their performance in answering a sequence of questions."

# prompt for evaluating the model's performance based on whole multi-turn Q&A history
# args: base_questions, addition_questions
EVALUATE_PROMPT = """
The following is the performance of an LLM in answering base questions and their related derivative questions: 
[base questions]: [
{base_questions}
]

[derivative questions]: [
{addition_questions}
]

Please evaluate and analyze the interviewee’s performance based on the above performance using concise language from the following perspectives, and provide suggestions that help the LLM answer the same questions better. Suggestions should provide specific and detailed guidance on logical thinking steps, required knowledge, and abilities, ensuring the LLM can answer correctly for the same questions.
Output in the following JSON format:
{{
    "lack_of_knowledge": "The lack of background knowledge.", 
    "lack_of_ability": "The flaws in logic and capability.", 
    "comprehensive_performance": "The Comprehensive performance of all questions.", 
    "suggestions": "Suggestions that help the LLM answer questions better."
}}
""".strip()



# Dictionary of mapping from QTYPE (question type) to prompt
from enum import Enum
from typing import Dict
from .utils import QTYPE


class PTYPE(Enum):
    SOG = "sythetisis_on_graph"
    SOG_FULL_DIFFICULTY = "sog_with_all_difficulty"
    ANSWER = "answer"
    REANSWER = "answer_with_suggestions"


class PromptFiller:
    def __init__(self) -> None:
        self.prompt_dict = {
            PTYPE.SOG: {
                QTYPE.KMC: SOG_PROMPT_MC, 
                QTYPE.KQA: SOG_PROMPT_QA, 
                QTYPE.RMC: SOG_PROMPT_MC
            }, 
            PTYPE.SOG_FULL_DIFFICULTY: {
                QTYPE.KMC: SOG_FULL_DIFFICULTY_PROMPT_MC, 
                QTYPE.KQA: SOG_FULL_DIFFICULTY_PROMPT_QA, 
                QTYPE.RMC: SOG_FULL_DIFFICULTY_PROMPT_MC
            }, 
            PTYPE.ANSWER: {
                QTYPE.KMC: ANSWER_PROMPT_MC, 
                QTYPE.KQA: ANSWER_PROMPT_QA, 
                QTYPE.RMC: ANSWER_PROMPT_RMC
            }, 
            PTYPE.REANSWER: {
                QTYPE.KMC: REANSWER_PROMPT_MC, 
                QTYPE.KQA: REANSWER_PROMPT_QA, 
                QTYPE.RMC: REANSWER_PROMPT_RMC
            }
        }
        self.default_example_dict = {
            QTYPE.KMC: DEFAULT_EXAMPLE_MC, 
            QTYPE.KQA: DEFAULT_EXAMPLE_QA, 
            QTYPE.RMC: ""
        }

    def fill(self, prompt_type: PTYPE, question_type: QTYPE, data: Dict) -> str:
        prompt_template = self.prompt_dict[prompt_type][question_type]
        if prompt_type in [PTYPE.SOG, PTYPE.SOG_FULL_DIFFICULTY]:
            example = data.get("example", self.default_example_dict[question_type])
            context = data["context"]
            difficulty = data.get("difficulty", None)
            prompt = prompt_template.format(difficulty=difficulty, example=example, context=context)
        elif prompt_type in [PTYPE.ANSWER, PTYPE.REANSWER]:
            article = data.get("article", None)
            questions = data["questions"]
            suggestions = data.get("suggestions", None)
            prompt = prompt_template.format(article=article, question=questions, suggestions=suggestions)
        else:
            raise ValueError(f"Prompt of type {prompt_type} is not supported.")
        
        return prompt

PROMPT_FILLER = PromptFiller()