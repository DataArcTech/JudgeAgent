# Entity Labeling
LABEL_PROMPT = {
    "Chinese": """
请为以下多个句子进行实体识别，并为每个句子返回实体标注结果。
每个句子的结果应独立，格式为JSON，包含句子编号、句子文本、和识别到的实体列表（包括实体文本、类型、和位置）。
返回格式为一个dictionary, key只有一个'labeled_data'，value为一个list，每个元素为一个dictionary，包含句子文本和实体列表。
{{
    'labeled_data': 
    [
        {{"text":"句子1", "entity_list": [{{"entity_text": "", "entity_type": ""}}]}}, 
        {{"text":"句子2", "entity_list": [{{"entity_text": "", "entity_type": ""}}]}}, 
        ...
    ]
}}
注意："text"应该只是句子，而不是整篇文章。
{annotation}
句子列表:
{sentences}
""".strip(), 

    "English": """
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
}


# Question Generation
DIFFICULTY = {
    "Chinese": {
        "low": "**鼓励知识记忆**：设计能够考察回答者是否记忆相关知识的题目，通过简单地对给定文段内容进行截取和挖空设计题目。", 
        "middle": "**鼓励知识理解**：设计能够促使回答者对题目涉及概念进行剖析和理解的题目，不要只是简单地考察知识记忆。", 
        "high": "**鼓励深度分析**：设计能够促使回答者进行深入思考和分析的题目，不要只是简单考察知识记忆、概念理解辨析，不要只是截取给定文段的片段并挖空，要能够鼓励回答者围绕题目中的实体、深入使用逻辑技巧进行复杂推理。"
    }, 
    "English": {
        "low": "**Encourage Knowledge Memorization**: Design questions that assess whether respondents have memorized relevant knowledge. Create questions by directly extracting and blanking out content from the given passage.", 
        "middle": "**Encourage Knowledge Comprehension**: Design questions that prompt respondents to dissect and comprehend concepts involved in the topic. Avoid assessing only superficial knowledge retention", 
        "high": "**Encourage Knowledge Deep Analysis**: Design questions that prompt respondents to engage in deep thinking and analysis. Avoid merely testing knowledge recall or conceptual comprehension; do not simply extract fragments from the given passage to create fill-in-the-blank items. Encourage respondents to focus on entities within the question and employ logical skills for complex reasoning."
    }
}


SOG_PROMPT_CHOICE = {
    "Chinese": """
作为出题人，您需要根据所提供的文段内容，设计单项选择题式的问答题目。您的职责是编写符合以下标准的问题、相关的选项及正确的答案：

1. **聚焦实体**：所有问题都应围绕给定文段中的实体展开，但题面中不能出现诸如“根据文段”、“按照给定文段”等显式要求学生开卷作答的提示语。
2. **合理相关的选项**：根据您编写的题面，您提供的选项中必须包含一个符合给定文段内容的正确选项，并且保证其他提供的选项与正确选项是相似类型、不同内容的错误选项，同时保证正确选项的位置是随机的。
{difficulty}
**样例**：
- **文本片段**：
文本片段 1：门静脉、肝动脉小分支之间的交通支在门静脉高压症发病中的作用(l)正常时，门静脉、肝动脉小分支分别流入肝窦，它们之间的交通支细而不开放(2)肝硬化时，交通支开放，压力高的肝动脉血流注入压力低的门静脉，从而使门静脉高压进一步增高肝后型门静脉高压症的常见病因包括巴德－吉亚利综合征(Budd-Chiari syndrome汃缩窄性心包炎、严重右心衰竭等。
文本片段 2：上述各种情况引起门静脉高压持续存在后，可发生下列病理变化：1脾大(splenomegaly)、脾功能亢进(hypersplenism)门静脉压力升高后，脾静脉血回流受阻，脾窦扩张，脾髓组织增生，脾脏肿大。
- **生成问题**：
{{
    "generated_question": {{
        "question": "肝硬化时，脾肿大的主要原因是（ ）。", 
        "options": {{"A": "脾窦巨噬细胞增多", "B": "脾索纤维组织增生", "C": "脾窦扩张红细胞淤滞", "D": "淋巴小结内大量中性粒细胞浸润"}}, 
        "answer": "C"
    }}
}}

请你根据上述要求，参考样例的生成格式，在给定文本片段后，按照指定格式生成单项选择题式的问答题目，以如下JSON格式输出。
- **文本片段**：
{context}
- **生成问题**:
{{
    "generated_question": {{
        "question": "Generated Question", 
        "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}}, 
        "answer": "Correct Option Index"
    }}
}}
""".strip(), 
    "English": """
As an interviewer, you are tasked with designing multiple-choice style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Plausibly Related Options**: Based on the question you design, the options must include one correct choice aligned with the given passage content. Additionally, ensure all other provided options are incorrect alternatives of the same category but differing content, while randomizing the position of the correct answer.
{difficulty}
**Example**: 
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
}


SOG_PROMPT_SIMPLE_QA = {
    "Chinese": """
作为出题人，您需要根据所提供的文段内容，设计短语问答式的问答题目。您的职责是编写符合以下标准的问题、相关的选项及正确的答案：

1. **聚焦实体**：所有问题都应围绕给定文段中的实体展开，但题面中不能出现诸如“根据文段”、“按照给定文段”等显式要求学生开卷作答的提示语。
2. **确保答案的准确与精简**：请确保您提供的答案在给定文段内容及逻辑下，是符合您设计题目的正确答案，并确保答案足够精简、不冗长、以词语或短语的形式呈现。
{difficulty}
**样例**：
- **文本片段**：
文本片段 1：纽约州法官作出重大否定性裁决，裁定前总统唐纳德·特朗普通过虚构估值手段欺诈银行以构建其房地产帝国。
文本片段 2：检方指控称，此举意在掩盖其名下另一处资产的贬值事实。
- **生成问题**：
{{
    "generated_question": {{
        "question": "哪位被指虚抬曼哈顿某公寓估值至纽约市房地产史上前所未有的高度？此人还被控通过调整该公寓估值来抵偿另一项资产的亏损？",  
        "answer": "唐纳德·特朗普"
    }}
}}

请你根据上述要求，参考样例的生成格式，在给定文本片段后，按照指定格式生成短语问答式的问答题目，以如下JSON格式输出。
- **文本片段**：
{context}
- **生成问题**:
{{
    "generated_question": {{
        "question": "Generated Question", 
        "answer": "Correct Answer"
    }}
}}
""".strip(), 
    "English": """
As an interviewer, you are tasked with designing phrase-based Q&A style questions based on the provided articles. Your role involves crafting questions, relevant options, and correct answers that fulfill the following criteria:

1. **Focus on the Entity**: Ensure all questions consistently center around the specified entity from the article. 
2. **Ensure Accuracy and Conciseness of Answers**: Verify that the provided answer is both correct for your designed question within the context and logic of the given text fragments, and ensure the answer is sufficiently concise—presented as a word or phrase, avoiding redundancy.
{difficulty}
**Example**: 
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
}


# Answer Question
### answer questions
ANSWER_PROMPT_MC = {
    "Chinese": """
完成下列单项选择题：
{question}
{options}
请从上述选项中选择正确的一项，按照下列格式输出，其中<your choice>为选项的序号（A/B/C/D/……）：
[Answer]: <your choice>
""".strip(), 
    "English": """
Please complete the following multiple-choice question:
{question}
{options}
The question has only one correct option. Please ONLY output the index (A/B/C/D) of the correct answer in the following JSON format:
{{
    "answer": "correct choice index"
}}
""".strip()
}


ANSWER_PROMPT_QA = {
    "Chinese": """
完成下列短语问答题：
{question}
请尽可能精简地使用一个词语或短语回答上述问题，按照下列格式输出：
[Answer]: <your answer>
""".strip(), 
    "English": """
Please complete the following phrase Q&A question:
{question}
Please answer the above question as minimally as possible using one word or phrase, in the following JSON format:
{{
    "answer": "Your answer"
}}
""".strip()
}


### Re-answer questions with suggestions from JudgeAgent
REANSWER_PROMPT_MC = {
    "Chinese": """
现有下列单项选择题：
{question}
{options}

在之前针对该题以及相关知识的衍生题目的回答中，面试老师对你的建议如下：
{suggestions}

请你适当参考如上建议，回答给出的单项选择题。请你从上述选项中选择正确的一项，按照下列格式输出，其中<your choice>为选项的序号（A/B/C/D/……）：
[Answer]: <your choice>
""".strip(), 
    "English": """
Please complete the following multiple-choice question:
{question}
{options}

In your previous responses to the question, the interviewer has provided the following suggestions for you to help you answer better:
[suggestions]: {suggestions}

The question has only one correct option. Please consider the above suggestions, and output the index (A/B/C/D) of the correct answer in the following JSON format:
{{
    "answer": "correct choice index"
}}
""".strip()
}


REANSWER_PROMPT_QA = {
    "Chinese": """
完成下列短语问答题：
{question}

在之前针对该题以及相关知识的衍生题目的回答中，面试老师对你的建议如下：
{suggestions}

请你适当参考如上建议，尽可能精简地使用一个词语或短语回答上述问题，按照下列格式输出：
[Answer]: <your answer>
""".strip(), 
    "English": """
Please complete the following phrase Q&A question:
[question]: {question}

In previous responses to this question and related derivative questions, the interviewer has provided the following suggestions for you:
[suggestions]: {suggestions}

Please consider the above [suggestions], and answer the above [question] as minimally as possible using one word or phrase, in the following JSON format:
{{
    "answer": "Your answer"
}}
""".strip()
}


VALIDATE_PROMPT = {
    "Chinese": """
下列是一个问题及其正确答案，以及一位受面试者对该问题的回答：
[问题]：{question}
[正确答案]：{correct_answer}
[受面试者回答]：{candidate_answer}
参考[正确答案]，请问[受面试者回答]对于[问题]来说是否正确？请按照如下JSON格式，仅输出YES或者NO：
{{
    "judge": <YES or NO>
}}
""".strip(), 
    "English": """
The following is a question with the correct answer, and the response of a candate in answering this question: 
[question]: {question}
[correct answer]: {correct_answer}
[candidate answer]: {candidate_answer}
With reference to [correct answer], is [candidate answer] correct for [question]? Please output only YES or NO in the following JSON format:
{{
    "judge": <YES or NO>
}}
""".strip()
}


# Evaluation && Suggestions
EVAL_SYSTEM_PROMPT = {
    "Chinese": "你是一位面试官，需要根据学生在一连串问题中的回答表现，评定该学生的知识水平、能力层次。", 
    "English": "You are an examiner who needs to assess the student's knowledge level and ability tier based on their performance in answering a sequence of questions."
}


EVALUATE_PROMPT = {
    "Chinese": """
以下是一名受面试者关于一道基础问题及其相关衍生问题的回答表现:
[基础问答]: {base_question}

[衍生问答列表]: [
{addition_questions}
]

请根据上述表现，从以下角度用简洁语言对受访者的表现进行评估分析，并提供有助于LLM更好回答同类问题的建议。提供的改正建议应围绕逻辑思维步骤、必备知识及能力提供具体详细的指导，确保LLM能针对同类问题作出正确回答。
按照如下JSON格式输出: 
{{
    "lack_of_knowledge": "是否存在知识方面的欠缺？若存在欠缺，需要补充哪些知识？", 
    "lack_of_ability": "是否存在能力方面的欠缺？若存在欠缺，哪些能力需要加强？", 
    "comprehensive_performance": "模型回答问题的综合表现", 
    "suggestions": "能够帮助模型更好回答问题的建议。"
}}
""".strip(), 
    "English": """
The following is the performance of an LLM in answering a base question and its related derivative questions: 
[base question]: {base_question}

[derivative question list]: [
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
}