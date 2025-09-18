'''
File for the parameters of LLMs 
'''


class LLMParams:
    def __init__(self, 
        api_key: str, 
        base_url: str, 
        model: str, 
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


MODEL_PARAMS = {
    # LLM for chat
    "qwen3": LLMParams(
        api_key="YOUR API KEY", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
        model="qwen-plus-2025-04-28"
    ), 
    "glm4-flash": LLMParams(
        api_key="YOUR API KEY", 
        base_url="https://open.bigmodel.cn/api/paas/v4/", 
        model="glm-4-flash", 
    ), 
    "gpt": LLMParams(
        api_key="YOUR API KEY", 
        base_url="BASE URL OF GPT", 
        model="gpt-4.1"
    ), 
    "gemini": LLMParams(
        api_key="YOUR API KEY", 
        base_url="BASE URL OF GEMINI", 
        model="gemini-2.5-pro-preview-06-05"
    ), 
    # LLM for embedding
    "qwen3-embedding": LLMParams(
        api_key="YOUR API KEY", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
        model="text-embedding-v4"
    ),
}