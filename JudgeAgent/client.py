import time
import openai
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict

from .llm_params import LLMParams



class LLMClient:
    def __init__(self, params: LLMParams) -> None:
        self.api_key = params.api_key
        self.base_url = params.base_url
        self.model = params.model
        self.json_format = params.json_format
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_llm_params(self):
        return LLMParams(self.api_key, self.base_url, self.model)

    def chat(self, 
        messages: List[Dict[str, str]], 
        **kwargs   
    ) -> str:
        get_res, try_cnt, error = False, 0, None

        if self.json_format:
            kwargs["response_format"] = {"type": "json_object"}

        res = None
        while not get_res and try_cnt < 3:
            try_cnt += 1
            try:
                completion: ChatCompletion = self.client.chat.completions.create(
                    messages=messages, 
                    model=self.model, 
                    **kwargs
                )
                res = completion.choices[0].message.content
                get_res = res is not None
            except Exception as e:
                error = e
                err_msg = str(error)
                if any([s in err_msg for s in ["You exceeded your current requests list", "您当前使用该API的并发数过高"]]):
                    time.sleep(30)
                    try_cnt = 1
                elif any([s in err_msg for s in ["data_inspection_failed", "系统检测到输入或生成内容可能包含不安全或敏感内容"]]):
                    time.sleep(5)
                    get_res = True
                    res = f"[ERROR]: {err_msg}"
                else:
                    time.sleep(5)
        
        if not get_res:
            if error is not None:
                raise error
            else:
                raise Exception("try to max.")

        if get_res and not res.startswith("[ERROR]"):
            if res.startswith("```json"):
                res: str = res[7:]
            if res.endswith("```"):
                res: str = res[:-3]
            res = res.strip()
            
        return res

    def get_embeddings(self, 
        inputs: List[str], 
        dimension: int, 
        encoding_format: str = "float"
    ) -> List[List[float]]:
        completion = self.client.embeddings.create(
            model=self.model, 
            input=inputs, 
            dimensions=dimension, 
            encoding_format=encoding_format
        )
        result = completion.model_dump()
        embeddings = [d["embedding"] for d in result["data"]]

        return embeddings



