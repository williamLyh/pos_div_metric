import os
import time
from typing import List, Dict
from openai import OpenAI
import openai
import json

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_openai_chat_completion(prompt: str,
                                model='gpt-3.5-turbo',
                                temperature=0.5,
                                max_tokens=128,
                                wait_sec=0.3,
                                max_wait_sec=0.3,
                                **kwargs) -> str:
    client = OpenAI()
    msg = [
        {'role': 'system',
        'content': "You are a helpful assistant. Please follow the user's instructions and examples' format."},
        {'role': 'user', 'content': prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        llm_output = response.choices[0].message.content.strip()
        # print(llm_output)
        llm_output = json.loads(llm_output)
    except Exception as e:
        print(e)
        llm_output = None

    # while True:
    #     try:
    #         response = client.chat.completions.create(
    #             model=model,
    #             messages=msg,
    #             temperature=temperature,
    #             max_tokens=max_tokens,
    #             **kwargs
    #         )
    #         llm_output = response.choices[0].message.content.strip()
    #         llm_output = json.loads(llm_output)
            # break
        # except Exception as e:
        #     print(e)
        #     wait_sec = 1
        #     time.sleep(wait_sec)
    return llm_output
