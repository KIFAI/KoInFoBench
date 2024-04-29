import argparse
import json
import logging
import os
import time
from functools import partial

import requests
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import google.generativeai as genai


load_dotenv()


class ModelClient:
    """
    A class for managing and utilizing multiple AI APIs for text generation tasks.
    This class abstracts the complexity of interfacing with different AI model APIs
    and provides a unified method to generate text based on given instructions.
    """

    def __init__(self, init_api='all', **kwargs):
        logging.info(f"Initializing ModelClient with API setting: {init_api}")
        self.available_api_list = ['all', 'openai', 'caude', 'gemini', 'hpx003']
        self.clients = {}
        self._init_api_clients(init_api)

    def _init_api_clients(self, init_api):
        if init_api not in self.available_api_list:
            logging.error(f"`init_api` should be one of {self.available_api_list}")
            raise ValueError(f"`init_api` should be one of {self.available_api_list}")
        
        api_client_initializers = {
            'openai': partial(OpenAI, api_key=os.getenv('OPENAI_API_KEY')),
            'claude': partial(Anthropic, api_key=os.getenv('CLAUDE_API_KEY')),
            'gemini': partial(Gemini, api_key=os.getenv('GEMINI_API_KEY')),
            'hpx003': partial(HyperClovaX,
                              api_key=os.getenv('HYPERCLOVAX_API_KEY'),
                              api_key_primary_eval=os.getenv('HYPERCLOVAX_API_KEY_PRIMARY_EVAL'))
        }

        for api, initializer in api_client_initializers.items():
            if init_api == 'all' or init_api == api:
                self.clients[api] = initializer()

    def generate(self, model_name, **kwargs):
        instruction = kwargs.pop("instruction", "").strip()
        inputs = kwargs.pop("input", "").strip()
        content = self._make_instructions(instruction, inputs)
        api = 'openai' if model_name.startswith('gpt-') else model_name.split('-')[0]
        if api in self.clients:
            return self._generate_content(api, model_name, content, **kwargs)
        else:
            logging.error(f"Unsupported `model_name` prefix. Should be one of: {self.available_api_list[1:]}")
            raise ValueError(f"Unsupported `model_name` prefix. Should be one of: {self.available_api_list[1:]}")

    def _generate_content(self, api: str, model_name: str, content: str, **kwargs):
        generation_methods = {
            'openai': partial(self._generate_openai),
            'claude': partial(self._generate_claude),
            'gemini': partial(self._generate_gemini),
            'hpx003': partial(self._generate_hyperclovax)
        }
        result = generation_methods[api](model_name, content, **kwargs)
        result['input'] = content
        return result
        
    def _make_instructions(self, instruction, inputs="") -> str:
        content = f"{instruction}\n\n{inputs}" if inputs else instruction
        if not content:
            raise ValueError("No instructions text provided")
        return content

    def _generate_openai(self, model_name, content, **kwargs):
        client = self.clients['openai']
        completion = client.chat.completions.create(
            model=model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
            **kwargs
        )
        
        result = {
            'output': completion.choices[0].message.content,
            'usage': {
                'input_tokens': completion.usage.prompt_tokens, 
                'outupt_tokens': completion.usage.completion_tokens,
                },
            }
        return result

    def _generate_claude(self, model_name, content, **kwargs):
        client = self.clients['claude']
        completion = client.messages.create(
            model=model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
            **kwargs
        )

        result = {
            'output': completion.content[0].text,
            'usage': {
                'input_tokens': completion.usage.input_tokens,
                'output_tokens': completion.usage.output_tokens,
            },
        }
        return result

    def _generate_gemini(self, model_name, content, **kwargs):
        client = self.clients['gemini']
        # https://github.com/google/generative-ai-python/issues/170
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        completion = client.generate_content(contents=content, safety_settings=safety_settings, **kwargs)
        output = ''.join([part.text for part in completion.candidates[0].content.parts])
        print(output)
        input_tokens = client.count_tokens(contents=content).total_tokens
        output_tokens = client.count_tokens(contents=output).total_tokens

        result = {
            'output': output,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            },
        }
        return result

    def _generate_hyperclovax(self, model_name, content, **kwargs):
        client = self.clients['hpx003']
        completion = client.chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            **kwargs
        )
        
        result = {
            'output': completion['message']['content'],
            'usage': {
                'input_tokens': completion['inputLength'],
                'output_tokens': completion['outputLength'],
            }
        }
        return result
        

class Gemini(genai.GenerativeModel):
    """
    A specialized class for interfacing with the Gemini AI model.
    Extends the GenerativeModel class to utilize specific functionalities of the Gemini API.

    Initialization of this class is dependent on a valid API key which is mandatory for operation.
    """

    def __init__(self, model_name=None, **kwargs):
        super().__init__(model_name='gemini-1.5-pro-latest' if not model_name else model_name)
        self._api_key = kwargs.get('api_key')
        if not self._api_key:
            raise ValueError("Missing required arguments: 'api_key'")
        global genai
        genai.configure(api_key=self._api_key)
        

class HyperClovaX:
    """
    A class designed to interact with the HyperClovaX for generating chat completions.
    It manages the communication with the HyperClovaX API, handling token-based request and response.
    """

    def __init__(self, **kwargs):
        self._api_key = kwargs.get('api_key')
        self._api_key_primary_eval = kwargs.get('api_key_primary_eval')
        if not self._api_key or not self._api_key_primary_eval:
            raise ValueError("Missing required arguments: 'api_key', 'api_key_primary_eval'")

        self._headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_eval,
            'Content-Type': 'application/json; charset=utf-8',
        }
        self._host = 'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003'
    
    def chat_completion(self, messages, max_tokens=2048, retry=3, **kwargs):
        request_data = {'messages': messages, 'maxTokens': max_tokens, **kwargs}
        for attempt in range(retry):
            try:
                with requests.post(self._host, headers=self._headers, json=request_data, stream=False) as r:
                    if r.status_code != 200 and r.json()['status'] != "20000":
                        r.raise_for_status()
                    completion = r.json()['result']
                    return completion
            except requests.HTTPError:
                time.sleep(3)
                logging.warning(f"[{attempt + 1}/{retry}] Retry requesting chat completion on HyperClovaX-003")
        logging.error("Failed to obtain completion after multiple retries.")
        raise Exception("Failed to obtain completion after multiple retries.")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = load_dataset('kifai/KoInFoBench')['train']
    client = ModelClient(init_api=args.api_type)

    for model_name in args.models:
        output_path = os.path.join(args.output_dir, f'generations_{model_name}.jsonl')
        if os.path.exists(output_path):
            logging.info(f"File already exists: '{output_path}', skipping generation.")
            continue
        
        generated_results = []
        for datum in tqdm(datasets, desc=f"Generating responses of '{model_name}': ", ncols=100, total=len(datasets)):
            content = {"instruction": datum['instruction'].strip(), "input": datum['input'].strip()}
            result = client.generate(model_name=model_name, **content)
            result['id'] = datum['id']
            result['model_name'] = model_name
            generated_results.append(result)
            time.sleep(2)
            
        with open(output_path, 'w') as w:
            for result in generated_results:
                w.write(json.dumps(result, ensure_ascii=False) + "\n")
        logging.info(f"Results successfully written to '{output_path}'.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model outputs based on provided instructions and inputs.")
    parser.add_argument("--api_type", type=str, default="all", help="API Client to initialize ('all', 'openai', 'claude', 'gemini', 'hpx003').")
    parser.add_argument("--output_dir", type=str, default='generations', help="Directory where the output files will be saved.")
    parser.add_argument("--models", type=str, required=True, action='append', help="Specify one or more model names. This option can be repeated. e.g. --models `gpt-3.5-turbo-preview`")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.output_dir, 'generations.log'), filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
    logging.info("Script finished.")
