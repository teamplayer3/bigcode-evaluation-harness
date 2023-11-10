"""Testing
from datasets import load_dataset

ds = load_dataset("bigcode/humaneval-x-bugs", "python")["test"]
idx = 0

def get_prompt_base(doc, language="python"):
    # See 
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"] + doc["prompt"]
    else:
        prompt_base = doc["prompt"]
    return prompt_base

prompt_base = get_prompt_base(ds[idx], language="python")
    
messages = [
    {
        "role": "user",
        "content": ds[idx]["instruction"],
    },
    {
        "role": "assistant",
        "content": prompt_base,
    },
]

gpt-4-0613
response = openai.ChatCompletion.create(
model="gpt-4-0613",
messages=messages
)
"""

import os
import openai
import jsonlines
import termcolor

from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from datasets import load_dataset
from typing import List
from tqdm import tqdm

_CITATION = """
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
"""

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}


def get_prompt_base(doc, language):
    # See
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    return prompt_base


def get_prompt_synthesize(doc, language="python"):
    # addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    # return doc["instruction"] + "\n" + addon # Results in worse performance for GPT4

    # Problem: Difficult for problems that have helper functions

    # Code LLAMA
    def prompt_template(instruction: str, context: str, function_start: str) -> str:
        return f"[INST] {instruction}\n[/INST]\n"

    # For octocoder
    # def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #     return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"

    # For wizardcoder
    # def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #     return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

    return prompt_template(doc["instruction"], "", "")


def get_base_prompt_fix(doc, language="python", mode="tests"):
    if language == "rust":
        if mode == "tests":
            return "fn main(){}\n" + doc["declaration"]
        elif mode == "docs":
            return "fn main(){}\n" + doc["declaration"] + doc["prompt"]
        else:
            raise ValueError
    else:
        if mode == "tests":
            return doc["declaration"]
        elif mode == "docs":
            return doc["prompt"]
        else:
            raise ValueError


def get_prompt_fix(doc, language="python", mode="tests"):
    prompt_base = get_base_prompt_fix(doc, language, mode)
    func = prompt_base + doc["buggy_solution"]
    instruction = f'Fix bugs in {doc["entry_point"]}.'
    return func + "\n" + instruction


def get_prompt_explain_desc(doc, language="python"):
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    docstring_len = len(doc["docstring"])

    instruction = f"Provide a concise natural language description of the code using at most {docstring_len} characters."
    func = prompt_base + doc["canonical_solution"]

    return instruction + "\n" + func, docstring_len


def get_prompt_explain_syn(sample, desc, language="python"):
    instruction = f"Write functional code in {LANGUAGE_TO_NAME[language]} according to the description."
    addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    return desc + "\n" + instruction + "\n" + addon


class ParseError(Exception):
    pass


class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"Prompt is not in content:\n{content}")


def call_anyscale_api(prompt: str, temperature: float) -> str:

    import requests

    s = requests.Session()

    BEARER = "esecret_pea3m6uymn3rlp1v57hmn3ctqc"
    url = "https://api.endpoints.anyscale.com/v1/chat/completions"
    body = {
        "model": "codellama/CodeLlama-34b-Instruct-hf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
        "top_p": 0.95,
    }

    response = s.post(
        url, headers={"Authorization": f"Bearer {BEARER}"}, json=body)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()["choices"][0]["message"]["content"]


def call_huggingface_api(url: str, prompt: str, temperature: str) -> str:
    import requests

    BEARER = "tEwqBLUbErqsmwkcqfHfbBXjrxSlAGVaPQLLwIUlWPQHZZAYWYzgAJXWtamYFgioQnUeYfMLDXUdolleQIHbQKaCKHdSCothEQcHDUTJGusyUXTnCGPkknpdTJICVRhi"

    headers = {
        "Authorization": f"Bearer {BEARER}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json={
        "inputs": prompt,
        "parameters": {
            "top_p": 0.95,
            "temperature": temperature,
            "repetition_penalty": 1.15,
            "max_new_tokens": 512,
            "do_sample": True,
            "max_time": None,
            "return_full_text": False,
        },
        "options": {
            "use_cache": False,
            "wait_for_model": True,
        }
    })

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()[0]["generated_text"]


class ApiWrapper:

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def __call__(self, prompt: str, n: int) -> [str]:

        while True:
            try:
                content_list = list()
                for i in range(n):
                    print(f"Generate sample {i} for task...")

                    response = call_anyscale_api(prompt, 0.2)

                    content_list.append(response)
                return content_list
            except Exception as e:
                print("API EXCEPTION:", e)


if __name__ == '__main__':
    TIMES = 20
    VERBOSE = True
    LANGUAGE = "rust"
    # wizardcoder
    # ENDPOINT_URL = "https://xy32vdwpo5jeptys.us-east-1.aws.endpoints.huggingface.cloud"
    # octocoder
    ENDPOINT_URL = "https://me9rxdof1htyppbi.us-east-1.aws.endpoints.huggingface.cloud"
    TASK = "humanevalsynthesize"
    RESULT_FILENAME = f"completions_{LANGUAGE}_{TASK}.jsonl"

    # Load descriptions
    if TASK == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{LANGUAGE}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    samples = [s for s in load_dataset(
        "bigcode/humanevalpack", LANGUAGE)["test"]]

    api_wrapper = ApiWrapper(ENDPOINT_URL)
    parse_errors = 0
    parser = ContentParser()
    for idx, sample in enumerate(tqdm(samples)):
        if TASK == "humanevalfix":
            prompt = get_prompt_fix(sample, language=LANGUAGE, mode="tests")
        elif TASK == "humanevalsynthesize":
            prompt = get_prompt_synthesize(sample, language=LANGUAGE)
        elif TASK == "humanevalexplaindescribe":
            prompt, docstring_len = get_prompt_explain_desc(
                sample, language=LANGUAGE)
            gen = api_wrapper(prompt, TIMES)
            sample["raw_generation"] = gen
            sample["generation"] = [gen_item[:docstring_len]
                                    for gen_item in gen]
            continue
        elif TASK == "humanevalexplainsynthesize":
            desc = descriptions[idx]
            prompt = get_prompt_explain_syn(sample, desc, language=LANGUAGE)
        if VERBOSE:
            print(
                f"Processing {sample['task_id']} ({idx + 1}/{len(samples)}))...")
        sample["raw_generation"] = api_wrapper(prompt, TIMES)

        parsed_samples = list()
        for generation_item in sample["raw_generation"]:
            try:
                parsed_samples.append(
                    parser(prompt, generation_item, sample["entry_point"]))
            except ParseError as e:
                parse_errors += 1
                print("PARSE EXCEPTION:", e)
                parsed_samples.append("")

        sample["generation"] = parsed_samples

        if VERBOSE:
            for i in range(TIMES):
                print(termcolor.colored(
                    sample["entry_point"], "yellow", attrs=["bold"]))
                print(termcolor.colored(prompt, "yellow"))
                print(termcolor.colored(sample["canonical_solution"], "red"))
                print(termcolor.colored(
                    sample["generation"][i], "green")+"\n\n")

        with jsonlines.open(RESULT_FILENAME, "w") as writer:
            writer.write_all(samples)

    if VERBOSE:
        print("parse error rate:", parse_errors / len(samples))

    with jsonlines.open(RESULT_FILENAME, "w") as writer:
        writer.write_all(samples)
