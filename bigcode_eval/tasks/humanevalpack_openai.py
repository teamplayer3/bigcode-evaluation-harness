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

import constants
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
    return doc["instruction"]


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
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.
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


class ChatWrapper:

    def __init__(self, model: str, api_key: str):
        self._model = model
        self.client = openai.OpenAI(
            api_key=api_key
        )

    def __call__(self, prompt: str, n: int) -> str:
        messages = [
            {"role": "system", "content": "You are a rust expert. Answer only with the requested function and maybe helpers after the target function in one code block."},
            {
                "role": "user",
                "content": prompt,
            }
        ]
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.8,
                    top_p=0.95,
                    n=n
                )
                content_list = list()
                for i in range(n):
                    message = response.choices[i].message.content
                    content_list.append(message)
                return content_list
            except Exception as e:
                print("API EXCEPTION:", e)


if __name__ == '__main__':
    TIMES = 1
    VERBOSE = True
    LANGUAGE = "rust"
    MODEL = "gpt-4-1106-preview"
    TASK = "humanevalsynthesize"
    API_KEY = constants.OPENAI_KEY
    SAMPLE_TYPE = "own"  # "humaneval"

    RESULTS_FILENAME = f"completions_{LANGUAGE}_{TASK}.jsonl"

    # Load descriptions
    if TASK == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{LANGUAGE}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if SAMPLE_TYPE == "humaneval":
        samples = [s for s in load_dataset(
            "bigcode/humanevalpack", LANGUAGE)["test"]]
    elif SAMPLE_TYPE == "own":
        PATH = "/home/al9hu7/workspace/ma/generated-data/own-rust-benchmark/rust-benchmark.json"
        with open(PATH, "r") as f:
            import json
            samples = json.load(f)

        def sample_mapper(sample):
            sample["task_id"] = f"Rust/{sample['task_id']}"
            sample["declaration"] = sample["declaration"]
            sample["test"] = sample["test"]
            sample["entry_point"] = sample["entry_point"]
            sample["canonical_solution"] = sample["canonical_solution"]
            sample["instruction"] = sample["instruction"] + \
                "\n" + sample["prompt"] + "\n" + sample["helper"]
            return sample

        samples = list(map(sample_mapper, samples))

    chat_wrapper = ChatWrapper(MODEL, API_KEY)
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
            gen = chat_wrapper(prompt, TIMES)
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
        sample["raw_generation"] = chat_wrapper(prompt, TIMES)
        parsed_samples = list()
        try:
            for generation_item in sample["raw_generation"]:
                parsed_sample = parser(
                    prompt, generation_item, sample["entry_point"])
                parsed_samples.append(parsed_sample)
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

        with jsonlines.open(RESULTS_FILENAME, "w") as writer:
            writer.write_all(samples)

    if VERBOSE:
        print("parse error rate:", parse_errors / len(samples))

    with jsonlines.open(RESULTS_FILENAME, "w") as writer:
        writer.write_all(samples)
