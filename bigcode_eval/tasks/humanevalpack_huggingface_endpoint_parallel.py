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

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import os
import openai
import jsonlines
import termcolor
import asyncio
import aiohttp

from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from datasets import load_dataset
from typing import Any, List
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

# "OCTOCODER", "CODE_LLAMA", "CODE_LLAMA_7B", "CODEGEN_RUST", "GPT"
LLM_TYPE = "CODE_LLAMA"


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
    global LLM_TYPE
    # addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    # return doc["instruction"] + "\n" + addon # Results in worse performance for GPT4

    # Problem: Difficult for problems that have helper functions

    # Code LLAMA
    if LLM_TYPE == "CODE_LLAMA" or LLM_TYPE == "CODE_LLAMA_7B":
        def prompt_template(instruction: str, context: str, function_start: str) -> str:
            # sys_instruct = "Below is a task description. Write code in idiomatic Rust, by using iterators, match operator, pattern matching, build in std functions, returning values without using `return`, use sage integer operators like `checked_add`, use `?` for error handling, use Option or Result for failed operations, naming convention, that completes the task."
            # return f"<s>[INST] <<SYS>>\n{sys_instruct}\n<</SYS>>\n\n{instruction}\n{function_start}[/INST]"
            return f"<s>[INST] {instruction}[/INST]"

    # For octocoder
    if LLM_TYPE == "OCTOCODER":
        def prompt_template(instruction: str, context: str, function_start: str) -> str:
            return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"

    # For wizardcoder
    if LLM_TYPE == "WIZARD_CODER":
        def prompt_template(instruction: str, context: str, function_start: str) -> str:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

    if LLM_TYPE == "CODEGEN_RUST":
        def prompt_template(docstring: str) -> str:
            return f"{docstring}"

        return prompt_template(docstring=doc["prompt"])

    if LLM_TYPE == "GPT":
        def prompt_template(instruction: str, context: str, function_start: str) -> str:
            return f"{instruction}"

    return prompt_template(doc["instruction"], "", doc["prompt"])


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

    def __call__(self, prompt: str, content: str, entry_point: str, func_start_in_prompt: bool = True, language: str = "python"):
        raw_content = content

        content = content.replace(prompt, "", 1)

        # remove comments
        content = "".join(filter(lambda x: not x.startswith(
            "//"), content.splitlines(keepends=True)))

        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        # matcher = CSequenceMatcher(None, prompt, content)
        # tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        # if tag == "insert":
        #     return content[j1:j2]
        # second parse content with assumption that model wrote code without description

        if not func_start_in_prompt:
            # split at func start
            for entry_point in self._entry_point_variations(entry_point):
                if entry_point in content:
                    if language == "python":
                        func_prefix = "def"
                    elif language == "rust":
                        func_prefix = "fn"

                    parts = content.split(f"{func_prefix} {entry_point}")
                    if len(parts) > 1:
                        content = "".join(
                            parts[1].splitlines(keepends=True)[1:])

                    parts = content.split(f"{func_prefix} main")
                    if len(parts) > 1:
                        content = parts[0]

        if language == "python":
            content_lines = content.splitlines(keepends=True)[1:]
            func_lines = []
            for line in content_lines:
                if line.startswith("    ") or line.startswith("\n"):
                    func_lines.append(line)
                else:
                    break
            content = "".join(func_lines)
        elif language == "rust":
            start_idx = 0
            end_idx = 0
            open_brackets = 1
            in_string = False
            in_char_string = False
            char_str_len = 0
            content_out = ""
            in_func = True
            cycle_buf = ""
            wait_for_func_start = False

            for idx, char in enumerate(content):
                if in_func:
                    last_was_escape = len(
                        cycle_buf) == 0 or not cycle_buf[-1] == "\\"

                    if not in_char_string and not last_was_escape and char == '"':
                        in_string = not in_string

                    if not in_string and not not last_was_escape and char == "'":
                        char_str_len = 0
                        in_char_string = not in_char_string

                    if in_char_string:
                        char_str_len += 1
                        if char_str_len == 3:
                            in_char_string = False
                            char_str_len = 0

                    if not in_string and not in_char_string:
                        if char == '{':
                            open_brackets += 1
                        elif char == '}':
                            open_brackets -= 1

                    if open_brackets == 0:
                        end_idx = idx
                        in_func = False
                else:
                    if cycle_buf[-2:] + char == "\nfn":
                        wait_for_func_start = True

                    if wait_for_func_start and char == "{":
                        open_brackets += 1
                        wait_for_func_start = False
                        in_func = True

                content_out += char

                if len(cycle_buf) > 3:
                    cycle_buf = cycle_buf[1:]
                cycle_buf += char

            content = content_out[start_idx:end_idx + 1]

        if len(content.strip()) == 0:
            raise ParseError(f"Prompt is not in content:\n{raw_content}")

        return content


async def post(session: aiohttp.ClientSession, url: str, headers=dict, body=dict):
    try:
        async with session.post(url=url, headers=headers, json=body) as response:

            if response.status != 200:
                raise Exception(response.text)

            return await response.json()
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e))


async def call_anyscale_api(session: aiohttp.ClientSession, prompt: str, temperature: float) -> str:

    BEARER = "esecret_pea3m6uymn3rlp1v57hmn3ctqc"
    url = "https://api.endpoints.anyscale.com/v1/chat/completions"
    body = {
        "model": "codellama/CodeLlama-34b-Instruct-hf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
        "top_p": 0.95,
    }

    response = await post(session,
                          url, headers={"Authorization": f"Bearer {BEARER}"}, body=body)

    return response["choices"][0]["message"]["content"]


async def call_huggingface_api(session: aiohttp.ClientSession, url: str, prompt: str, temperature: str) -> str:

    BEARER = "tEwqBLUbErqsmwkcqfHfbBXjrxSlAGVaPQLLwIUlWPQHZZAYWYzgAJXWtamYFgioQnUeYfMLDXUdolleQIHbQKaCKHdSCothEQcHDUTJGusyUXTnCGPkknpdTJICVRhi"
    headers = {
        "Authorization": f"Bearer {BEARER}",
        "Content-Type": "application/json"
    }
    body = {
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
    }

    response = await post(session, url, headers=headers, body=body)

    return response[0]["generated_text"]


class CodegenRustModel:

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature
        model_name = "ammarnasr/codegen-350M-mono-rust"
        peft_config = PeftConfig.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path)

        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, device_map="cuda:0")
        self.model = PeftModel.from_pretrained(model, model_name)

    async def __call__(self, prompt: str, n: int) -> [str]:

        prompts = [prompt for i in range(n)]

        input_ids = self.tokenizer(prompts, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **input_ids, max_length=512, temperature=self.temperature, do_sample=True, top_p=0.95)
        resp = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        return resp


class ApiWrapper:

    def __init__(self, endpoint_url: str, temperature: float, openai_api_key: str = None) -> None:
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        if LLM_TYPE == "GPT":
            self.client = openai.OpenAI(
                api_key=openai_api_key
            )

    async def __call__(self, prompt: str, n: int) -> [str]:
        global LLM_TYPE

        if LLM_TYPE == "GPT":
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
                        model="gpt-4-1106-preview",
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
        else:
            while True:
                try:
                    async with aiohttp.ClientSession() as session:
                        if LLM_TYPE == "WIZARD_CODER" or LLM_TYPE == "OCTOCODER" or LLM_TYPE == "CODE_LLAMA_7B":
                            content_list = await asyncio.gather(*[call_huggingface_api(
                                session, self.endpoint_url, prompt, self.temperature) for _ in range(n)])
                        if LLM_TYPE == "CODE_LLAMA":
                            content_list = await asyncio.gather(*[call_anyscale_api(
                                session, prompt, self.temperature) for _ in range(n)])

                        return content_list
                except Exception as e:
                    print("API EXCEPTION:", e)


async def main():
    global LLM_TYPE

    START_TASK = 0

    TIMES = 1
    VERBOSE = True
    LANGUAGE = "rust"
    TEMPERATURE = 0.2
    ENDPOINT_URL = None
    SAMPLE_TYPE = "own"
    TASK = "humanevalsynthesize"

    RESULT_FILENAME = f"completions_{LANGUAGE}_{TASK}.jsonl"
    # RESULT_FILENAME = "/home/al9hu7/workspace/ma/generated-data/humaneval-rust-samples/completions_rust_humanevalsynthesize_codellama_instruct_34b_t0.2_tp0.95.jsonl"
    # RESULT_FILENAME = "/home/al9hu7/workspace/ma/generated-data/ownbenchmark-samples/completions_rust_ownbenchmark_gpt_4turbo_t0.8_tp0.95.jsonl"

    import constants
    API_KEY = constants.OPENAI_KEY
    # wizardcoder
    if LLM_TYPE == "WIZARD_CODER":
        ENDPOINT_URL = "https://xy32vdwpo5jeptys.us-east-1.aws.endpoints.huggingface.cloud"
    # octocoder
    if LLM_TYPE == "OCTOCODER":
        ENDPOINT_URL = "https://faidngpb26hefmo4.us-east-1.aws.endpoints.huggingface.cloud"
        # ENDPOINT_URL = "https://me9rxdof1htyppbi.us-east-1.aws.endpoints.huggingface.cloud"
    if LLM_TYPE == "CODE_LLAMA_7B":
        # ENDPOINT_URL = "https://b9al2eru8mdlz5uo.us-east-1.aws.endpoints.huggingface.cloud"
        ENDPOINT_URL = ""

    # Load descriptions
    if TASK == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{LANGUAGE}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def tasks_mapper(task):
        task["task_id"] = f"Rust/{task['task_id']}" if isinstance(
            task["task_id"], int) else task["task_id"]
        task["declaration"] = task["declaration"]
        task["test"] = task["test"]
        task["entry_point"] = task["entry_point"]
        task["canonical_solution"] = task["canonical_solution"]
        task["instruction"] = task["instruction"] + \
            "\n" + task["prompt"] + "\n" + task["helper"]
        return task

    if SAMPLE_TYPE == "humaneval":
        tasks = [s for s in load_dataset(
            "bigcode/humanevalpack", LANGUAGE)["test"]]
    elif SAMPLE_TYPE == "own":
        PATH = "/home/al9hu7/workspace/ma/generated-data/own-rust-benchmark/rust-benchmark.json"
        with open(PATH, "r") as f:
            import json
            tasks = json.load(f)

    # start at a specific task from already sampled file
    if START_TASK > 0:
        with jsonlines.open(RESULT_FILENAME, "r") as reader:
            samples = [s for s in reader]

        # add missing tasks from benchmark
        out_samples = []
        tasks_iter = tasks.__iter__()
        for line in samples:
            out_samples.append(line)
            tasks_iter.__next__()
        for line in tasks_iter:
            out_samples.append(line)

        tasks = out_samples

    # map from own benchmark to humaneval format
    if SAMPLE_TYPE == "own":
        tasks = list(map(tasks_mapper, tasks))

    if LLM_TYPE == "CODEGEN_RUST":
        api_wrapper = CodegenRustModel(TEMPERATURE)
    else:
        api_wrapper = ApiWrapper(ENDPOINT_URL, TEMPERATURE, API_KEY)

    parse_errors = 0

    errors_per_sample = []

    parser = ContentParser()
    for idx, task in enumerate(tqdm(tasks)):
        if idx < START_TASK:
            continue

        if TASK == "humanevalfix":
            prompt = get_prompt_fix(task, language=LANGUAGE, mode="tests")
        elif TASK == "humanevalsynthesize":
            prompt = get_prompt_synthesize(task, language=LANGUAGE)
        elif TASK == "humanevalexplaindescribe":
            prompt, docstring_len = get_prompt_explain_desc(
                task, language=LANGUAGE)
            gen = await api_wrapper(prompt, TIMES)
            task["raw_generation"] = gen
            task["generation"] = [gen_item[:docstring_len]
                                  for gen_item in gen]
            continue
        elif TASK == "humanevalexplainsynthesize":
            desc = descriptions[idx]
            prompt = get_prompt_explain_syn(task, desc, language=LANGUAGE)
        if VERBOSE:
            print(
                f"Processing {task['task_id']} ({idx + 1}/{len(tasks)}))...")

        task["raw_generation"] = await api_wrapper(prompt, TIMES)

        parsed_samples = list()
        errors = 0
        for generation_item in task["raw_generation"]:
            try:
                parsed_samples.append(
                    parser(prompt, generation_item, task["entry_point"], False, LANGUAGE))
            except ParseError as e:
                errors += 1
                parse_errors += 1
                print("PARSE EXCEPTION:", e)
                parsed_samples.append("")

        errors_per_sample.append({
            "task_id": task["task_id"],
            "errors": errors,
        })

        task["generation"] = parsed_samples

        if VERBOSE:
            for i in range(TIMES):
                print(termcolor.colored(
                    task["entry_point"], "yellow", attrs=["bold"]))
                print(termcolor.colored(prompt, "yellow"))
                print(termcolor.colored(task["canonical_solution"], "red"))
                print(termcolor.colored(
                    task["generation"][i], "green")+"\n\n")

        with jsonlines.open(RESULT_FILENAME, "w") as writer:
            writer.write_all(tasks)

    if VERBOSE:
        print("parse error rate:", parse_errors / len(tasks))
        print("errors per task:", list(
            filter(lambda e: e["errors"] > 0, errors_per_sample)))

    with jsonlines.open(RESULT_FILENAME, "w") as writer:
        writer.write_all(tasks)

if __name__ == '__main__':
    asyncio.run(main())
