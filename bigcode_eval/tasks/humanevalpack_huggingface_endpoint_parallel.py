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

from importlib.metadata import entry_points
from urllib import response
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

from resp_parser import RespParser, ParseError, find_function_names
import constants
import replicate

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

# "OCTOCODER", "CODE_LLAMA", "CODE_LLAMA_7B", "CODEGEN_RUST", "GPT", "WIZARD_CODER"
LLM_TYPE = "GPT"


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


def get_prompt_synthesize(doc, language="python", prompt_type="prompt"):
    global LLM_TYPE
    # addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    # return doc["instruction"] + "\n" + addon # Results in worse performance for GPT4

    # Problem: Difficult for problems that have helper functions

    # Code LLAMA
    # if LLM_TYPE == "CODE_LLAMA" or LLM_TYPE == "CODE_LLAMA_7B":
    #     def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #         # sys_instruct = "Below is a task description. Write code in idiomatic Rust, by using iterators, match operator, pattern matching, build in std functions, returning values without using `return`, use sage integer operators like `checked_add`, use `?` for error handling, use Option or Result for failed operations, naming convention, that completes the task."
    #         # return f"<s>[INST] <<SYS>>\n{sys_instruct}\n<</SYS>>\n\n{instruction}\n{function_start}[/INST]"
    #         return f"<s>[INST] {instruction}[/INST]"

    # # For octocoder
    # if LLM_TYPE == "OCTOCODER":
    #     def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #         return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"

    # # For wizardcoder
    # if LLM_TYPE == "WIZARD_CODER":
    #     def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #         return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

    # if LLM_TYPE == "CODEGEN_RUST":
    #     def prompt_template(docstring: str) -> str:
    #         return f"{docstring}"

    #     return prompt_template(docstring=doc["prompt"])

    # if LLM_TYPE == "GPT":
    #     def prompt_template(instruction: str, context: str, function_start: str) -> str:
    #         return f"{instruction}"

    def get_prompt(prompt_type, prompt_base, instruction, context=None, prompt_type_prompt="prompt"):

        if context is None:
            inp = instruction
        # `Context first then instruction` methods
        elif prompt_type in ["continue", "instruct"]:
            inp = context + "\n" + instruction
        else:
            inp = instruction + "\n" + context

        if prompt_type == "GPT":
            if prompt_type_prompt == "instruction":
                prompt = instruction.strip()
            elif prompt_type_prompt == "prompt":
                prompt = prompt_base
        elif prompt_type == "continue":
            assert context is None, "The `continue` prompt should only be used for HumanEvalSynthesize. Use `instruct` for HumanEvalFix and HumanEvalExplain."
            prompt = prompt_base
        elif prompt_type == "instruct":
            prompt = inp + "\n\n" + prompt_base
        elif prompt_type in ["CODE_LLAMA", "CODE_LLAMA_7B"]:
            # system = "Provide answers in Rust. Your code should start with ```rust and end with ```."
            # user = f"You are an expert Rust programmer, and here is your task: {inp}\nYour code should start with ```rust and end with ```."
            # user = prompt_base
            # prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"
            # inp = prompt_base
            # prompt = f"You are an expert Rust programmer. Write a idiomatic Rust function to complete the following prompt. Your code should start with ```rust and end with ```.\n{inp}"
            if prompt_type_prompt == "instruction":
                prompt = f"<s>[INST] {instruction.strip()} [/INST]"
            elif prompt_type_prompt == "prompt":
                prompt = prompt_base
        elif prompt_type == "OCTOCODER":
            prompt = f'Question: {inp}\n\nAnswer:\n{prompt_base}'
        elif prompt_type == "octogeex":
            prompt = f'Question: {inp.strip()}\n\nAnswer:\n{prompt_base}'
        elif prompt_type == "starchat":
            # https://huggingface.co/HuggingFaceH4/starchat-beta
            prompt = f'<|system|>\n<|end|>\n<|user|>\n{inp}<|end|>\n<|assistant|>\n{prompt_base}'
        elif prompt_type == "starcodercommit":
            prompt = f'<commit_before><commit_msg>{inp}<commit_after>{prompt_base}'
        elif prompt_type == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inp}\n\n### Response:{prompt_base}'
        elif prompt_type == "WIZARD_CODER":
            # https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen.py#L37
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inp}\n\n### Response:\n{prompt_base}'
        else:
            raise NotImplementedError

        return prompt

    return get_prompt(LLM_TYPE, doc["prompt"], doc["instruction"], "", prompt_type)


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


async def post(session: aiohttp.ClientSession, url: str, headers=dict, body=dict):
    while True:
        try:
            async with session.post(url=url, headers=headers, json=body) as response:

                if response.status != 200:
                    raise Exception(response.text)

                return await response.json()
        except Exception as e:
            print("Unable to get url {} due to {}.".format(url, e))
            await asyncio.sleep(5)
            continue


async def call_replicate_api(session: aiohttp.ClientSession, prompt: str, temperature: float) -> str:

    TOKEN = constants.REPLICATE_TOKEN

    client = replicate.Client(api_token=TOKEN)

    response = await client.async_run(
        ref="lucataco/wizardcoder-15b-v1.0:b8c554180169aa3ea1c8b95dd6af4c24dd9e59dce55148c8f3654752aa641c87",
        input={
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": 512
        },
    )

    return response


async def call_anyscale_api(session: aiohttp.ClientSession, prompt: str, temperature: float) -> str:

    BEARER = constants.ANYSCALE_TOKEN

    client = openai.AsyncClient(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=BEARER
    )

    response = await client.completions.create(
        model="codellama/CodeLlama-34b-Instruct-hf",
        prompt=prompt,
        temperature=temperature,
        max_tokens=512,
        top_p=0.95,
        seed=0,
    )

    return response.choices[0].text


async def call_run_pod(session: aiohttp.ClientSession, model: str, url: str, prompt: str, temperature: str) -> [str]:

    headers = {
        # "authorization": f"Bearer {BEARER}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": temperature,
        "top_p": 0.95,
        "do_sample": True,
        "seed": 0,
        "add_bos_token": False,
    }

    response = await post(session, url, headers=headers, body=body)

    return response["choices"][0]["text"]


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
            "max_new_tokens": 512,
            "do_sample": True,
            "seed": 0
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
                        if LLM_TYPE == "WIZARD_CODER":
                            content_list = await asyncio.gather(*[call_replicate_api(
                                session, prompt, self.temperature) for _ in range(n)])
                        elif LLM_TYPE == "CODE_LLAMA_7B":
                            content_list = list([await call_run_pod(
                                session, "codellama/CodeLlama-7b-Instruct-hf", self.endpoint_url, prompt, self.temperature) for _ in range(n)])
                        elif LLM_TYPE in ["OCTOCODER"]:
                            content_list = await asyncio.gather(*[call_huggingface_api(
                                session, self.endpoint_url, prompt, self.temperature) for _ in range(n)])
                        elif LLM_TYPE == "CODE_LLAMA":
                            content_list = await asyncio.gather(*[call_anyscale_api(
                                session, prompt, self.temperature) for _ in range(n)])

                        return content_list
                except Exception as e:
                    print("API EXCEPTION:", e)


def parse_prompt(prompt: str):
    lines = prompt.splitlines(keepends=True)

    docstring = ""
    func_decl = ""
    examples_docstring = ""
    in_examples = False

    for line in lines:

        if line.startswith("/// Examples"):
            in_examples = True
            continue

        if not in_examples and line.startswith("///"):
            docstring += line

        if in_examples and line.startswith("///"):
            examples_docstring += line

        if line.startswith("fn "):
            func_decl += line

    import re
    pattern = re.compile(
        r"fn\s(?P<func_name>[\w\_]*)[\<\(](.*?)\{", re.MULTILINE | re.DOTALL)

    func_name = pattern.search(func_decl).group("func_name")
    task_instruction = "".join(l[4:]
                               for l in docstring.splitlines(keepends=True))

    return {
        "docstring": docstring,
        "func_decl": func_decl,
        "examples_docstring": examples_docstring,
        "func_name": func_name,
        "prompt": f"Write a Rust function `{func_decl}` for the following task: {task_instruction}"
    }


def do_skip_if_not_zero(idx: int, array: [int]) -> bool:
    if array[idx] == 0:
        return False

    return True


async def main():
    global LLM_TYPE

    START_TASK = 0

    TIMES = 1
    VERBOSE = True
    LANGUAGE = "rust"
    TEMPERATURE = 0.2
    ENDPOINT_URL = None
    SAMPLE_TYPE = "own"  # "humaneval", "own"
    TASK = "humanevalsynthesize"
    PROMPT_TYPE = "instruction"  # "prompt", "instruction"

    RESULT_FILENAME = f"completions_rust_own_bench_gpt4_turbo_{PROMPT_TYPE}.jsonl"
    # RESULT_FILENAME = "/home/al9hu7/workspace/ma/generated-data/humaneval-rust-samples/completions_rust_humanevalsynthesize_codellama_instruct_34b_t0.2_tp0.95.jsonl"
    # RESULT_FILENAME = "/home/al9hu7/workspace/ma/generated-data/ownbenchmark-samples/completions_rust_ownbenchmark_gpt_4turbo_t0.8_tp0.95.jsonl"
    # RESULT_FILENAME = "/home/al9hu7/workspace/ma/generated-data/humaneval-rust-samples/completions_rust_humanevalsynthesize_wizardcoder_t0.2_tp0.95.jsonl"

    import constants
    API_KEY = constants.OPENAI_KEY
    # wizardcoder
    if LLM_TYPE == "WIZARD_CODER":
        ENDPOINT_URL = "https://xy32vdwpo5jeptys.us-east-1.aws.endpoints.huggingface.cloud"
    # octocoder
    if LLM_TYPE == "OCTOCODER":
        # ENDPOINT_URL = "https://faidngpb26hefmo4.us-east-1.aws.endpoints.huggingface.cloud"
        ENDPOINT_URL = "https://w1zpuf47j65gski5.us-east-1.aws.endpoints.huggingface.cloud"
        # ENDPOINT_URL = "https://me9rxdof1htyppbi.us-east-1.aws.endpoints.huggingface.cloud"
    if LLM_TYPE == "CODE_LLAMA_7B":
        ENDPOINT_URL = "https://b9al2eru8mdlz5uo.us-east-1.aws.endpoints.huggingface.cloud"
        ENDPOINT_URL = "https://b9al2eru8mdlz5uo.us-east-1.aws.endpoints.huggingface.cloud"

        POD_ID = "m1ashbwwphz9qj"
        POD_URL = f"https://{POD_ID}-5000.proxy.runpod.net/"
        ENDPOINT_URL = f"{POD_URL}v1/completions"

    # Load descriptions
    if TASK == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{LANGUAGE}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # start at a specific task from already sampled file
    if START_TASK > 0 or True:
        with jsonlines.open(RESULT_FILENAME, "r") as reader:
            samples = [s for s in reader]

        if SAMPLE_TYPE == "humaneval":
            BENCHMARK_FILE = "/home/al9hu7/workspace/ma/generated-data/benchmarks/merged_humaneval-pack_multipl-e/rust-benchmark.jsonl"
            tasks = [s for s in jsonlines.open(BENCHMARK_FILE, "r")]

        elif SAMPLE_TYPE == "own":
            BENCHMARK_FILE = "/home/al9hu7/workspace/ma/generated-data/benchmarks/own-rust-benchmark/rust-benchmark.jsonl"
            tasks = [s for s in jsonlines.open(BENCHMARK_FILE, "r")]

        for (sample, task) in zip(samples, tasks):
            sample["prompt"] = task["prompt"]
            sample["instruction"] = task["instruction"]
            sample["canonical_solution"] = task["canonical_solution"]
            sample["entry_point"] = task["entry_point"]
            sample["test"] = task["test"]

        tasks = samples
    else:
        if SAMPLE_TYPE == "humaneval":
            BENCHMARK_FILE = "/home/al9hu7/workspace/ma/generated-data/benchmarks/merged_humaneval-pack_multipl-e/rust-benchmark.jsonl"
            tasks = [s for s in jsonlines.open(BENCHMARK_FILE, "r")]

        elif SAMPLE_TYPE == "own":
            BENCHMARK_FILE = "/home/al9hu7/workspace/ma/generated-data/benchmarks/own-rust-benchmark/rust-benchmark.jsonl"
            tasks = [s for s in jsonlines.open(BENCHMARK_FILE, "r")]

    if LLM_TYPE == "CODEGEN_RUST":
        api_wrapper = CodegenRustModel(TEMPERATURE)
    else:
        api_wrapper = ApiWrapper(ENDPOINT_URL, TEMPERATURE, API_KEY)

    parse_errors = 0

    errors_per_sample = []

    parser = RespParser(starts_with_function=True if PROMPT_TYPE ==
                        "prompt" else False, collect_imports_into_func_body=True if PROMPT_TYPE == "instruction" else False, language="rust")
    for idx, task in enumerate(tqdm(tasks)):
        # if do_skip_if_not_zero(idx, [])):
        #     continue

        if START_TASK != 0 and idx < START_TASK:
            continue

        task_id = task['task_id'] if "task_id" in task else "Rust/" + \
            task["name"].split("_")[1]

        task_idx = int(task_id.split("/")[1])

        # sample only the given tasks
        # [115, 160, 74, 109, 129, 91, 46, 78, 94, 28, 81]
        # if task_idx not in [84, 91]:
        if task_idx not in [32]:
            continue

        if TASK == "humanevalfix":
            prompt = get_prompt_fix(task, language=LANGUAGE, mode="tests")
        elif TASK == "humanevalsynthesize":
            prompt = get_prompt_synthesize(
                task, language=LANGUAGE, prompt_type=PROMPT_TYPE)
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
                f"Processing {task_id} ({idx + 1}/{len(tasks)}))...")

        task["raw_prompt"] = prompt

        task["raw_generation"] = await api_wrapper(prompt, TIMES)

        entry_point = task["entry_point"] if "entry_point" in task else find_function_names(
            task["prompt"])[0]

        if "entry_point" not in task:
            task["entry_point"] = entry_point

        parsed_samples = list()
        errors = 0
        for generation_item in task["raw_generation"]:
            try:
                parsed_samples.append(
                    parser(generation_item, entry_point, helper_decl="")[0])
            except ParseError as e:
                errors += 1
                parse_errors += 1
                print("PARSE EXCEPTION:", e)
                parsed_samples.append("")

        task_id = task['task_id'] if "task_id" in task else "Rust/" + task["name"].split("_")[
            1]

        errors_per_sample.append({
            "task_id": task_id,
            "errors": errors,
        })

        task["generation"] = parsed_samples

        if VERBOSE:
            for i in range(TIMES):
                entry_point = task["entry_point"] if "entry_point" in task else find_function_names(
                    task["prompt"])[0]
                print(termcolor.colored(
                    entry_point, "yellow", attrs=["bold"]))
                print(termcolor.colored(prompt, "yellow"))
                print(termcolor.colored(
                    task["canonical_solution"] if "canonical_solution" in task else "-", "red"))
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
