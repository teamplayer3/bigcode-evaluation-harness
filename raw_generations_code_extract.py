from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from typing import List

import jsonlines
from bigcode_eval.tasks.resp_parser import RespParser


def prompt_template(instruction: str, context: str, function_start: str) -> str:
    return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"


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

        if func_start_in_prompt:
            # split at func start
            for entry_point in self._entry_point_variations(entry_point):
                if language == "python":
                    func_prefix = "def"
                elif language == "rust":
                    func_prefix = "fn"

                entry_point = f"{func_prefix} {entry_point}"
                if entry_point in content:

                    parts = content.split(entry_point)
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


# prompt = """\
# Question: Write a Rust function `below_zero(operations:Vec<i32>) -> bool` to solve the following problem:
# You're given a list of deposit and withdrawal operations on a bank account that starts with
# zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
# at that point function should return True. Otherwise it should return False.


# Answer:
# fn main(){}

# use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
# use rand::Rng;
# use regex::Regex;
# use md5;
# use std::any::{Any, TypeId};

# /*
#  You're given a list of deposit and withdrawal operations on a bank account that starts with
#     zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
#     at that point function should return True. Otherwise it should return False.

# */
# fn below_zero(operations:Vec<i32>) -> bool{\
# """
raw_out = """```rust\nuse std::iter::Sum;\n\nfn sumup_values<T: Sum + Clone>(values: &[T]) -> T {\n    values.iter().cloned().sum()\n}\n\n// Example usage:\n// let nums = vec![1, 2, 3, 4, 5];\n// let result = sumup_values(&nums);\n// println!(\"The sum is: {}\", result);\n```
"""

# print(RespParser(pretend_entry_func_included=False)(raw_out, "sumup_values", ""))

# exit()

PATH = "/home/al9hu7/workspace/ma/generated-data/humaneval-rust-samples/"
# PATH = "/home/al9hu7/workspace/ma/generated-data/ownbenchmark-samples/"
# PATH = "./"
FILE_NAME = "completions_rust_humanevalsynthesize_codellama_instruct_7b_t0.2_tp0.95"
# FILE_NAME = "completions_rust_humanevalsynthesize_gpt_4turbo_t0.8_tp0.95"
# FILE_NAME = "completions_rust_humanevalsynthesize"
# FILE_NAME = "completions_rust_ownbenchmark_gpt_4turbo_t0.8_tp0.95"

FUNC_START_IN_PROMPT = True

content_parser = RespParser(language="rust")

errors_per_sample = []

with jsonlines.open(f"{PATH}{FILE_NAME}.jsonl") as reader:
    with jsonlines.open(f"{PATH}{FILE_NAME}_conf.jsonl", mode="w") as writer:
        parser = ContentParser()
        parse_errors = 0
        samples = 0
        for line in reader:
            if "raw_generation" not in line:
                continue

            raw_generations = line["raw_generation"]
            parsed = list()
            errors = 0
            for raw_generation in raw_generations:
                samples += 1
                try:
                    parsed_gen = content_parser(
                        response=raw_generation, entry_point=line["entry_point"], helper_decl="")
                    parsed.append(parsed_gen)
                except ParseError as e:
                    parse_errors += 1
                    errors += 1
                    if line["task_id"] == "Rust/129":
                        print(e)
                    parsed.append("")

            errors_per_sample.append({
                "task_id": line["task_id"],
                "errors": errors,
            })
            line["generation"] = parsed

            writer.write(line)

        print("parse error rate:", parse_errors / samples)

print("errors per task:", list(
    filter(lambda e: e["errors"] > 0, errors_per_sample)))
