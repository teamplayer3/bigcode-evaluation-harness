from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from typing import List

import jsonlines
from bigcode_eval.tasks.resp_parser import RespParser, ParseError, find_function_names


def prompt_template(instruction: str, context: str, function_start: str) -> str:
    return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"


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

# PATH = "/home/al9hu7/workspace/ma/generated-data/humaneval-rust-samples/"
# PATH = "/home/al9hu7/workspace/ma/generated-data/ownbenchmark-samples/"
PATH = "./"
# FILE_NAME = "completions_rust_humanevalsynthesize_codellama_instruct_7b_t0.2_tp0.95"
# FILE_NAME = "completions_rust_ownbenchmark_codellame_instruct_34b_t0.2_tp0.95"
# FILE_NAME = "completions_rust_humanevalsynthesize_wizardcoder_t0.2_tp0.95"
# FILE_NAME = "completions_rust_humanevalsynthesize_gpt_4turbo_t0.8_tp0.95"
FILE_NAME = "completions_rust_multiple_gpt4_turbo_instruction"
# FILE_NAME = "completions_rust_ownbenchmark_gpt_4turbo_t0.8_tp0.95"


content_parser = RespParser(
    collect_imports_into_func_body=True, starts_with_function=False, remove_entry_func_head=False, multiple_samples=False, pretend_entry_func_included=True, language="rust")

errors_per_sample = []

with jsonlines.open(f"{PATH}{FILE_NAME}.jsonl") as reader:
    with jsonlines.open(f"{PATH}{FILE_NAME}_conf.jsonl", mode="w") as writer:
        parse_errors = 0
        samples = 0
        for idx, line in enumerate(reader):
            if "raw_generation" not in line:
                continue

            task_id = line['task_id'] if "task_id" in line else "Rust/" + \
                line["name"].split("_")[1]
            entry_point = line["entry_point"] if "entry_point" in line else find_function_names(
                line["prompt"])[0]

            line["task_id"] = task_id
            line["entry_point"] = entry_point

            raw_generations = line["raw_generation"]
            parsed = list()
            errors = 0
            for raw_generation in raw_generations:
                samples += 1
                try:
                    parsed_gen = content_parser(
                        response=raw_generation, entry_point=entry_point, helper_decl="")[0]
                    parsed.append(parsed_gen)
                except ParseError as e:
                    parse_errors += 1
                    errors += 1
                    if task_id == "Rust/33":
                        print(e)
                    parsed.append("")

            errors_per_sample.append({
                "task_id": task_id,
                "errors": errors,
            })
            line["generation"] = parsed

            writer.write(line)

        print("parse error rate:", parse_errors / samples)

print("errors per task:", list(
    filter(lambda e: e["errors"] > 0, errors_per_sample)))
