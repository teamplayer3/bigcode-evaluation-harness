import bigcode_eval.tasks.constants as constants
import asyncio
import aiohttp


async def post(session: aiohttp.ClientSession, url: str, headers=dict, body=dict):
    try:
        async with session.post(url=url, headers=headers, json=body) as response:

            if response.status != 200:
                raise Exception(response.text)

            return await response.json()
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e))


async def call_anyscale_api(session: aiohttp.ClientSession, prompt: str, temperature: float) -> str:

    BEARER = constants.ANYSCALE_TOKEN
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


def prompt_template(instruction: str, context: str, function_start: str) -> str:
    sys_instruct = "Below is a task description. Write code in idiomatic Rust and wrap it with ```."
    # return f"<s>[INST] <<SYS>>\n{sys_instruct}\n<</SYS>>\n\n{instruction}[/INST]"
    return f"\n{instruction}\n\n{function_start}"


def fix_prompt_template(sample: str, function_task: str, compiler_error: str | None = None, runtime_error: str | None = None) -> str:

    if compiler_error:
        error_type = "compiler"
        error = compiler_error
    elif runtime_error:
        error_type = "runtime"
        error = runtime_error

    instruction = f"You are an expert Rust programmer, and here is your task: You are provided with a function written in rust with an bug and the function usage. In addition you get the {error_type} error. Give two possible fixed versions of the function which compile and run without errors. Fix only the function body, not the function declaration. Your code should start with ```rust and end with ```."

    template = f"{instruction}\n\n{function_task}\n```rust\n{sample}\n```\n\n{error}"

    return template


async def main():

    function_head = "count_most_occurrent_number(numbers: &[u32]) -> usize"

    instruction = f"Write an idiomatic Rust function `{function_head}` to solve the given problem. Think in terms of the following:\nuse as many Rust features as possible, e.g., iterators, closures, etc.\n And the best fitting function to use as little code as possible. Use the std function `starts_with()`."

    example = """
Follow this example:

Take this solution:
```
a.saturating_add(b)
```
Instead of this:
```
match a.checked_add(b) {
    Some(r) => r,
    None => u32::MAX
}
```
and this:
```
if (a as u64 + b as u64) > u32::MAX as u64 {
    u32::MAX
} else {
    a + b
}
```
    """

    task = """The function takes in a slice of `u32`. It should count all individual numbers and return the number of the most occurrent number. If there are multiple numbers with the same number of occurrences, return the first one."""

    prompt = f"You are an expert Rust programmer, and here is your task: {instruction}{task}\nYour code should start with ```rust and end with ```."

#     prompt = """
# Give me an optimized idiomatic Rust function for the following function. Use the std function `starts_with()`:
# pub fn number_slice_starts_with(numbers: &[u32], start_pattern: &[u32]) -> bool {
#     if numbers.len() < start_pattern.len() {
#         return false
#     }

#     numbers.iter().zip(start_pattern.iter()).all(|(n, p)| n == p)
# }
#     """

    # prompt = "Write a Rust function `has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool` to solve the following problem: Check if in given list of numbers, are any two numbers closer to each other than given threshold."
    # prompt_base = "fn main(){} use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt}; use rand::Rng; use regex::Regex; use md5; use std::any::{Any, TypeId}; /* Check if in given list of numbers, are any two numbers closer to each other than given threshold. */ fn has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool{"

    # prompt = prompt_template(prompt, "", prompt_base)

    sample = "fn first_n_are_prefix(numbers: &[u32], prefix: &[u32]) -> bool {\n    numbers.len() >= prefix.len() && numbers[..prefix.len()] == prefix\n}"
    compiler_error = "E0277 : can't compare `[u32]` with `&[u32]`"

    function_task = "Write an idiomatic Rust function `strip_all_newlines(lines: &mut [String])` to solve the following problem: The function takes in a mutable slice of strings. It should remove all new line character `\\n` from the end of each line by using the function `strip_newline`."
    sample = "\nfn strip_newline(s: &mut String) {\n    if s.ends_with('\\n') {\n        s.pop();\n    }\n}\n\nfn strip_all_newlines(lines: &mut [String]) {    lines.iter_mut().for_each(|line| line.strip_newline());\n}"
    compiler_error = "E0599 : no method named `strip_newline` found for mutable reference `&mut String` in the current scope"

    prompt = fix_prompt_template(
        sample, function_task, compiler_error=compiler_error)

    print(f"Prompt:\n\n {prompt}")

    async with aiohttp.ClientSession() as session:
        answer = await call_anyscale_api(session, prompt, 0.2)

    print(f"Answer:\n\n{answer}")

if __name__ == "__main__":
    asyncio.run(main())
