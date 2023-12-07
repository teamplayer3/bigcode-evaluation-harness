import json
import bigcode_eval.tasks.constants as constants
import asyncio
import aiohttp
from bigcode_eval.tasks.resp_parser import RespParser, ParseError
import lib


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


def fix_prompt_template(sample: str, function_task: str, compiler_error: str | None = None, runtime_error: str | None = None) -> str:

    if compiler_error is not None:
        error_type = "compiler"
        error = compiler_error
    elif runtime_error is not None:
        error_type = "runtime"
        error = runtime_error
    else:
        error_type = "timeout"
        error = "Program not finished in time."

    instruction = f"You are an expert Rust programmer, and here is your task: You are provided with a function written in rust with an bug and the function usage. In addition you get the {error_type} error. Give two possible fixed versions of the function which compile and run without errors. Fix only the function body, not the function declaration. Your code should start with ```rust and end with ```."

    template = f"{instruction}\n\n{function_task}\n```rust\n{sample}\n```\n\n{error}"

    return template


WORKSPACE = "/home/al9hu7/workspace/ma/generated-data/ownbenchmark-evaluation-results/"

INPUT_FILE = f"{WORKSPACE}out_completions_rust_ownbenchmark_codellame_instruct_34b_t0.2_tp0.95_23-11-29_19-05"

data = {}

with open(INPUT_FILE + ".json", "r") as f:
    data = json.load(f)


tasks = data["compiler_stats"]["compiler_out"]

content_parser = RespParser(
    collect_imports_into_func_body=True, multiple_samples=True, language="rust")


async def run_episode(episode_idx: 0, sample: str, err_type: str, comp_run_out: str, entry_point: str, instruction: str, declaration: str, test: str, helper: str, timeout: float = 5.0) -> dict:

    compiler_error = comp_run_out if err_type == "compiler_error" else None
    runtime_error = comp_run_out if err_type == "runtime_error" else None

    prompt = fix_prompt_template(
        sample, instruction, compiler_error=compiler_error, runtime_error=runtime_error)

    async with aiohttp.ClientSession() as session:
        raw_generation = await call_anyscale_api(session, prompt, 0.2)

    parsed_samples = content_parser(response=raw_generation,
                                    entry_point=entry_point, helper_decl=helper)

    fix_results = []
    fixed = False
    best_sample = None

    for fix_sample in parsed_samples:

        out = lib.test_in_cargo_project(
            "tmp", test, fix_sample, declaration, helper, timeout=timeout)

        if isinstance(out, lib.Timeout):
            fix_results.append({
                "type": "timeout",
                "sample_code": fix_sample,
            })
            continue

        compiler_warnings, compiler_errors, test_result = out

        sample_passed = compiler_errors is None and not isinstance(
            test_result, lib.TestRuntimeError)

        sample_code = f"{declaration}\n{fix_sample}"
        if sample_passed:
            fix_results.append({
                "passed": True,
                "sample_code": sample_code,
                "run_time": test_result,
            })

            best_sample = {
                "sample_code": sample_code,
                "run_time": test_result,
            }

            fixed = True

        else:
            runtime_error = test_result
            new_err_type = "compiler_error" if compiler_errors is not None else "runtime_error"
            fix_results.append({
                "passed": False,
                "type": new_err_type,
                "sample_code":  sample_code,
                "out": [str(c) for c in compiler_errors] if compiler_errors is not None else str(runtime_error),
            })

            if err_type in ["compiler_error", "timeout"] and new_err_type == "runtime_error":
                best_sample = {
                    "type": new_err_type,
                    "sample_code": sample_code,
                    "out": str(runtime_error),
                }

    return {
        "episode_idx": episode_idx,
        "samples": fix_results,
        "best_sample": best_sample,
        "fixed": fixed,
    }


async def main():

    SKIP_ALREADY_FIXED = True

    for task in tasks:

        task_id = task["task_id"]

        print(f"Fix task {task_id}.")

        if task_id in ["Rust/11"]:
            continue

        helper = task["helper"] if "helper" in task else ""
        instruction = task["instruction"]
        entry_point = task["entry_point"]
        declaration = task["declaration"]
        test = task["test"]

        for sample in task["faults"]:

            if SKIP_ALREADY_FIXED and "fixed_with" in sample and sample["fixed_with"] is not None:
                continue

            err_type = sample["type"]
            sample_code = sample["sample_code"]
            out = sample["out"]

            if err_type == "runtime_error":
                continue

            sample["fix_episodes"] = []
            fixed_with = None

            samples_to_fix_in_episode = [(err_type, sample_code, out)]
            samples_to_fix_in_next_episode = []

            occurred_errors = [{
                "type": err_type,
                "out": out,
            }]

            def append_fix_sample(fix_sample, occurred_errors, samples_to_fix_in_next_episode):
                err_type = fix_sample["type"]
                out = fix_sample["out"]
                error_already_occurred = any([True for err in occurred_errors if err["type"] == err_type and len(
                    err["out"]) == len(out) and all([e in out for e in err["out"]])])

                if not error_already_occurred:
                    occurred_errors.append({
                        "type": err_type,
                        "out": out,
                    })

                    samples_to_fix_in_next_episode.append(
                        (err_type, fix_sample["sample_code"], out))
                else:
                    sample_code = fix_sample["sample_code"]
                    print(
                        f"Skip\n{sample_code}\n in next episode, already occurred error {err_type} {out}.")

            for episode in range(3):

                for (err_type, sample_code, out) in samples_to_fix_in_episode:

                    print(
                        f"Fix sample:\n{sample_code}\nwith error {err_type}.\nIn episode {episode}.")

                    if err_type == "runtime_error":
                        print("Skip runtime error.")
                        continue

                    fix_result = await run_episode(
                        episode, sample_code, err_type, out, entry_point, instruction, declaration, test, helper)

                    sample["fix_episodes"].append(fix_result)

                    if fix_result["fixed"]:
                        fixed_with = fix_result["best_sample"]
                        break

                    # if there is no best sample use both samples in next episode
                    if fix_result["best_sample"] is not None:
                        fix_sample = fix_result["best_sample"]
                        append_fix_sample(
                            fix_sample, occurred_errors, samples_to_fix_in_next_episode)

                    # else use all samples in next episode
                    else:
                        for fix_sample in fix_result["samples"]:
                            append_fix_sample(
                                fix_sample, occurred_errors, samples_to_fix_in_next_episode)

                if fixed_with is not None:
                    break

                samples_to_fix_in_episode = samples_to_fix_in_next_episode
                samples_to_fix_in_next_episode = []

                print(
                    f"Next episode samples to fix: {samples_to_fix_in_episode}")

            if fixed_with is not None:
                sample["fixed_with"] = fixed_with

            with open(INPUT_FILE + "_fixed.json", "w") as f:
                json.dump(data, f, indent=4)

            exit()

if __name__ == "__main__":
    asyncio.run(main())
