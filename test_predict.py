import os
import requests
from time import sleep
import logging
import argparse
import sys
import json

endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
URI = f"https://api.runpod.ai/v2/{endpoint_id}/run"


def run(prompt, params={}, stream=False):
    request = {
        'prompt': prompt,
        'max_new_tokens': 1800,
        'temperature': 0.3,
        'top_k': 50,
        'top_p': 0.7,
        'repetition_penalty': 1.2,
        'batch_size': 8,
        'stream': stream
    }

    request.update(params)

    print(URI)

    response = requests.post(URI, json=dict(input=request), headers={
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    })

    if response.status_code == 200:
        data = response.json()
        task_id = data.get('id')
        return stream_output(task_id, stream=stream)

    else:
        print(response)
        return None


def stream_output(task_id, stream=False):
    # try:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/stream/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }

    previous_output = ''

    try:
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if len(data['stream']) > 0:
                    new_output = data['stream'][0]['output']

                    if stream:
                        sys.stdout.write(new_output[len(previous_output):])
                        sys.stdout.flush()
                    previous_output = new_output

                if data.get('status') == 'COMPLETED':
                    if not stream:
                        return previous_output
                    break

            elif response.status_code >= 400:
                print(response)
            # Sleep for 0.1 seconds between each request
            sleep(0.1 if stream else 1)
    except Exception as e:
        print(e)
        cancel_task(task_id)


def cancel_task(task_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }
    response = requests.get(url, headers=headers)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runpod AI CLI')
    parser.add_argument(
        '-s', '--stream', action='store_true', help='Stream output')
    parser.add_argument('-p', '--params_json', type=str,
                        help='JSON string of generation params')

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Rust function `mean_absolute_deviation(numbers:Vec<f32>) -> f32` to solve the following problem:
For a given list of input numbers, calculate Mean Absolute Deviation
around the mean of this dataset.
Mean Absolute Deviation is the average absolute difference between each
element and a centerpoint (mean in this case):
MAD = average | x - x_mean |


### Response:
fn main(){}

use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
use rand::Rng;
use regex::Regex;
use md5;
use std::any::{Any, TypeId};

/*
 For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    
*/
fn mean_absolute_deviation(numbers:Vec<f32>) -> f32{"""
    args = parser.parse_args()
    params = json.loads(args.params_json) if args.params_json else "{}"
    import time
    start = time.time()
    print(run(prompt, params=params, stream=args.stream))
    print("Time taken: ", time.time() - start, " seconds")
