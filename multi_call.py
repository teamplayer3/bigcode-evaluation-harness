import aiohttp
import asyncio


async def post(session: aiohttp.ClientSession, url, headers=dict, prompt=str):
    try:
        async with session.post(url=url, headers=headers, json={
            "inputs": prompt,
            "parameters": {
                "top_p": 0.95,
                "temperature": 0.2,
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
        }) as response:

            if response.status != 200:
                raise Exception(response.text)

            return await response.json()
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e))


async def main(prompt):
    import time
    start = time.time()
    async with aiohttp.ClientSession() as session:
        BEARER = "tEwqBLUbErqsmwkcqfHfbBXjrxSlAGVaPQLLwIUlWPQHZZAYWYzgAJXWtamYFgioQnUeYfMLDXUdolleQIHbQKaCKHdSCothEQcHDUTJGusyUXTnCGPkknpdTJICVRhi"
        headers = {
            "Authorization": f"Bearer {BEARER}",
            "Content-Type": "application/json"
        }
        ret = await asyncio.gather(*[post(session, "https://me9rxdof1htyppbi.us-east-1.aws.endpoints.huggingface.cloud", headers, prompt) for _ in range(20)])

    print("took {} seconds".format(time.time() - start))
    print(ret)
    print("Finalized all. Return is a list of len {} outputs.".format(len(ret)))


prompt = """\
[INST] Write a Rust function `iscube(a:i32) -> bool` to solve the following problem:
Write a function that takes an integer a and returns True
if this ingeger is a cube of some integer number.
Note: you may assume the input is always valid.
[/INST]\
"""


asyncio.run(main(prompt))
