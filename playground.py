

occurred_errors = [
    {
        "type": "compiler_error",
        "out": ["a", "b"]
    }
]

err_type = "compiler_error"

out = [
    "a",
    "v",
]

res = any([True for err in occurred_errors if err["type"] == err_type and len(
    err["out"]) == len(out) and all([e in out for e in err["out"]])])


print(res)
