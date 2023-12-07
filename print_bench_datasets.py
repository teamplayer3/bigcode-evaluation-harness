from datasets import load_dataset


DATASET = "HUMANEVALPACK"  # "MULTIPL-E"
TASKS = [22, 32, 38, 50, 103, 125, 137, 148]

if DATASET == "HUMANEVALPACK":
    tasks = [s for s in load_dataset(
        "bigcode/humanevalpack", "rust")["test"]]
elif DATASET == "MULTIPL-E":
    tasks = [s for s in load_dataset(
        "nuprl/MultiPL-E", "humaneval-rs")["test"]]


for task in tasks:

    task_idx = int(task["task_id"].split("/")[1] if DATASET ==
                   "HUMANEVALPACK" else task["name"].split("_")[1])

    if task_idx not in TASKS:
        continue

    print(
        f"------------------------- task {task_idx} -------------------------")

    if DATASET == "MULTIPL-E":
        print(f"\nprompt:\n {task['prompt']}")
        print(f"\ntests:\n {task['tests']}")
        print("\n\n")

    elif DATASET == "HUMANEVALPACK":
        print(f"\ninstruction:\n {task['instruction']}")
        print(f"\nprompt:\n {task['prompt']}")
        print(f"\ntests:\n {task['test']}")
        print("\n\n")
