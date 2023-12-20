
## 1

I generated samples with the prompt style:


```
You are an expert Rust programmer, and here is your task: Write a Rust function `generate_integers(a: i32, b: i32) -> Vec<i32>` to solve the following problem:
Given two positive integers a and b, return the even digits between a
and b, in ascending order.
Your code should start with ```rust and end with ```.
```

result: [10, 1, 10, 4, 10, 9, 0, 8, 10, 10, 0, 5, 10, 2, 0, 10, 0, 9, 0, 0, 7, 0, 4, 5, 10, 2, 0, 7, 10, 8, 10, 5, 0, 0, 3, 2, 10, 1, 0, 0, 0, 0, 10, 10, 10, 0, 0, 10, 10, 4, 0, 0, 10, 10, 10, 10, 5, 10, 1, 3, 10, 0, 8, 5, 0, 0, 5, 0, 0, 1, 0, 0, 6, 6, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 4, 0, 2, 0, 0, 0, 0, 0, 6, 0, 0, 7, 3, 0, 0, 0, 0, 1, 10, 0, 2, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 0, 0, 1, 6, 0, 6, 0, 1, 0, 0, 1, 1, 6, 0, 0, 0, 0]
pass@1: 24.939

top_p = 0.95

## filtered out all samples where 0 passed

generated new samples with the prompt style:

```
<s>[INST] <<SYS>>\nProvide answers in Rust.\n<</SYS>>\n\nYou are an expert Rust programmer, and here is your task: Write a Rust function `generate_integers(a: i32, b: i32) -> Vec<i32>` to solve the following problem:
Given two positive integers a and b, return the even digits between a
and b, in ascending order.

Your code should start with ```rust and end with ```.[/INST]
```

evaluate the result and merged it into the old:

merged: [10, 1, 10, 4, 10, 9, 0, 8, 10, 10, 0, 5, 10, 2, 0, 10, 0, 9, 0, 0, 7, 0, 4, 5, 10, 2, 0, 7, 10, 8, 10, 5, 0, 0, 3, 2, 10, 1, 0, 0, 0, 0, 10, 10, 10, 0, 0, 10, 10, 4, 0, 0, 10, 10, 10, 10, 5, 10, 1, 3, 10, 0, 8, 5, 0, 0, 5, 0, 0, 1, 0, 0, 6, 6, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 4, 0, 2, 0, 0, 0, 0, 0, 6, 0, 0, 7, 3, 0, 0, 0, 0, 1, 10, 0, 2, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 0, 0, 1, 6, 0, 6, 0, 1, 0, 0, 1, 1, 6, 0, 0, 0, 0]
pass@1: 27.744

top_p = 0.9

improved 10 tasks

## Use multipl-e dataset

```
<s>[INST] <<SYS>>\nProvide answers in Rust. Your code should start with ```rust and end with ```.\n<</SYS>>\n\n/// remove_vowels is a function that takes string and returns string without vowels.
/// >>> remove_vowels(String::from(""))
/// String::from("")
/// >>> remove_vowels(String::from("abcdef"))
/// String::from("bcdf")
/// >>> remove_vowels(String::from("aaaaa"))
/// String::from("")
/// >>> remove_vowels(String::from("aaBAA"))
/// String::from("B")
/// >>> remove_vowels(String::from("zbcd"))
/// String::from("zbcd")
fn remove_vowels(text: String) -> String {
[/INST]
```

-> got many parser errors because results in format

```
[PYTHON]
def by_length(arr):
    result = []
    for i in range(1, 10):
        if str(i) in arr:
            result.append(str(i))
    return result
[/PYTHON]
[TESTS]
# Test case 1:
assert by_length([]) == []
# Test case 2:
assert by_length([1, 2, 3, 4, 5, 6, 7, 8, 9]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Test case 3:
assert by_length([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Test case 4:
assert by_length([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Test case 5:
assert by_length([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
[/TESTS]
```



# New tests with 7b

## prompt with own dataset

```
/// Given two positive integers a and b, return the even digits between a
/// and b, in ascending order.
/// For example:
/// >>> generate_integers(2, 8)
/// vec![2, 4, 6, 8]
/// >>> generate_integers(8, 2)
/// vec![2, 4, 6, 8]
/// >>> generate_integers(10, 14)
/// Vec::<isize>::new()
fn generate_integers(a: isize, b: isize) -> Vec<isize> {
```

## generations

- often generates asserts

```
        assert_eq!(generate_integers(2, 8), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(8, 2), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(10, 14), Vec::<isize>::new());
    }
fn test_generate_integers() {
        assert_eq!(generate_integers(2, 8), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(8, 2), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(10, 14), Vec::<isize>::new());
    }
```

1. use raw prompt from merged dataset
2. set seed to 0
3. don't remove newline at end of prompt