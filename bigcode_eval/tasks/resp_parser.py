import re


def extract_func_body(inp: str) -> str:
    end_idx = 0
    open_brackets = 1
    in_string = False
    in_char_string = False
    char_str_len = 0
    cycle_buf = ""

    for idx, char in enumerate(inp):
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
            break

        if len(cycle_buf) > 3:
            cycle_buf = cycle_buf[1:]
        cycle_buf += char

    return inp[0:end_idx + 1]


def find_function_names(inp: str) -> [str]:
    pattern = re.compile(
        r"fn\s(?P<func_name>[\w\_]*)[\<\(](.*?)\{", re.MULTILINE | re.DOTALL)

    return list(map(lambda x: x.group("func_name"), pattern.finditer(inp)))


def find_functions(inp: str) -> [(dict)]:
    pattern = re.compile(
        r"fn\s(?P<func_name>[\w\_]*)[\<\(](.*?)\{", re.MULTILINE | re.DOTALL)

    functions = []

    for match in pattern.finditer(inp):
        end_idx = match.end()
        func_head = inp[match.span(0)[0]:match.span(0)[1]]
        func_name = match.group("func_name")
        body = extract_func_body(inp[end_idx + 1:])

        functions.append({
            "name": func_name,
            "head": func_head,
            "body": body
        })

    return functions


class ParseError(Exception):
    pass


class RespParser:

    def __init__(self, remove_entry_func_head: bool = True, pretend_entry_func_included: bool = True, language: str = "rust") -> None:
        self.language = language
        self.remove_entry_func_head = remove_entry_func_head
        self.pretend_entry_func_included = pretend_entry_func_included

    def __call__(self, response: str, entry_point: str, helper_decl: str) -> str:
        content = response
        out = ""

        # remove comments
        if self.language == "rust":
            comment_pattern = ("//", "///", "//!")
        elif self.language == "python":
            comment_pattern = ("#",)

        content = "".join(filter(lambda x: not x.startswith(
            comment_pattern), content.splitlines(keepends=True)))

        # extract codeblock
        if "```" in content:
            content = content.split("```")[1]

        # find functions
        functions = find_functions(content)

        # find entry function
        entry_function_search = list(filter(
            lambda v: v["name"] == entry_point, functions))

        if len(entry_function_search) > 0:
            if not self.remove_entry_func_head:
                out += entry_function_search[0]["head"] + "\n"
            out += entry_function_search[0]["body"] + "\n"
        elif self.pretend_entry_func_included:
            entry_function_body = extract_func_body(content)
            out += entry_function_body + "\n"
        else:
            raise ParseError(
                f"Entry function not found response:\n{response}\nEntry Points:\n{entry_point}\nHelper:\n{helper_decl}")

        # find helper functions
        helper_decl_functions = find_function_names(helper_decl)

        for func in functions:
            if func["name"] not in [entry_point, "main"] and func["name"] not in helper_decl_functions:
                out += func["head"] + "\n"
                out += func["body"] + "\n"

        return out.strip("\n")
