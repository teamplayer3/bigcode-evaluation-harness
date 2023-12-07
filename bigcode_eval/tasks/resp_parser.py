import re


class ParseError(Exception):
    pass


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

    if open_brackets != 0:
        raise ParseError("Function body not closed")

    return inp[0:end_idx + 1]


def find_function_names(inp: str) -> [str]:
    pattern = re.compile(
        r"fn\s(?P<func_name>[\w\_]*)[\<\(](.*?)\{", re.MULTILINE | re.DOTALL)

    return list(map(lambda x: x.group("func_name"), pattern.finditer(inp)))


def look_before_functions_for_imports(inp: str, func_head_start_index: int) -> str:
    # Split the text into lines
    lines = inp.split('\n')

    # Find the line containing the index
    current_index = 0
    line_number_for_index = -1
    for i, line in enumerate(lines):
        if current_index + len(line) >= func_head_start_index:
            line_number_for_index = i
            break
        current_index += len(line) + 1  # Adding 1 for the newline character

    # Find the line starting with "use" looking backwards
    first_line_starts_with_use = -1
    for i in range(line_number_for_index, -1, -1):
        if lines[i].startswith("use "):
            first_line_starts_with_use = i

        if lines[i].startswith("}\n"):
            break

    if first_line_starts_with_use == -1:
        return ""
    else:
        return "\n".join(lines[first_line_starts_with_use:line_number_for_index - 1]).strip()

    return


def find_functions(inp: str) -> [(dict)]:
    pattern = re.compile(
        r"fn\s(?P<func_name>[\w\_]*)[\<\(](.*?)\{", re.MULTILINE | re.DOTALL)

    functions = []

    for match in pattern.finditer(inp):
        start_idx = match.start()
        end_idx = match.end()
        func_head = inp[match.span(0)[0]:match.span(0)[1]]
        func_name = match.group("func_name")
        try:
            body = extract_func_body(inp[end_idx + 1:])
            imports = look_before_functions_for_imports(inp, start_idx)
            functions.append({
                "imports": imports,
                "name": func_name,
                "head": func_head,
                "body": body
            })
        except ParseError:
            continue

    return functions


def func_name_variation(entry_point: str) -> [str]:
    from camel_converter import to_snake

    # fix wrong func naming in dataset (e.g, "helloGoo" instead of "hello_goo")
    return [
        entry_point,
        to_snake(entry_point),
        entry_point[0].lower() + entry_point[1:],
    ]


def extract_full_code_blocks(inp: str) -> [str]:
    if "```" not in inp:
        return [inp]

    lines = inp.splitlines(keepends=True)
    code_blocks = []
    current_block = ""
    in_block = False

    for line in lines:
        if line.startswith("```"):
            if in_block:
                code_blocks.append(current_block)
                current_block = ""

            in_block = not in_block

        if in_block:
            current_block += line

    if in_block:
        return [inp]

    return code_blocks


class RespParser:

    def __init__(self, remove_entry_func_head: bool = True, pretend_entry_func_included: bool = True, collect_imports_into_func_body: bool = False, multiple_samples: bool = False, language: str = "rust") -> None:
        self.language = language
        self.remove_entry_func_head = remove_entry_func_head
        self.pretend_entry_func_included = pretend_entry_func_included
        self.collect_imports_into_func_body = collect_imports_into_func_body
        self.multiple_samples = multiple_samples

    def __call__(self, response: str, entry_point: str, helper_decl: str) -> [str]:
        content = response

        # remove comments
        if self.language == "rust":
            comment_pattern = ("//", "///", "//!")
        elif self.language == "python":
            comment_pattern = ("#",)

        content = "".join(filter(lambda x: not x.startswith(
            comment_pattern), content.splitlines(keepends=True)))

        content_blocks = []

        # extract codeblocks
        if "```" in content:
            if self.multiple_samples:
                splitted = content.split("```")
                block_idx = 0
                while len(splitted) > block_idx * 2 + 2:
                    content_blocks.append(splitted[block_idx * 2 + 1])
                    block_idx += 1
            else:
                splitted = content.split("```")

                if len(splitted) % 2 == 0:
                    content_blocks = [content]
                else:
                    content_blocks.append(splitted[1])
        else:
            content_blocks.append(content)

        out_blocks = []

        for content in content_blocks:

            out = ""

            # find functions
            functions = find_functions(content)

            # find entry function
            entry_function_search = list(filter(
                lambda v: any(e for e in func_name_variation(entry_point) if e in v["name"]), functions))

            if len(entry_function_search) > 0:
                entry_function = entry_function_search[0]

                if not self.remove_entry_func_head:
                    if self.collect_imports_into_func_body:
                        out += entry_function["imports"] + "\n"
                    out += entry_function["head"] + "\n"
                elif self.collect_imports_into_func_body:
                    out += "".join(
                        ["    " + l for l in entry_function["imports"].splitlines(keepends=True)]) + "\n"

                out += entry_function["body"] + "\n"
            elif self.pretend_entry_func_included:
                try:
                    entry_function_body = extract_func_body(content)
                    out += entry_function_body + "\n"
                except ParseError:
                    continue
            else:
                continue
            # find helper functions
            helper_decl_functions = find_function_names(helper_decl)

            for func in functions:
                if func["name"] not in [entry_point, "main"] and func["name"] not in helper_decl_functions:
                    out += func["head"] + "\n"
                    out += func["body"] + "\n"

            out = out.strip("\n")

            out_blocks.append(out)

        if len(out_blocks) == 0:
            raise ParseError(
                f"Entry function not found response:\n{response}\nEntry Points:\n{entry_point}\nHelper:\n{helper_decl}")

        return out_blocks
