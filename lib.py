import tree_sitter
import subprocess
import os
import numpy as np
from typing import Union, List, Iterable, Tuple
import itertools
import json


def flatten(l):

    out = []

    for item in l:
        if isinstance(item, list):
            for el in item:
                out.append(el)
        else:
            out.append(item)

    return out


def create_project_if_not_exists(proj_name: str, deps: List[str], lib=False):
    if os.path.exists(proj_name) == False:
        args = ["cargo", "init", proj_name]
        if lib:
            args.append("--lib")
        subprocess.run(args)

        add_dependencies(proj_name, deps)


def execute_project(proj_name: str, timeout: float = None) -> (str, str):
    p = subprocess.Popen(
        ["cargo", "run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=proj_name)

    stdout, stderr = p.communicate(timeout=timeout)

    p.kill()

    return (stdout.decode("utf-8"), stderr.decode("utf-8"))


def test_project(proj_name: str, timeout: float = None) -> Iterable[dict]:
    p = subprocess.Popen(
        ["cargo", "test", "--message-format", "json", "--", "-Z", "unstable-options", "--format", "json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=proj_name, universal_newlines=True)

    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise subprocess.TimeoutExpired(p.args, timeout)

    def mapping(v):
        try:
            return json.loads(v)

        except json.decoder.JSONDecodeError:
            return None

    return filter(lambda x: x is not None, map(mapping, stdout.split("\n")))


def run_rust_code_analysis(proj_name: str) -> dict | None:
    p = subprocess.Popen(
        ["rust-code-analysis-cli", "-m", "-O", "json", "-p", "src/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=proj_name, universal_newlines=True)

    stdout, stderr = p.communicate()

    try:
        return json.loads(stdout)
    except json.decoder.JSONDecodeError:
        return None


def run_clippy(proj_name: str, lints: List[str]) -> Iterable[dict]:

    denies = []
    for lint in lints:
        denies.append(["-D", lint])

    cmd = ["cargo", "clippy", "--message-format", "json", "--"]
    cmd.extend(denies)

    p = subprocess.Popen(flatten(cmd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=proj_name, universal_newlines=True)

    stdout, stderr = p.communicate()

    def mapping(v):
        try:
            return json.loads(v)

        except json.decoder.JSONDecodeError:
            return None

    return filter(lambda x: x is not None, map(mapping, stdout.split("\n")))


SOLUTION_START = "// -------------------------------- solution --------------------------------"
CHECKS_START = "// --------------------------------- checks ---------------------------------"
MAIN_START = "// ---------------------------------- main ----------------------------------"


def create_main_file(proj_name: str, solution: str, checks: str, obj_function_name: str) -> str:
    with open(proj_name + "/src/main.rs", "w") as f:
        f.write(f"""
{SOLUTION_START}

{solution}

{CHECKS_START}

{checks}

{MAIN_START}

fn main() {{
    check({obj_function_name});
}}
""")


def create_test_file(proj_name: str, solution: str, test: str, declaration: str, helper: str) -> str:
    with open(proj_name + "/src/lib.rs", "w") as f:
        f.write(f"""
#![allow(unused_imports)]
#![allow(deprecated)]
#![allow(dead_code)]
{helper}
use std::{{cmp::Ordering, collections::VecDeque}};
{declaration}
{solution}

{test}
""")


def create_clippy_file(proj_name: str, solution: str, allow: [str]) -> str:

    allow_str = ""
    for a in allow:
        allow_str += f"#![allow({a})]\n"

    with open(proj_name + "/src/lib.rs", "w") as f:
        f.write(allow_str + solution)


def create_solution_file(proj_name: str, solution: str) -> str:
    with open(proj_name + "/src/lib.rs", "w") as f:
        f.write(solution)


def delete_proj(proj_name: str):
    subprocess.Popen(["rm", "-rf", proj_name]).wait()


class CompilerError:

    def __init__(self, ident, short_message: str, message: str, code: str) -> None:
        self.ident = ident
        self.short_message = short_message
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.code + " : " + self.short_message

    @classmethod
    def from_str_partial(cls, str_in: str):
        splitted = str_in.split(" : ")
        cls.ident = None
        cls.message = None
        cls.code = splitted[0]
        cls.short_message = splitted[1]

        return cls


class CompilerWarning:

    def __init__(self, ident, short_message: str, message: str) -> None:
        self.ident = ident
        self.short_message = short_message
        self.message = message

    def __str__(self) -> str:
        return self.short_message


class ClippyWarning:

    def __init__(self, ident: str, code: str, short_message: str, message: str) -> None:
        self.ident = ident
        self.code = code
        self.short_message = short_message
        self.message = message

    def __str__(self) -> str:
        return self.code + " : " + self.short_message


class TestRuntimeError:

    def __init__(self, error: str) -> None:
        self.error = error

    def __str__(self) -> str:
        return self.error


class Timeout:

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "Timeout"


def validate_compiler_output(output: List[dict]) -> (List[CompilerWarning], None | List[CompilerError], float | TestRuntimeError):

    build_success = [o for o in output if "reason" in o and o["reason"]
                     == "build-finished"][0]["success"]

    def compile_warning_mapper(compiler_msg) -> CompilerWarning:
        message_super = compiler_msg["message"]
        level = message_super["level"]

        if level == "warning" and len(message_super["children"]) != 0:
            spans = message_super["spans"]
            message = message_super["message"]
            spans_code = "_".join(map(
                lambda s: f"[{s['line_start']}:{s['column_start']}-{s['line_end']}:{s['column_end']}]", spans))
            return CompilerWarning(f"{message}/{spans_code}", message, message_super["rendered"])

    compiler_warnings = filter(lambda x: x is not None, [
        compile_warning_mapper(o) for o in output if "reason" in o and o["reason"] == "compiler-message"])

    unique_warnings = list(
        {each.ident: each for each in compiler_warnings}.values())

    if not build_success:

        def compiler_error_mapper(compiler_msg) -> CompilerError:
            # print(compiler_msg)
            message_super = compiler_msg["message"]
            level = message_super["level"]
            if level == "error" and not message_super["message"].startswith("aborting due to"):
                code = message_super["code"]["code"] if message_super["code"] is not None else "-"
                spans = message_super["spans"]
                spans_code = "_".join(map(
                    lambda s: f"[{s['line_start']}:{s['column_start']}-{s['line_end']}:{s['column_end']}]", spans))
                return CompilerError(f"{code}/{spans_code}", message_super["message"], message_super["rendered"], code)

        compiler_errors = filter(lambda x: x is not None, [
            compiler_error_mapper(o) for o in output if o["reason"] == "compiler-message"])

        unique_errors = list(
            {each.ident: each for each in compiler_errors}.values())

        return (unique_warnings, unique_errors, None)

    test_suit_result_filtered = [o for o in output if "type" in o and o["type"]
                                 == "suite" and (o["event"] == "failed" or o["event"] == "ok") and (o["passed"] > 0 or o["failed"] > 0)]

    if len(test_suit_result_filtered) == 0:
        return (unique_warnings, None, TestRuntimeError("runtime error"))

    test_suit_result = test_suit_result_filtered[0]

    if test_suit_result["event"] == "failed":
        test_fail = [o for o in output if "type" in o and o["type"]
                     == "test" and o["event"] == "failed"][0]
        return (unique_warnings, None, TestRuntimeError(test_fail["stdout"]))

    return (unique_warnings, None, test_suit_result["exec_time"])


def validate_clippy_output(output:  List[dict]) -> List[ClippyWarning]:

    def compile_message_mapper(compiler_msg) -> ClippyWarning:
        message_super = compiler_msg["message"]
        level = message_super["level"]

        if level in ["warning", "error"] and message_super["code"] is not None:
            spans = message_super["spans"]
            message = message_super["message"]
            code = message_super["code"]["code"]
            if not code.startswith("clippy::"):
                return None
            spans_code = "_".join(map(
                lambda s: f"[{s['line_start']}:{s['column_start']}-{s['line_end']}:{s['column_end']}]", spans))
            return ClippyWarning(f"{message}/{spans_code}", code, message, message_super["rendered"])

    clippy_warnings = filter(lambda x: x is not None, [
        compile_message_mapper(o) for o in output if "reason" in o and o["reason"] == "compiler-message"])

    return list(clippy_warnings)


def run_in_cargo_project(proj_name: str, checks: str, solution: str, obj_function_name: str, timeout: float = 10.0, delete_after: bool = False) -> None | CompilerError | TestRuntimeError | Timeout:
    create_project_if_not_exists(proj_name)

    create_main_file(proj_name, solution, checks, obj_function_name)

    try:
        stdout, stderr = execute_project(proj_name, timeout)
    except subprocess.TimeoutExpired:
        delete_proj(proj_name)
        return Timeout()
    finally:
        if delete_after:
            delete_proj(proj_name)

    return validate_compiler_output(stderr)


def clippy_lint_in_project(proj_name: str, solution: str, delete_after: bool = False) -> List[ClippyWarning]:
    lints = [
        "clippy::zero_sized_map_values",
        "clippy::wildcard_enum_match_arm",
        "clippy::arithmetic_side_effects",
        "clippy::unnecessary_cast",
        "clippy::cast_lossless",
        "clippy::cast_possible_truncation",
        "clippy::cast_possible_wrap",
        "clippy::cast_precision_loss",
        "clippy::cast_ptr_alignment"
        "clippy::cast_sign_loss",
        "clippy::fn_to_numeric_cast",
        "clippy::ref_to_mut",
        "clippy::ptr_as_ptr",
        "clippy::char_lit_as_u8",
        "clippy::as_ptr_cast_mut",
        "clippy::as_underscore",
        "clippy::assertions_on_result_states",
        "clippy::bool_to_int_with_if",
        "clippy::borrow_as_ptr",
        "clippy::branches_sharing_code",
        "clippy::case_sensitive_file_extension_comparisons",
        "clippy::checked_conversions",
        "clippy::clear_with_drain",
        "clippy::clone_on_ref_ptr",
        "clippy::cloned_instead_of_copied",
        "clippy::cognitive_complexity",
        "clippy::collection_is_never_read",
        "clippy::copy_iterator",
        "clippy::default_numeric_fallback",
        "clippy::default_trait_access",
        "clippy::default_union_representation",
        "clippy::deref_by_slicing",
        "clippy::derive_partial_eq_without_eq",
        "clippy::empty_drop",
        "clippy::empty_enum",
        "clippy::empty_structs_with_brackets",
        "clippy::equatable_if_let",
        "clippy::expect_used",
        "clippy::explicit_deref_methods",
        "clippy::explicit_into_iter_loop",
        "clippy::explicit_iter_loop",
        "clippy::filetype_is_file",
        "clippy::filter_map_next",
        "clippy::flat_map_option",
        "clippy::float_cmp",
        "clippy::float_cmp_const",
        "clippy::fn_params_excessive_bools",
        "clippy::fn_to_numeric_cast_any",
        "clippy::format_push_string",
        "clippy::from_iter_instead_of_collect",
        "clippy::get_unwrap",
        "clippy::host_endian_bytes",
        "clippy::if_not_else",
        "clippy::if_then_some_else_none",
        "clippy::ignored_unit_patterns",
        "clippy::implicit_clone",
        "clippy::imprecise_flops",
        "clippy::index_refutable_slice",
        "clippy::indexing_slicing",
        "clippy::inefficient_to_string",
        "clippy::invalid_upcast_comparisons",
        "clippy::items_after_statements",
        "clippy::iter_on_single_items",
        "clippy::iter_with_drain",
        "clippy::large_digit_groups",
        "clippy::large_stack_arrays",
        "clippy::large_stack_frames",
        "clippy::let_underscore_must_use",
        "clippy::let_underscore_untyped",
        "clippy::lossy_float_literal",
        "clippy::manual_assert",
        "clippy::manual_clamp",
        "clippy::manual_instant_elapsed",
        "clippy::manual_let_else",
        "clippy::manual_ok_or",
        "clippy::manual_string_new",
        "clippy::map_err_ignore",
        "clippy::map_unwrap_or",
        "clippy::match_bool",
        "clippy::match_on_vec_items",
        "clippy::match_same_arms",
        "clippy::match_wild_err_arm",
        "clippy::match_wildcard_for_single_variants",
        "clippy::maybe_infinite_iter",
        "clippy::mem_forget",
        "clippy::mixed_read_write_in_expression",
        "clippy::multiple_unsafe_ops_per_block",
        "clippy::mut_mut",
        "clippy::mutex_atomic",
        "clippy::mutex_integer",
        "clippy::naive_bytecount",
        "clippy::needless_bitwise_bool",
        "clippy::needless_collect",
        "clippy::needless_continue",
        "clippy::needless_for_each",
        "clippy::needless_pass_by_ref_mut",
        "clippy::needless_pass_by_value",
        "clippy::needless_raw_string_hashes",
        "clippy::needless_raw_strings",
        "clippy::no_effect_underscore_binding",
        "clippy::nonstandard_macro_braces",
        "clippy::option_if_let_else",
        "clippy::option_option",
        "clippy::or_fun_call",
        "clippy::panic",
        "clippy::panic_in_result_fn",
        "clippy::path_buf_push_overwrite",
        "clippy::pattern_type_mismatch",
        "clippy::ptr_as_ptr",
        "clippy::ptr_cast_constness",
        "clippy::range_minus_one",
        "clippy::range_plus_one",
        "clippy::rc_buffer",
        "clippy::rc_mutex",
        "clippy::read_zero_byte_vec",
        "clippy::readonly_write_lock",
        "clippy::redundant_clone",
        "clippy::redundant_closure_for_method_calls",
        "clippy::redundant_else",
        "clippy::redundant_type_annotations",
        "clippy::ref_binding_to_reference",
        "clippy::ref_option_ref",
        "clippy::ref_patterns",
        "clippy::rest_pat_in_fully_bound_structs",
        "clippy::same_functions_in_if_condition",
        "clippy::semicolon_if_nothing_returned",
        "clippy::shadow_same",
        "clippy::shadow_unrelated",
        "clippy::significant_drop_in_scrutinee",
        "clippy::significant_drop_tightening",
        "clippy::similar_names",
        "clippy::single_char_lifetime_names",
        "clippy::single_match_else",
        "clippy::stable_sort_primitive",
        "clippy::str_to_string",
        "clippy::string_add",
        "clippy::string_add_assign",
        "clippy::string_lit_as_bytes",
        "clippy::string_lit_chars_any",
        "clippy::string_to_string",
        "clippy::suboptimal_flops",
        "clippy::suspicious_operation_groupings",
        "clippy::suspicious_xor_used_as_pow",
        "clippy::too_many_lines",
        "clippy::transmute_ptr_to_ptr",
        "clippy::transmute_undefined_repr",
        "clippy::trivially_copy_pass_by_ref",
        "clippy::try_err"
    ]

    create_project_if_not_exists(proj_name, ["md5", "regex", "rand"], lib=True)

    create_clippy_file(proj_name, solution, [
                       "clippy::single_component_path_imports"])

    out = run_clippy(proj_name, lints)

    if delete_after:
        delete_proj(proj_name)

    return validate_clippy_output(out)


def validate_code_analysis_output(output: dict) -> dict:
    return output["metrics"]


def analyse_metrics(proj_name: str, solution: str, delete_after: bool = False) -> dict | None:
    create_project_if_not_exists(proj_name, ["md5", "regex", "rand"], lib=True)

    create_solution_file(proj_name, solution)

    out = run_rust_code_analysis(proj_name)

    if delete_after:
        delete_proj(proj_name)

    return validate_code_analysis_output(out) if out is not None else None


def add_dependencies(proj_name: str, deps: List[str]):
    args = ["cargo", "add"]
    args.extend(deps)
    subprocess.run(args, cwd=proj_name)
    print("added dependencies")


def test_in_cargo_project(proj_name: str, test: str, solution: str, declaration: str, helper: str, timeout: float = 10.0, delete_after: bool = False) -> Tuple[List[CompilerWarning], None | List[CompilerError], float | TestRuntimeError] | Timeout:
    create_project_if_not_exists(proj_name, ["md5", "regex", "rand"], lib=True)

    create_test_file(proj_name, solution, test, declaration, helper)

    try:
        out = list(test_project(proj_name, timeout))
    except subprocess.TimeoutExpired:
        return Timeout()
    finally:
        if delete_after:
            delete_proj(proj_name)

    return validate_compiler_output(out)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    From openAi humeneval repository:
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def init_tree_sitter_parser() -> tree_sitter.Parser:
    tree_sitter.Language.build_library(
        "tree-sitter/build/my-languages.so",

        [
            "tree-sitter/tree-sitter-rust",
        ]
    )

    RUST_LANGUAGE = tree_sitter.Language(
        "tree-sitter/build/my-languages.so", "rust")

    parser = tree_sitter.Parser()
    parser.set_language(RUST_LANGUAGE)

    return parser


def generate_ast_of_source(parser: tree_sitter.Parser, source: str) -> tree_sitter.Tree:
    return parser.parse(bytes(source, "utf-8"))


class ObjCall:

    def __init__(self, func_name: str) -> None:
        self.func_name = func_name

    def to_call_str(self) -> str:
        return self.func_name

    def __str__(self) -> str:
        return f"ObjCall({self.func_name})"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ObjCall):
            return self.func_name == o.func_name
        return False


class MacroCall:
    def __init__(self, func_name: str) -> None:
        self.func_name = func_name

    def to_call_str(self) -> str:
        return self.func_name

    def __str__(self) -> str:
        return f"MacroCall({self.func_name})"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MacroCall):
            return self.func_name == o.func_name
        return False


class StaticCall:
    def __init__(self, obj_name: str, func_name: str) -> None:
        self.func_name = func_name
        self.obj_name = obj_name

    def to_call_str(self) -> str:
        return f"{self.obj_name}::{self.func_name}"

    def __str__(self) -> str:
        return f"StaticCall({self.obj_name}, {self.func_name})"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StaticCall):
            return self.obj_name == o.obj_name and self.func_name == o.func_name
        return False


def func_call_from_str(func_str: str) -> ObjCall | StaticCall | MacroCall:
    if "::" in func_str:
        splitted = func_str.split("::")
        return StaticCall(splitted[0], splitted[1])

    if func_str.endswith("!"):
        return MacroCall(func_str[:-1])

    return ObjCall(func_str)


def is_rust_type(type: str):

    types = [
        "bool",
        "char",
        "str",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "f32",
        "f64",
        "isize",
        "usize",
        "array",
        "Vec",
        "Option",
        "Result",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "String",
        "Box",
        "Rc",
        "Arc",
        "Cell",
        "RefCell",
        "Mutex",
        "RwLock",
        "Cow",
        "PathBuf",
        "OsString",
        "Duration",
        "Instant",
        "SystemTime",
        "IpAddr",
        "SocketAddr",
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "Shutdown",
    ]

    return type in types


def is_std_func(func: Union[ObjCall | StaticCall | MacroCall]) -> bool:
    if isinstance(func, ObjCall):
        if func.func_name in ["swap_bytes", "copied", "wrapping_add", "clone", "to_string", "is_match", "to_owned", "to_lowercase", "next", "unwrap", "sort_by", "iter", "into_iter", "push_str", "sort_by", "sum", "chars", "count", "filter", "map", "collect", "split_whitespace", "join", "sum", "is_digit", "pow", "is_ascii_alphabetic", "len", "split", "extend", "push", "take", "contains", "unwrap_or", "into_iter", "to_ascii_lowercase", "is_empty", "is_whitespace", "parse", "fract", "sort_unstable", "enumerate", "skip", "rev", "is_uppercase", "is_alphabetic", "all", "min", "partial_cmp", "zip", "last", "is_lowercase", "abs", "min", "reverse", "floor", "ceil", "round", "char_indices", "is_ascii_lowercase", "to_uppercase", "map_or", "min_by_key", "cloned", "insert", "clear", "max", "expect", "starts_with", "to_uppercase", "ok", "filter_map", "position", "sort_unstable_by_key", "peek", "peekable", "fold", "windows", "saturating_add", "checked_add", "saturating_sub", "is_none", "checked_mul", "count_ones", "parent", "into_owned", "file_name", "unwrap_or_default", "product", "ends_with", "iter_mut", "exists", "to_string_lossy", "ancestors", "extend_from_slice"]:
            return True
    if isinstance(func, StaticCall):
        if func.obj_name in ["u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32", "f64"] and func.func_name in ["from", "try_from", "from_le"] or func.obj_name in ["String", "Vec", "HashMap", "HashSet", "PathBuf"] and func.func_name == "new" or func.obj_name in ["String", "PathBuf"] and func.func_name == "from" or func.obj_name == "Vec" and func.func_name in ["with_capacity"]:
            return True
    if isinstance(func, MacroCall):
        if func.func_name in ["format", "println", "eprintln", "panic", "vec", "matches", "unreachable"]:
            return True

    print(f"unknown function type {str(func)}")

    return False


def extract_function_calls(node: tree_sitter.Node) -> [ObjCall | StaticCall | MacroCall]:
    # print(node.type)

    calls = []
    if node.type == "identifier":
        left_ident = node.text.decode("utf-8")
        if is_rust_type(left_ident):
            sibling = node.next_sibling
            if sibling is not None and sibling.type == "::":
                sibling = sibling.next_sibling
                if sibling is not None and sibling.type == "identifier":
                    right_ident = sibling.text.decode("utf-8")
                    calls.append(StaticCall(left_ident, right_ident))

    elif node.type == "call_expression":
        child = node.children[0]
        if child.type == "field_expression":
            left_child = child.children[0]
            right_child = child.children[2]

            # for foo.bar() or (0..2).sum()
            if left_child.type == "identifier" or left_child.type == "call_expression" or left_child.type == "parenthesized_expression" and right_child.type == "field_identifier":
                calls.append(ObjCall(right_child.text.decode("utf-8")))

    elif node.type == "macro_invocation":
        macro_name = node.children[0].text.decode("utf-8")
        calls.append(MacroCall(macro_name))

    for child in node.children:
        calls.extend(extract_function_calls(child))

    return calls


def extract_lang_keywords(node: tree_sitter.Node) -> [str]:
    keywords = []

    # print(node.type)

    if node.type == "if_expression":
        try:
            if next(i for i in node.children if i.type == "let_condition") is not None:
                if node.parent.type == "let_declaration":
                    keywords.append("let_if_let")
                else:
                    keywords.append("if_let")
        except StopIteration:
            if node.parent.type == "let_declaration":
                keywords.append("let_if")

    elif node.type == "match_expression":
        if node.parent.type == "let_declaration":
            keywords.append("let_match")
        else:
            keywords.append("match")
    elif node.type == "loop_expression":
        def has_break(node) -> bool:
            if node.type == "break_expression":
                return True

            for n in node.children:
                if has_break(n):
                    return True

            return False
        try:
            block = next(
                i for i in node.children if i.type == "block")
            if has_break(block):
                if node.parent.type == "let_declaration":
                    keywords.append("let_loop_break")
                else:
                    keywords.append("loop_break")
        except StopIteration:
            pass
    elif node.type == "range_expression":
        keywords.append("range")

    if node.type == "unsafe_block":
        keywords.append("unsafe")

    if node.type == "function_item":
        try:
            block = next(i for i in node.children if i.type == "block")

            expressions = [e for e in block.children if e.type ==
                           "expression_statement"]
            if len(expressions) != 0 and next(e for e in expressions[-1].children if e.type == "return_expression") is not None:
                keywords.append("explicit_return")
        except StopIteration:
            pass

    for child in node.children:
        keywords.extend(extract_lang_keywords(child))

    return keywords


def decode_nodes_text(nodes: tree_sitter.Node) -> [str]:
    return list(map(lambda n: n.text.decode("utf-8"), nodes))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, TestRuntimeError):
            return str(obj)
        if isinstance(obj, CompilerError):
            return str(obj)
        if isinstance(obj, CompilerWarning):
            return str(obj)
        if isinstance(obj, Timeout):
            return str(obj)
        return super(MyEncoder, self).default(obj)


def evaluate_compile_errors(log: dict) -> dict:

    compiler_outs = log["compiler_stats"]["compiler_out"]

    compiler_errors = dict()
    for compiler_out in compiler_outs:
        for fault in compiler_out["faults"]:
            if fault["type"] == "compiler_error":
                outs = [CompilerError.from_str_partial(
                    f) for f in fault["out"]]
                already_in = []

                def error_filter(error):
                    if error not in already_in:
                        already_in.append(error)
                        return True
                    return False

                outs_filtered_unique = list(filter(error_filter, outs))

                for error in outs_filtered_unique:
                    if error.code == "-":
                        if "unknown" not in compiler_errors:
                            compiler_errors["unknown"] = 0
                        compiler_errors["unknown"] += 1
                    else:
                        if error.code not in compiler_errors:
                            compiler_errors[error.code] = 0
                        compiler_errors[error.code] += 1

    return compiler_errors


def evaluate_clippy_warnings(log: dict, code_filter: None | List[str] = None) -> dict:

    compiler_outs = log["compiler_stats"]["compiler_out"]
    clippy = dict()

    for compiler_out in compiler_outs:
        if "passed_samples" not in compiler_out:
            continue
        for passed in compiler_out["passed_samples"]:
            clippy_warnings = passed["clippy_warnings"]

            for clippy_warning in clippy_warnings:
                code = clippy_warning["code"]
                if code_filter is not None and code in code_filter:
                    continue
                if code not in clippy:
                    clippy[code] = 0

                clippy[code] += 1

    return clippy
