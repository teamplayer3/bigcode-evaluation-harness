from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from typing import List

import jsonlines


def prompt_template(instruction: str, context: str, function_start: str) -> str:
    return f"Question: {instruction}\n{context}\n\nAnswer:\n{function_start}"


class ParseError(Exception):
    pass


class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str, language: str = "python"):
        raw_content = content
        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        # matcher = CSequenceMatcher(None, prompt, content)
        # tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        # if tag == "insert":
        #     return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                if language == "python":
                    func_prefix = "def"
                elif language == "rust":
                    func_prefix = "fn"
                content = content.split(f"{func_prefix} {entry_point}")[-1]

        if language == "python":
            content_lines = content.splitlines(keepends=True)[1:]
            func_lines = []
            for line in content_lines:
                if line.startswith("    ") or line.startswith("\n"):
                    func_lines.append(line)
                else:
                    break
            content = "".join(func_lines)
        elif language == "rust":
            start_idx = 0
            end_idx = 0
            open_brackets = 1
            in_string = False
            content_out = ""
            in_func = True

            for idx, char in enumerate(content):
                if char == '"':
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        if open_brackets == 0:
                            in_func = True
                        open_brackets += 1
                    elif char == '}':
                        open_brackets -= 1

                if in_func and open_brackets == 0:
                    end_idx = idx
                    in_func = False

                content_out += char

            content = content_out[start_idx:end_idx + 1]

        if len(content.strip()) == 0:
            raise ParseError(f"Prompt is not in content:\n{raw_content}")

        return content


# prompt = """\
# Question: Write a Rust function `below_zero(operations:Vec<i32>) -> bool` to solve the following problem:
# You're given a list of deposit and withdrawal operations on a bank account that starts with
# zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
# at that point function should return True. Otherwise it should return False.


# Answer:
# fn main(){}

# use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
# use rand::Rng;
# use regex::Regex;
# use md5;
# use std::any::{Any, TypeId};

# /*
#  You're given a list of deposit and withdrawal operations on a bank account that starts with
#     zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
#     at that point function should return True. Otherwise it should return False.

# */
# fn below_zero(operations:Vec<i32>) -> bool{\
# """
# raw_out = """// Create a hashmap for storing all the extensions and their respective strengths
#         let mut extn_hash = HashMap::<&str,&i32>::new();

#         for i in 0..extensions.len() {
#                 extn_hash.insert(&extensions[i], &get_strength(&extensions[i]));
#         }

#         // Get the maximum value from the hash map
#         let max_value = get_maximum(&mut extn_hash);

#         // Find out which key has the maximum value
#         let max_key = match extn_hash.iter().find(|(_, v)| *v == max_value){
#                 Some((k,_)) => k,
#                 None => panic!("No keys found")
#         };

#         // Return the resultant string
#         String::from(format!("{}.{}", class_name, max_key));
# }

# // Function to calculate the strength of an extension based on its name
# fn get_strength(extension:&str)->i32{
#         let capitals=count_capitals(extension);
#         let smalls=count_smalls(extension);
#         if capitals==0 && smalls==0 {
#                 0
#         }else{
#                 match capitals-smalls<0 {
#                         true=>-1*capitals+smalls,
#                         false=>capitals-smalls
#                 }
#         }
# }

# // Function to count the capital letters in the extension's name
# fn count_capitals(extension:&str)->u8{
#         let mut counter=0;
#         for c in extension.chars(){
#                 counter+=c.is_ascii_uppercase() as u8;
#         }
#         counter
# }

# // Function to count the small letters in the extension's name
# fn count_smalls(extension:&str)->u8{
#         let mut counter=0;
#         for c in extension.chars(){
#                 counter+=c.is_ascii_lowercase() as u8;
#         }
#         counter
# }

# // Function to get the maximum value from the hashmap
# fn get_maximum<'a>(hash:&'a mut HashMap<&str,&i32>)->i32{
#         let mut max=-99999999999999"""

# print(ContentParser()(prompt, raw_out, "below_zero", "rust"))

FILE_NAME = "completions_rust_humanevalsynthesize"


content_parser = ContentParser()

with jsonlines.open(f"{FILE_NAME}.jsonl") as reader:
    with jsonlines.open(f"{FILE_NAME}_conf.jsonl", mode="w") as writer:
        parser = ContentParser()
        parse_errors = 0
        samples = 0
        for line in reader:
            raw_generations = line["raw_generation"]
            parsed = list()
            for raw_generation in raw_generations:
                samples += 1
                try:
                    parsed_gen = content_parser(prompt=prompt_template(
                        line["instruction"], "", ""), content=raw_generation, entry_point=line["entry_point"], language="rust")
                    parsed.append(parsed_gen)
                except ParseError as e:
                    parse_errors += 1
                    print(e)
                    parsed.append("")
            line["generation"] = parsed

            writer.write(line)

        print("parse error rate:", parse_errors / samples)
