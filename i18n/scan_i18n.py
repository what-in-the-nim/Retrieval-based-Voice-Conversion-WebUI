"""
This script will scan through every python file in the project
and extract all strings that are wrapped in the `i18n` function.
It will then compare the extracted strings with the standard file.
Any missing or unused strings will be printed out and the standard
file (zh_CN.json) will be updated accordingly.
"""
import ast
import glob
import json
from collections import OrderedDict
from typing import List


def extract_i18n_strings(node) -> List[str]:
    """Extract all i18n strings from an AST node."""
    i18n_strings = list()

    # Check if the node is a call to the i18n function
    # e.g. i18n("Hello World")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                i18n_strings.append(arg.s)

    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings

def find_all_i18n_strings() -> List[str]:
    """Find all unique i18n strings inside the project."""
    strings = list()
    # Get all python files in the project.
    python_files = glob.iglob("**/*.py", recursive=True)
    for python_file in python_files:
        # Read the code inside the python file.
        with open(python_file, "r", encoding="utf-8") as file:
            code = file.read()

            # Check if the code imports I18nAuto
            if "I18nAuto" in code:
                # Parse the code into an AST (Abstract Syntax Tree)
                tree = ast.parse(code)
                # Extract the i18n strings from the AST
                i18n_strings = extract_i18n_strings(tree)
                # Print the number of i18n strings found in the file.
                total_strings = len(i18n_strings)
                print(f"{python_file} contains {total_strings} i18n strings.")
                # Add the i18n strings to the list of strings.
                strings.extend(i18n_strings)
    return strings

if __name__ == "__main__":
    # Find all i18n strings in the project.
    strings = find_all_i18n_strings()
    # Remove duplicate strings.
    code_keys = set(strings)

    print("Total unique i18n keys:", len(code_keys))

    # Load the standard file
    standard_file = "i18n/locale/zh_CN.json"
    with open(standard_file, "r", encoding="utf-8") as file:
        standard_data = json.load(file, object_pairs_hook=OrderedDict)
    # Get the keys of the standard file
    standard_keys = set(standard_data.keys())

    # Print unused keys
    unused_keys = standard_keys - code_keys
    print("Unused keys:", len(unused_keys))
    for unused_key in unused_keys:
        print("\t", unused_key)

    # Print missing keys
    missing_keys = code_keys - standard_keys
    print("Missing keys:", len(missing_keys))
    for missing_key in missing_keys:
        print("\t", missing_key)

    # Create a dictionary of all code keys we found
    code_keys_dict = OrderedDict()
    for s in strings:
        code_keys_dict[s] = s

    # Update the standard file with the new code keys
    with open(standard_file, "w", encoding="utf-8") as file:
        json.dump(code_keys_dict, file, ensure_ascii=False, indent=4, sort_keys=True)
        file.write("\n")
