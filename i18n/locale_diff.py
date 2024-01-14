"""
This script will update all language files to have all the keys
to match with standard file (zh_CN.json), any extra keys
will be deleted and any missing keys will be added.
"""

import json
import os
import os.path as op
from collections import OrderedDict

LOCALE_DIR = "locale/"
standard_file = op.join(LOCALE_DIR, "zh_CN.json")

# Find other language files in the locale directory (except the standard file).
language_files = [
    op.join(LOCALE_DIR, f)
    for f in os.listdir(LOCALE_DIR)
    if f.endswith(".json") and f != standard_file
]

# Load the standard file
with open(standard_file, "r", encoding="utf-8") as f:
    standard_data = json.load(f, object_pairs_hook=OrderedDict)

# Loop through each language file
for language_file in language_files:
    # Load the language file
    with open(language_file, "r", encoding="utf-8") as f:
        lang_data = json.load(f, object_pairs_hook=OrderedDict)

    # Find the difference between the language file and the standard file
    diff = set(standard_data.keys()) - set(lang_data.keys())

    miss = set(lang_data.keys()) - set(standard_data.keys())

    # Add any missing keys to the language file
    for key in diff:
        lang_data[key] = key

    # Del any extra keys to the language file
    for key in miss:
        del lang_data[key]

    # Sort the keys of the language file to match the order of the standard file
    lang_data = OrderedDict(
        sorted(lang_data.items(), key=lambda x: list(standard_data.keys()).index(x[0]))
    )

    # Save the updated language file
    with open(language_file, "w", encoding="utf-8") as f:
        json.dump(lang_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
