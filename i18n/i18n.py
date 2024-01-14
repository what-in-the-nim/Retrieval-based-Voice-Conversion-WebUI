import json
import locale
import os

from typing import Any, Dict

class I18nAuto:
    """
    A class for handling internationalization (i18n) and language translation.

    Use this class to translate a zh_CN to other languages.

    Example:
    --------
        >>> i18n = I18nAuto("en_US")
        >>> i18n("你好")
        "Hello"
    """

    def __init__(self, language: str = "auto") -> None:
        # If the language is auto, get the system's language.
        if language == "auto":
            # getlocale can't identify the system's language ((None, None))
            language = locale.getdefaultlocale()[0]
        # If the language is not supported, fallback to English.
        filename = I18nAuto.get_file_location(language)
        if not os.path.exists(filename):
            language = "en_US"

        self.language = language
        self.language_dict = self.load_language_dict(language)

    def __call__(self, key: str) -> str:
        """Translate a key to the current language."""
        # If the translation is not found, return the original word.
        translation = self.language_dict.get(key, key)
        return translation

    def __repr__(self) -> str:
        """Return the string representation of the current language."""
        return "Use Language: " + self.language

    @staticmethod
    def load_language_dict(language : str = "en_US") -> Dict[str, Any]:
        """Load language dict from its json file."""
        language_file = I18nAuto.get_file_location(language)
        with open(language_file, "r", encoding="utf-8") as file:
            language_list = json.load(file)
        return language_list

    @staticmethod
    def get_file_location(language : str) -> str:
        """Get the file location of a language's json file."""
        return f"./i18n/locale/{language}.json"
