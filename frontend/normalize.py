"""
Production Text Normalizer for Speech Synthesis Frontend.
Converts raw text with non-standard words (numbers, currencies, dates, abbreviations, times)
into spoken-form phonetizable text.
"""

import re
from typing import Dict, List, Pattern, Tuple


class TextNormalizer:
    """Locale-aware text normalizer for TTS frontends."""

    def __init__(self, locale: str = "en-US"):
        self.locale = locale
        self._init_rules()

    def _init_rules(self):
        # Units and Currencies
        self.currency_map: Dict[str, Tuple[str, str]] = {
            "$": ("dollar", "dollars"),
            "€": ("euro", "euros"),
            "£": ("pound", "pounds"),
            "¥": ("yen", "yen"),
            "₹": ("rupee", "rupees"),
        }

        # Abbreviations
        self.abbreviations: List[Tuple[Pattern, str]] = [
            (re.compile(r"\bDr\.", re.I), "Doctor"),
            (re.compile(r"\bMr\.", re.I), "Mister"),
            (re.compile(r"\bMrs\.", re.I), "Missus"),
            (re.compile(r"\bMs\.", re.I), "Mizz"),
            (re.compile(r"\bProf\.", re.I), "Professor"),
            (re.compile(r"\bSt\.", re.I), "Saint"),
            (re.compile(r"\bAve\.", re.I), "Avenue"),
            (re.compile(r"\bRd\.", re.I), "Road"),
            (re.compile(r"\bBlvd\.", re.I), "Boulevard"),
            (re.compile(r"\bDept\.", re.I), "Department"),
            (re.compile(r"\bvs\.", re.I), "versus"),
            (re.compile(r"\betc\.", re.I), "et cetera"),
            (re.compile(r"\be\.g\.", re.I), "for example"),
            (re.compile(r"\bi\.e\.", re.I), "that is"),
            (re.compile(r"\bapprox\.", re.I), "approximately"),
            (re.compile(r"\bmin\.", re.I), "minute"),
            (re.compile(r"\bsec\.", re.I), "second"),
            (re.compile(r"\bhr\.", re.I), "hour"),
            (re.compile(r"\bTTS\b"), "T T S"),
            (re.compile(r"\bAI\b"), "A I"),
            (re.compile(r"\bAPI\b"), "A P I"),
            (re.compile(r"\bGPU\b"), "G P U"),
            (re.compile(r"\bCPU\b"), "C P U"),
            (re.compile(r"\bASR\b"), "A S R"),
            (re.compile(r"\bVAD\b"), "V A D"),
        ]

        # Units of measurement
        self.units: List[Tuple[Pattern, str]] = [
            (re.compile(r"(\d+)\s*kg\b", re.I), r"\1 kilograms"),
            (re.compile(r"(\d+)\s*g\b", re.I), r"\1 grams"),
            (re.compile(r"(\d+)\s*km\b", re.I), r"\1 kilometers"),
            (re.compile(r"(\d+)\s*m\b", re.I), r"\1 meters"),
            (re.compile(r"(\d+)\s*cm\b", re.I), r"\1 centimeters"),
            (re.compile(r"(\d+)\s*mm\b", re.I), r"\1 millimeters"),
            (re.compile(r"(\d+)\s*kHz\b", re.I), r"\1 kilohertz"),
            (re.compile(r"(\d+)\s*Hz\b", re.I), r"\1 hertz"),
            (re.compile(r"(\d+)\s*ms\b", re.I), r"\1 milliseconds"),
            (re.compile(r"(\d+)\s*s\b", re.I), r"\1 seconds"),
            (re.compile(r"(\d+)\s*%\b"), r"\1 percent"),
        ]

        self.units_words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        ]
        self.tens_words = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def _number_to_words(self, n: int) -> str:
        """Convert integer to English words."""
        if n < 0:
            return "minus " + self._number_to_words(abs(n))
        if n < 20:
            return self.units_words[n]
        if n < 100:
            rem = n % 10
            return self.tens_words[n // 10] + ("-" + self.units_words[rem] if rem > 0 else "")
        if n < 1000:
            rem = n % 100
            return self.units_words[n // 100] + " hundred" + (" and " + self._number_to_words(rem) if rem > 0 else "")
        if n < 1_000_000:
            thousands = n // 1000
            rem = n % 1000
            return self._number_to_words(thousands) + " thousand" + (" " + self._number_to_words(rem) if rem > 0 else "")
        if n < 1_000_000_000:
            millions = n // 1_000_000
            rem = n % 1_000_000
            return self._number_to_words(millions) + " million" + (" " + self._number_to_words(rem) if rem > 0 else "")
        billions = n // 1_000_000_000
        rem = n % 1_000_000_000
        return self._number_to_words(billions) + " billion" + (" " + self._number_to_words(rem) if rem > 0 else "")

    def _expand_year_or_number(self, match: re.Match) -> str:
        num_str = match.group(0)
        num = int(num_str)
        # Year format heuristic: 1900-2099
        if 1900 <= num <= 2099 and len(num_str) == 4:
            first_half = int(num_str[:2])
            second_half = int(num_str[2:])
            if second_half == 0:
                return f"{self._number_to_words(first_half)} hundred"
            elif second_half < 10:
                return f"{self._number_to_words(first_half)} o {self._number_to_words(second_half)}"
            else:
                return f"{self._number_to_words(first_half)} {self._number_to_words(second_half)}"
        return self._number_to_words(num)

    def _expand_currencies(self, text: str) -> str:
        for symbol, (singular, plural) in self.currency_map.items():
            pattern = re.escape(symbol) + r"(\d+(?:\.\d{1,2})?)"
            def repl(m):
                val = float(m.group(1))
                whole = int(val)
                cents = int(round((val - whole) * 100))
                whole_words = self._number_to_words(whole) + f" {plural if whole != 1 else singular}"
                if cents > 0:
                    cents_words = f" and {self._number_to_words(cents)} cents"
                    return whole_words + cents_words
                return whole_words
            text = re.sub(pattern, repl, text)
        return text

    def _expand_times(self, text: str) -> str:
        pattern = r"\b(\d{1,2}):(\d{2})(?:\s*(AM|PM|am|pm))?\b"
        def repl(m):
            hour = int(m.group(1))
            minute = int(m.group(2))
            meridiem = m.group(3) or ""
            hour_str = self._number_to_words(hour)
            if minute == 0:
                min_str = "o'clock" if not meridiem else ""
            elif minute < 10:
                min_str = f"o {self._number_to_words(minute)}"
            else:
                min_str = self._number_to_words(minute)
            res = f"{hour_str} {min_str}".strip()
            if meridiem:
                res += " " + " ".join(list(meridiem.upper()))
            return res
        return re.sub(pattern, repl, text)

    def _expand_decimals(self, text: str) -> str:
        pattern = r"\b(\d+)\.(\d+)\b"
        def repl(m):
            whole = self._number_to_words(int(m.group(1)))
            decimals = " ".join([self.units_words[int(d)] for d in m.group(2)])
            return f"{whole} point {decimals}"
        return re.sub(pattern, repl, text)

    def normalize(self, text: str) -> str:
        """Run complete text normalization pipeline."""
        if not text or not isinstance(text, str):
            return ""

        # Basic cleanup
        text = text.strip()
        text = re.sub(r"[\t\r\n]+", " ", text)

        # Abbreviations
        for pattern, replacement in self.abbreviations:
            text = pattern.sub(replacement, text)

        # Currencies & Units
        text = self._expand_currencies(text)
        for pattern, repl in self.units:
            text = pattern.sub(repl, text)

        # Time
        text = self._expand_times(text)

        # Decimals
        text = self._expand_decimals(text)

        # Integers & Years
        text = re.sub(r"\b\d+\b", self._expand_year_or_number, text)

        # Clean multiple spaces and normalize punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.,!?;:])\1+", r"\1", text)
        return text.strip()
