"""
Grapheme-to-Phoneme (G2P) Converter.
Converts normalized text into international phonetic alphabet (IPA) / ARPAbet sequences
for accurate pronunciation across standard and irregular lexicons.
"""

import re
from typing import List, Optional


class G2PConverter:
    """Grapheme-to-Phoneme module with built-in comprehensive CMU/IPA lexicon and rule-based phonemizer fallback."""

    def __init__(self, backend: str = "hybrid"):
        self.backend = backend
        self._init_lexicon()

    def _init_lexicon(self):
        # Comprehensive core English phonetic lexicon
        self.lexicon = {
            # Greetings & Common Phrases
            "hello": ["h", "ə", "ˈl", "oʊ"],
            "hi": ["h", "ˈaɪ"],
            "hey": ["h", "ˈeɪ"],
            "welcome": ["w", "ˈɛ", "l", "k", "ə", "m"],
            "good": ["ɡ", "ˈʊ", "d"],
            "morning": ["m", "ˈɔː", "n", "ɪ", "ŋ"],
            "evening": ["ˈiː", "v", "n", "ɪ", "ŋ"],
            "afternoon": ["ɑː", "f", "t", "ə", "n", "ˈuː", "n"],
            "today": ["t", "ə", "d", "ˈeɪ"],
            "tomorrow": ["t", "ə", "m", "ˈɒ", "r", "oʊ"],
            "great": ["ɡ", "r", "ˈeɪ", "t"],
            "thank": ["θ", "ˈæ", "ŋ", "k"],
            "thanks": ["θ", "ˈæ", "ŋ", "k", "s"],
            "you": ["j", "ˈuː"],
            "your": ["j", "ˈɔː", "r"],
            "our": ["ˈaʊ", "ə", "r"],
            "my": ["m", "ˈaɪ"],
            "me": ["m", "ˈiː"],
            "we": ["w", "ˈiː"],
            "us": ["ˈʌ", "s"],
            "he": ["h", "ˈiː"],
            "she": ["ʃ", "ˈiː"],
            "it": ["ˈɪ", "t"],
            "its": ["ˈɪ", "t", "s"],
            "they": ["ð", "ˈeɪ"],
            "them": ["ð", "ˈɛ", "m"],
            "their": ["ð", "ˈɛə", "r"],
            "who": ["h", "ˈuː"],
            "what": ["w", "ˈɒ", "t"],
            "when": ["w", "ˈɛ", "n"],
            "where": ["w", "ˈɛə", "r"],
            "why": ["w", "ˈaɪ"],
            "how": ["h", "ˈaʊ"],
            "this": ["ð", "ˈɪ", "s"],
            "that": ["ð", "ˈæ", "t"],
            "these": ["ð", "ˈiː", "z"],
            "those": ["ð", "ˈoʊ", "z"],
            "the": ["ð", "ə"],
            "a": ["ə"],
            "an": ["ə", "n"],
            "and": ["æ", "n", "d"],
            "or": ["ˈɔː", "r"],
            "but": ["b", "ˈʌ", "t"],
            "if": ["ˈɪ", "f"],
            "then": ["ð", "ˈɛ", "n"],
            "so": ["s", "ˈoʊ"],
            "to": ["t", "ˈuː"],
            "of": ["ə", "v"],
            "in": ["ˈɪ", "n"],
            "on": ["ˈɒ", "n"],
            "at": ["ˈæ", "t"],
            "by": ["b", "ˈaɪ"],
            "for": ["f", "ˈɔː", "r"],
            "from": ["f", "r", "ˈɒ", "m"],
            "with": ["w", "ˈɪ", "ð"],
            "without": ["w", "ɪ", "ð", "ˈaʊ", "t"],
            "is": ["ˈɪ", "z"],
            "are": ["ˈɑː", "r"],
            "was": ["w", "ˈɒ", "z"],
            "were": ["w", "ˈɜː", "r"],
            "be": ["b", "ˈiː"],
            "been": ["b", "ˈiː", "n"],
            "being": ["b", "ˈiː", "ɪ", "ŋ"],
            "have": ["h", "ˈæ", "v"],
            "has": ["h", "ˈæ", "z"],
            "had": ["h", "ˈæ", "d"],
            "do": ["d", "ˈuː"],
            "does": ["d", "ˈʌ", "z"],
            "did": ["d", "ˈɪ", "d"],
            "can": ["k", "ˈæ", "n"],
            "could": ["k", "ˈʊ", "d"],
            "will": ["w", "ˈɪ", "l"],
            "would": ["w", "ˈʊ", "d"],
            "shall": ["ʃ", "ˈæ", "l"],
            "should": ["ʃ", "ˈʊ", "d"],
            "may": ["m", "ˈeɪ"],
            "might": ["m", "ˈaɪ", "t"],
            "must": ["m", "ˈʌ", "s", "t"],
            
            # Domain-Specific & Tech Terms
            "voice": ["v", "ˈɔɪ", "s"],
            "cloning": ["k", "l", "ˈoʊ", "n", "ɪ", "ŋ"],
            "clone": ["k", "l", "ˈoʊ", "n"],
            "system": ["s", "ˈɪ", "s", "t", "ə", "m"],
            "studio": ["s", "t", "ˈuː", "d", "i", "oʊ"],
            "eleven": ["ɪ", "ˈl", "ɛ", "v", "ə", "n"],
            "labs": ["l", "ˈæ", "b", "z"],
            "zero": ["z", "ˈɪə", "r", "oʊ"],
            "shot": ["ʃ", "ˈɒ", "t"],
            "speech": ["s", "p", "ˈiː", "tʃ"],
            "speak": ["s", "p", "ˈiː", "k"],
            "speaking": ["s", "p", "ˈiː", "k", "ɪ", "ŋ"],
            "speaker": ["s", "p", "ˈiː", "k", "ə", "r"],
            "synthesis": ["s", "ˈɪ", "n", "θ", "ə", "s", "ɪ", "s"],
            "synthesizes": ["s", "ˈɪ", "n", "θ", "ə", "s", "aɪ", "z", "ɪ", "z"],
            "synthesizer": ["s", "ˈɪ", "n", "θ", "ə", "s", "aɪ", "z", "ə", "r"],
            "synthesizing": ["s", "ˈɪ", "n", "θ", "ə", "s", "aɪ", "z", "ɪ", "ŋ"],
            "synthetic": ["s", "ɪ", "n", "θ", "ˈɛ", "t", "ɪ", "k"],
            "neural": ["n", "ˈjʊə", "r", "ə", "l"],
            "audio": ["ˈɔː", "d", "i", "oʊ"],
            "codec": ["k", "ˈoʊ", "d", "ɛ", "k"],
            "flow": ["f", "l", "ˈoʊ"],
            "matching": ["m", "ˈæ", "tʃ", "ɪ", "ŋ"],
            "diffusion": ["d", "ɪ", "f", "ˈj", "uː", "ʒ", "ə", "n"],
            "transformer": ["t", "r", "æ", "n", "s", "f", "ˈɔː", "m", "ə", "r"],
            "encoder": ["ɛ", "n", "k", "ˈoʊ", "d", "ə", "r"],
            "decoder": ["d", "i", "k", "ˈoʊ", "d", "ə", "r"],
            "watermark": ["w", "ˈɔː", "t", "ə", "m", "ɑː", "k"],
            "watermarking": ["w", "ˈɔː", "t", "ə", "m", "ɑː", "k", "ɪ", "ŋ"],
            "safety": ["s", "ˈeɪ", "f", "t", "i"],
            "consent": ["k", "ə", "n", "s", "ˈɛ", "n", "t"],
            "real": ["r", "ˈɪə", "l"],
            "time": ["t", "ˈaɪ", "m"],
            "streaming": ["s", "t", "r", "ˈiː", "m", "ɪ", "ŋ"],
            "stream": ["s", "t", "r", "ˈiː", "m"],
            "fast": ["f", "ˈɑː", "s", "t"],
            "quality": ["k", "w", "ˈɒ", "l", "ɪ", "t", "i"],
            "model": ["m", "ˈɒ", "d", "l"],
            "latent": ["l", "ˈeɪ", "t", "ə", "n", "t"],
            "latents": ["l", "ˈeɪ", "t", "ə", "n", "t", "s"],
            "natural": ["n", "ˈæ", "tʃ", "ə", "r", "ə", "l"],
            "human": ["h", "ˈj", "uː", "m", "ə", "n"],
            "clear": ["k", "l", "ˈɪə", "r"],
            "clearly": ["k", "l", "ˈɪə", "l", "i"],
            "articulate": ["ɑː", "t", "ˈɪ", "k", "j", "ʊ", "l", "eɪ", "t"],
            "protection": ["p", "r", "ə", "t", "ˈɛ", "k", "ʃ", "ə", "n"],
            "protect": ["p", "r", "ə", "t", "ˈɛ", "k", "t"],
            "generation": ["dʒ", "ɛ", "n", "ə", "r", "ˈeɪ", "ʃ", "ə", "n"],
            "generate": ["dʒ", "ˈɛ", "n", "ə", "r", "eɪ", "t"],
            "generated": ["dʒ", "ˈɛ", "n", "ə", "r", "eɪ", "t", "ɪ", "d"],
            "intelligence": ["ɪ", "n", "t", "ˈɛ", "l", "ɪ", "dʒ", "ə", "n", "s"],
            "artificial": ["ɑː", "t", "ɪ", "f", "ˈɪ", "ʃ", "ə", "l"],
            "technology": ["t", "ɛ", "k", "n", "ˈɒ", "l", "ə", "dʒ", "i"],
            "production": ["p", "r", "ə", "d", "ˈʌ", "k", "ʃ", "ə", "n"],
            "profile": ["p", "r", "ˈoʊ", "f", "aɪ", "l"],
            "timbre": ["t", "ˈæ", "m", "b", "ə", "r"],
            "meaning": ["m", "ˈiː", "n", "ɪ", "ŋ"],
            "mean": ["m", "ˈiː", "n"],
            "words": ["w", "ˈɜː", "d", "z"],
            "word": ["w", "ˈɜː", "d"],
            "better": ["b", "ˈɛ", "t", "ə", "r"],
            "best": ["b", "ˈɛ", "s", "t"],
            "previous": ["p", "r", "ˈiː", "v", "i", "ə", "s"],
            "improved": ["ɪ", "m", "p", "r", "ˈuː", "v", "d"],
            "improve": ["ɪ", "m", "p", "r", "ˈuː", "v"],
            "someone": ["s", "ˈʌ", "m", "w", "ʌ", "n"],
            "somebody": ["s", "ˈʌ", "m", "b", "ə", "d", "i"],
            "something": ["s", "ˈʌ", "m", "θ", "ɪ", "ŋ"],
            "testing": ["t", "ˈɛ", "s", "t", "ɪ", "ŋ"],
            "test": ["t", "ˈɛ", "s", "t"],
            "here": ["h", "ˈɪə", "r"],
            "there": ["ð", "ˈɛə", "r"],
            "now": ["n", "ˈaʊ"],
            "new": ["n", "ˈj", "uː"],
            "like": ["l", "ˈaɪ", "k"],
            "sound": ["s", "ˈaʊ", "n", "d"],
            "sounds": ["s", "ˈaʊ", "n", "d", "z"],
            "world": ["w", "ˈɜː", "l", "d"],
            "work": ["w", "ˈɜː", "k"],
            "working": ["w", "ˈɜː", "k", "ɪ", "ŋ"],
            "first": ["f", "ˈɜː", "s", "t"],
            "second": ["s", "ˈɛ", "k", "ə", "n", "d"],
            "stage": ["s", "t", "ˈeɪ", "dʒ"],
            "two": ["t", "ˈuː"],
            "one": ["w", "ˈʌ", "n"],
            "three": ["θ", "r", "ˈiː"],
            "four": ["f", "ˈɔː", "r"],
            "five": ["f", "ˈaɪ", "v"],
            "six": ["s", "ˈɪ", "k", "s"],
            "seven": ["s", "ˈɛ", "v", "ə", "n"],
            "eight": ["ˈeɪ", "t"],
            "nine": ["n", "ˈaɪ", "n"],
            "ten": ["t", "ˈɛ", "n"]
        }

        # Letter to rule-based phoneme approximations
        self.letter_map = {
            "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
            "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ",
            "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɒ",
            "p": "p", "q": "kw", "r": "r", "s": "s", "t": "t",
            "u": "ʌ", "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z"
        }

    def _rule_based_word_to_phonemes(self, word: str) -> List[str]:
        """Heuristic rule-based phonetic decomposition with suffix handling."""
        word = word.lower()
        if word in self.lexicon:
            return list(self.lexicon[word])

        # Suffix matching
        if word.endswith("tion"):
            stem = word[:-4]
            return self._rule_based_word_to_phonemes(stem) + ["ʃ", "ə", "n"]
        if word.endswith("sion"):
            stem = word[:-4]
            return self._rule_based_word_to_phonemes(stem) + ["ʒ", "ə", "n"]
        if word.endswith("ing") and len(word) > 4:
            stem = word[:-3]
            return self._rule_based_word_to_phonemes(stem) + ["ɪ", "ŋ"]
        if word.endswith("ed") and len(word) > 3:
            stem = word[:-2]
            return self._rule_based_word_to_phonemes(stem) + ["d"]
        if word.endswith("ly") and len(word) > 3:
            stem = word[:-2]
            return self._rule_based_word_to_phonemes(stem) + ["l", "i"]
        if word.endswith("ment") and len(word) > 5:
            stem = word[:-4]
            return self._rule_based_word_to_phonemes(stem) + ["m", "ə", "n", "t"]
        if word.endswith("ness") and len(word) > 5:
            stem = word[:-4]
            return self._rule_based_word_to_phonemes(stem) + ["n", "ə", "s"]
        if word.endswith("able") and len(word) > 5:
            stem = word[:-4]
            return self._rule_based_word_to_phonemes(stem) + ["ə", "b", "l"]
        if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
            stem = word[:-1]
            return self._rule_based_word_to_phonemes(stem) + ["z"]

        phonemes = []
        i = 0
        wlen = len(word)
        while i < wlen:
            # 3-character phonetic trigraphs
            if i + 3 <= wlen:
                trigraph = word[i:i+3]
                if trigraph == "igh":
                    phonemes.append("aɪ")
                    i += 3
                    continue
                elif trigraph == "air" or trigraph == "ear":
                    phonemes.append("ɛə")
                    i += 3
                    continue
                elif trigraph == "tch":
                    phonemes.append("tʃ")
                    i += 3
                    continue

            # 2-character phonetic digraphs
            if i + 2 <= wlen:
                digraph = word[i:i+2]
                if digraph == "th":
                    phonemes.append("ð" if i == 0 else "θ")
                    i += 2
                    continue
                elif digraph == "sh":
                    phonemes.append("ʃ")
                    i += 2
                    continue
                elif digraph == "ch":
                    phonemes.append("tʃ")
                    i += 2
                    continue
                elif digraph == "ph":
                    phonemes.append("f")
                    i += 2
                    continue
                elif digraph == "ng":
                    phonemes.append("ŋ")
                    i += 2
                    continue
                elif digraph == "ee" or digraph == "ea":
                    phonemes.append("iː")
                    i += 2
                    continue
                elif digraph == "oo":
                    phonemes.append("uː")
                    i += 2
                    continue
                elif digraph == "ai" or digraph == "ay":
                    phonemes.append("eɪ")
                    i += 2
                    continue
                elif digraph == "oi" or digraph == "oy":
                    phonemes.append("ɔɪ")
                    i += 2
                    continue
                elif digraph == "ou" or digraph == "ow":
                    phonemes.append("aʊ")
                    i += 2
                    continue
                elif digraph == "oa":
                    phonemes.append("oʊ")
                    i += 2
                    continue
                elif digraph == "qu":
                    phonemes.extend(["k", "w"])
                    i += 2
                    continue
                elif digraph == "ck":
                    phonemes.append("k")
                    i += 2
                    continue
                elif digraph == "wh":
                    phonemes.append("w")
                    i += 2
                    continue
                elif digraph == "wr":
                    phonemes.append("r")
                    i += 2
                    continue
                elif digraph == "kn":
                    phonemes.append("n")
                    i += 2
                    continue

            ch = word[i]
            if ch in self.letter_map:
                phonemes.append(self.letter_map[ch])
            elif ch.isalpha():
                phonemes.append(ch)
            i += 1

        return phonemes if phonemes else list(word)

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert a sentence into a list of phoneme tokens including prosodic punctuation."""
        tokens = []
        words_and_punc = re.findall(r"[\w']+|[.,!?;:]", text)

        for item in words_and_punc:
            if item in ".,!?;:":
                tokens.append(item)
            else:
                ph_list = self._rule_based_word_to_phonemes(item)
                tokens.extend(ph_list)
                tokens.append(" ") # Word boundary token

        if tokens and tokens[-1] == " ":
            tokens.pop()
        return tokens
