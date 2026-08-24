"""
Text and Phoneme Tokenizer for Zero-Shot Voice Synthesis.
Manages vocabulary mapping, special tokens, and formatted prompt construction.
"""

from typing import Dict, List, Optional, Union
from .g2p import G2PConverter
from .normalize import TextNormalizer


class VoiceTokenizer:
    """Manages vocabulary, tokenization, and sequence packaging for the TTS generative backbone."""

    # Special tokens
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPK_START = "<spk_start>"
    SPK_END = "<spk_end>"
    TEXT_START = "<text_start>"
    TEXT_END = "<text_end>"
    AUDIO_START = "<audio_start>"
    AUDIO_END = "<audio_end>"

    SPECIAL_TOKENS = [
        PAD, UNK, BOS, EOS,
        SPK_START, SPK_END,
        TEXT_START, TEXT_END,
        AUDIO_START, AUDIO_END
    ]

    def __init__(self, vocab_path: Optional[str] = None):
        self.normalizer = TextNormalizer()
        self.g2p = G2PConverter()
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        # 1. Add special tokens
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.token2id[tok] = idx
            self.id2token[idx] = tok

        # 2. Add standard IPA phonemes & punctuations
        phoneme_symbols = [
            " ", ".", ",", "!", "?", ";", ":", "'", "-",
            "æ", "b", "d", "ɛ", "f", "ɡ", "h", "ɪ", "dʒ", "k", "l", "m", "n", "ɒ", "p", "kw", "r", "s", "t", "ʌ", "v", "w", "ks", "j", "z",
            "θ", "ʃ", "tʃ", "ŋ", "iː", "uː", "eɪ", "ɔɪ", "aʊ", "ˈl", "ˈɜː", "ˈɔɪ", "ˈoʊ", "ˈɪ", "ˈɛ", "ˈæ", "ˈɒ", "ˈiː", "ˈjʊə", "ˈɔː",
            "ˈaɪ", "ˈɑː", "ˈʌ", "ˈɪə", "ə", "oʊ", "ʊə", "ɑː", "ɔː", "ɜː", "ð", "ʒ"
        ]
        
        # Add basic ASCII characters
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            if ch not in phoneme_symbols:
                phoneme_symbols.append(ch)

        for sym in phoneme_symbols:
            if sym not in self.token2id:
                idx = len(self.token2id)
                self.token2id[sym] = idx
                self.id2token[idx] = sym

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token2id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token2id[self.EOS]

    def encode_text(self, text: str, normalize: bool = True) -> List[int]:
        """Convert raw text to a sequence of phoneme token IDs."""
        if normalize:
            text = self.normalizer.normalize(text)
        phonemes = self.g2p.text_to_phonemes(text)
        
        unk_id = self.token2id[self.UNK]
        ids = [self.token2id.get(p, unk_id) for p in phonemes]
        return ids

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a phonetic/text string."""
        return "".join([self.id2token.get(i, "") for i in token_ids if i not in (self.pad_id, self.bos_id, self.eos_id)])

    def build_prompt_sequence(self, text_token_ids: List[int], speaker_prompt_len: int = 0) -> List[int]:
        """
        Builds the Stage 1 LM input sequence:
        [BOS] [SPK_START] ... [SPK_END] [TEXT_START] ... [TEXT_END] [AUDIO_START]
        """
        seq = [self.token2id[self.BOS]]
        if speaker_prompt_len > 0:
            seq.append(self.token2id[self.SPK_START])
            # Placeholder IDs for speaker prefix tokens
            seq.extend([self.token2id[self.UNK]] * speaker_prompt_len)
            seq.append(self.token2id[self.SPK_END])

        seq.append(self.token2id[self.TEXT_START])
        seq.extend(text_token_ids)
        seq.append(self.token2id[self.TEXT_END])
        seq.append(self.token2id[self.AUDIO_START])
        return seq
