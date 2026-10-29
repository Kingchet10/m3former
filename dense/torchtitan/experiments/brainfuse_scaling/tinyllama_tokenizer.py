"""Original SentencePiece model; training reads existing IDs without re-tokenizing."""
from dataclasses import dataclass
from pathlib import Path

from torchtitan.components.tokenizer import BaseTokenizer


class TinyLlamaTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config=None, *, tokenizer_path):
        import sentencepiece as spm
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.processor = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_path) / "tokenizer.model"))
        self.bos_id, self.eos_id = self.processor.bos_id(), self.processor.eos_id()
        if self.get_vocab_size() != 32000 or self.eos_id != 2:
            raise ValueError("Expected TinyLlama 32K tokenizer with EOS=2")

    def encode(self, text, *, add_bos=False, add_eos=False):
        ids = self.processor.encode(text, out_type=int)
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids):
        return self.processor.decode(ids)

    def get_vocab_size(self):
        return self.processor.get_piece_size()
