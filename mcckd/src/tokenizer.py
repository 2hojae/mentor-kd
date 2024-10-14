from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.model = SentencePieceProcessor()
        self.model.Init(model_file=model_path)
        self.n_words: int = self.model.vocab_size()
        self.bos_id: int = self.model.bos_id()
        self.eos_id: int = self.model.eos_id()
        # TODO: pad id to be -1
        self.pad_id: int = 0 if self.model.pad_id() == -1 else self.model.pad_id()
        assert self.n_words == self.model.GetPieceSize()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        return self.model.Encode(s, add_bos=bos, add_eos=eos)

    def decode(self, t: List[int]) -> str:
        return self.model.Decode(t)

    def tokenize(self, s: str, bos: bool = False, eos: bool = False):
        return self.model.Encode(s, out_type=str, add_bos=bos, add_eos=eos)

    def detokenize(self, t: List[int]) -> List[str]:
        return [self.id2piece(i) for i in t]

    def id2piece(self, idx: int) -> str:
        return self.model.IdToPiece(idx)

    def piece2id(self, s: str) -> int:
        return self.model.PieceToId(s)

    def whitespace_id(self):
        return self.piece2id('▁')
