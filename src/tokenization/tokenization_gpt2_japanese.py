# coding=utf-8
#
# MIT License
#
# Copyright (c) 2022 Shu Takayama
# Copyright (c) 2019 tanreinama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" Tokenization class for model GPT2Japanese."""


import os
import jaconv
import json
import re
import numpy as np
from shutil import copyfile


from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "ja-bpe.txt",
    "emoji_file": "emoji.json",
}


class GPT2JapaneseTokenizer(PreTrainedTokenizer):
    """
    Constructs a GPT2Japanese tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        emoji_file (`str`):
            Path to the emoji file.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token that was used during pretraining.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        do_clean (`bool`, *optional*, defaults to `False`):
            Whether or not to mask URLs, email addresses, phone numbers, dates, and prices included in the input.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        emoji_file: str,
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        do_clean: bool = False,
        **kwargs
    ) -> None:

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.emoji_file = emoji_file
        self.do_clean = do_clean

        with open(vocab_file, encoding='utf-8') as f:
            self.bpe = f.read().split('\n')
        self.bpe_idx = {k: v for v, k in enumerate(self.bpe)}
        with open(emoji_file, encoding='utf-8') as f:
            self.emoji = json.loads(f.read())
        self.maxlen = np.max([len(w) for w in self.bpe])
        self.text_pattern1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.text_pattern2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.text_pattern3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.text_pattern4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
        self.text_pattern5 = re.compile(r"(明治|大正|昭和|平成|令和)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
        self.text_pattern6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')

    @property
    def vocab_size(self):
        return len(self.bpe)

    def get_vocab(self):
        return self.bpe_idx

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X`
        - pair of sequences: `A B`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def clean(self, text: str) -> str:
        text = jaconv.z2h(text, kana=False, ascii=True, digit=True)
        text = self.text_pattern1.sub("<URL>", text)
        text = self.text_pattern2.sub("<EMAIL>", text)
        text = self.text_pattern3.sub("<TEL>", text)
        text = self.text_pattern4.sub("<DATE>", text)
        text = self.text_pattern5.sub("<DATE>", text)
        text = self.text_pattern6.sub("<PRICE>", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a string."""
        text = text.replace(' ', '<SP>')
        text = text.replace('　', '<SP>')
        text = text.replace('\r\n', '<BR>')
        text = text.replace('\n', '<BR>')
        text = text.replace('\r', '<BR>')
        text = text.replace('\t', '<TAB>')
        text = text.replace('—', 'ー')
        text = text.replace('−', 'ー')
        for k, v in self.emoji['emoji'].items():
            if k in text:
                text = text.replace(k, v)
        if self.do_clean:
            text = self.clean(text)

        pos = 0
        result = []
        while pos < len(text):
            bp = False
            end = min(len(text), pos+self.maxlen+1) if text[pos] == '<' else pos+2
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.bpe_idx:
                    result.append(wd)
                    pos = e
                    bp = True
                    break
            if not bp:
                end = pos+1
                wd = text[pos:end]
                for i in wd.encode('utf-8'):
                    result.append('<|byte%d|>' % i)
                pos = end
        return result

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        return self.bpe_idx[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.bpe[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        words = []
        byte_tokens = []
        for word in tokens:
            if word[:6] == '<|byte' and word[-2:] == '|>':
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
                    byte_tokens = []
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append('\n')
                elif word == '<TAB>':
                    words.append('\t')
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
        text = ''.join(words)
        return text

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None
    ) -> tuple[str, str]:

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        emoji_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.bpe)+"\n")

        with open(emoji_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.emoji))

        return vocab_file, emoji_file

    def _build_conversation_input_ids(self, conversation: "Conversation") -> list[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length:]
        return input_ids
