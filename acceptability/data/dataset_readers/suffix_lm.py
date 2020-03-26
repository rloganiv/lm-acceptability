import json
import logging
from typing import Dict, Iterable

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
import numpy as np
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("suffix_lm")
class SuffixLmReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(
        self,
        file_path: str,
    ) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(
                    prefix=data['prefix'],
                    suffix_a=data['suffix_a'],
                    suffix_b=data['suffix_b']
                )

    @overrides
    def text_to_instance(
        self,
        prefix: str,
        suffix_a: str,
        suffix_b: str,
    ) -> Instance:

        # HuggingFace's tokenizers require leading whitespace.
        prefix_tokens = self._tokenizer.tokenize(' ' + prefix)
        suffix_a_tokens = self._tokenizer.tokenize(' ' + suffix_a)
        suffix_b_tokens = self._tokenizer.tokenize(' ' + suffix_b)

        tokens_a = prefix_tokens + suffix_a_tokens
        tokens_b = prefix_tokens + suffix_b_tokens

        eval_mask_a = np.array([0] * len(prefix_tokens) + [1] * len(suffix_a_tokens))
        eval_mask_b = np.array([0] * len(prefix_tokens) + [1] * len(suffix_b_tokens))

        metadata = {
            'prefix': [t.text for t in prefix_tokens],
            'suffix_a': [t.text for t in suffix_a_tokens],
            'suffix_b': [t.text for t in suffix_b_tokens],
        }

        fields = {
            'tokens_a': TextField(tokens_a, token_indexers=self._token_indexers),
            'tokens_b': TextField(tokens_b, token_indexers=self._token_indexers),
            'eval_mask_a': ArrayField(eval_mask_a, dtype=bool),
            'eval_mask_b': ArrayField(eval_mask_b, dtype=bool),
            'metadata': MetadataField(metadata),
        }

        return Instance(fields)
