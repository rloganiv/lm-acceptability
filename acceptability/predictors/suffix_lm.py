from typing import Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors import Predictor


@Predictor.register('suffix_lm')
class PrefixLmPredictor(Predictor):
    def predict(
        self,
        prefix: str,
        suffix_a: str,
        suffix_b: str,
    ) -> JsonDict:
        return self.predict_json({
            'prefix': prefix,
            'suffix_a': suffix_a,
            'suffix_b': suffix_b,
        })

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) ->  Instance:
        prefix = json_dict['prefix']
        suffix_a = json_dict['suffix_a']
        suffix_b = json_dict['suffix_b']
        return self._dataset_reader.text_to_instance(
            prefix=prefix,
            suffix_a=suffix_a,
            suffix_b=suffix_b,
        )
