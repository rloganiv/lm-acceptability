from typing import Any, Dict, List

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.nn import util
from overrides import overrides
import torch
from transformers import AutoModelWithLMHead

from acceptability.common.util import adapt_for_transformer


@Model.register("pretrained_transformer_lm")
class PretrainedTransformerLm(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)

    @staticmethod
    def _adapt_for_transformer(tokens):
        """
        Adapts tokens to input expected by transformers model.
        """
        assert len(tokens) == 1, "Only expecting one sequence of tokens..."
        jankerator = iter(tokens.values())
        token_dict = next(jankerator)
        parameters = {
            'input_ids': token_dict['token_ids'],
            'attention_mask': token_dict['mask'].float(),
        }
        if 'type_ids' in token_dict:
            parameters['token_type_ids'] = token_dict['type_ids']
        return parameters

    def _evaluate(self, tokens, eval_mask):
        transformer_input = self._adapt_for_transformer(tokens)
        logits, *_ = self.model(**transformer_input)
        token_ids = util.get_token_ids_from_text_field_tensors(tokens)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_likelihood = log_probs[:,:-1].gather(-1, token_ids[:,1:].unsqueeze(-1)).squeeze(-1)
        suffix_log_likelihood = (eval_mask[:,1:] * token_log_likelihood).sum(-1)
        return token_log_likelihood, suffix_log_likelihood

    def forward(
        self,
        tokens_a: TextFieldTensors,
        tokens_b: TextFieldTensors,
        eval_mask_a: torch.BoolTensor,
        eval_mask_b: torch.BoolTensor,
        metadata: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        token_log_likelihood_a, suffix_log_likelihood_a = self._evaluate(tokens_a, eval_mask_a)
        token_log_likelihood_b, suffix_log_likelihood_b = self._evaluate(tokens_b, eval_mask_b)

        output_dict = {
            'eval_mask_a': eval_mask_a,
            'eval_mask_b': eval_mask_b,
            'token_log_likelihood_a': token_log_likelihood_a,
            'token_log_likelihood_b': token_log_likelihood_b,
            'suffix_log_likelihood_a': suffix_log_likelihood_a,
            'suffix_log_likelihood_b': suffix_log_likelihood_b,
            'metadata': metadata
        }

        return output_dict
