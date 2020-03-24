from typing import Any, Dict, List

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from overrides import overrides
import torch
from transformers import AutoModelWithLMHead

from acceptability.modules.language_model_head import LanguageModelHead


@Model.register("prefix_lm")
class PrefixLm(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)

    @staticmethod
    def _adapt_tokens(tokens):
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

    def forward(
        self,
        tokens: TextFieldTensors,
        eval_mask: torch.BoolTensor,
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        # A little hacky
        input_ = self._adapt_tokens(tokens)
        logits, *_ = self.model(**input_)
        token_ids = util.get_token_ids_from_text_field_tensors(tokens)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_likelihood = log_probs[:,:-1].gather(-1, token_ids[:,1:].unsqueeze(-1)).squeeze()
        suffix_log_likelihood = (eval_mask[:,1:] * token_log_likelihood).sum(-1)

        output_dict = {
            'token_ids': token_ids,
            'eval_mask': eval_mask,
            'token_log_likelihood': token_log_likelihood,
            'suffix_log_likelihood': suffix_log_likelihood,
            'loss': -suffix_log_likelihood.mean(),
            'metadata': metadata
        }

        return output_dict

    @overrides
    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        suffix_log_likelihoods = []
        eval_mask = output_dict['eval_mask']
        token_log_likelihood = output_dict['token_log_likelihood']
        for _mask, _log_likelihood in zip(eval_mask, token_log_likelihood):
            suffix_log_likelihood = _log_likelihood.masked_select(_mask[1:])
            suffix_log_likelihoods.append(suffix_log_likelihood.tolist())

        readable = {
            'suffix': output_dict['metadata']['suffix'],
            'suffix_log_likelihood': output_dict['suffix_log_likelihood'],
            'token_log_likehihood': suffix_log_likelihoods
        }
