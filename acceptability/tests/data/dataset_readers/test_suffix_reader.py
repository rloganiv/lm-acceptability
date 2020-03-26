from unittest import TestCase

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import Vocabulary

from acceptability.data.dataset_readers import SuffixReader


class TestSuffixReader(TestCase):
    def test_vanilla_text_to_instance(self):
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(
            [
                'This',
                'is',
                'not',
                'a',
                'difficult',
                'test'
            ],
            namespace='tokens'
        )
        reader = SuffixReader()
        instance = reader.text_to_instance(
            prefix='This is',
            suffix_a='a difficult test',
            suffix_b='not a difficult test',
        )
        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())

        tokens_a = instance['tokens_a']
        self.assertListEqual(
            [t.text for t in tokens_a],
            ['This', 'is', 'a', 'difficult', 'test']
        )

        token_ids_a = tensor_dict['tokens_a']['tokens']['tokens']
        self.assertListEqual(token_ids_a.tolist(), [2, 3, 5, 6, 7])

        eval_mask_a = tensor_dict['eval_mask_a']
        self.assertListEqual(eval_mask_a.tolist(), [0, 0, 1, 1, 1])

        tokens_b = instance['tokens_b']
        self.assertListEqual(
            [t.text for t in tokens_b],
            ['This', 'is', 'not', 'a', 'difficult', 'test']
        )

        token_ids_b = tensor_dict['tokens_b']['tokens']['tokens']
        self.assertListEqual(token_ids_b.tolist(), [2, 3, 4, 5, 6, 7])

        eval_mask_b = tensor_dict['eval_mask_b']
        self.assertListEqual(eval_mask_b.tolist(), [0, 0, 1, 1, 1, 1])

        metadata = tensor_dict['metadata']
        self.assertListEqual(metadata['prefix'],  ['This', 'is',])
        self.assertListEqual(metadata['suffix_a'],  ['a', 'difficult', 'test'])
        self.assertListEqual(metadata['suffix_b'],  ['not', 'a', 'difficult', 'test'])

    def test_gpt2_text_to_instance(self):
        tokenizer = PretrainedTransformerTokenizer(
            model_name='gpt2',
            add_special_tokens=False
        )
        token_indexers={
            'tokens': PretrainedTransformerIndexer(model_name='gpt2')
        }
        reader = SuffixReader(
            tokenizer=tokenizer,
            token_indexers=token_indexers
        )
        instance = reader.text_to_instance(
            prefix='This is',
            suffix_a='a difficult test',
            suffix_b='not a difficult test'
        )
        instance.index_fields(Vocabulary())
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())

        tokens_a = instance['tokens_a']
        self.assertListEqual(
            [t.text for t in tokens_a],
            ['ĠThis', 'Ġis', 'Ġa', 'Ġdifficult', 'Ġtest']
        )

        eval_mask_a = tensor_dict['eval_mask_a']
        self.assertListEqual(eval_mask_a.tolist(), [0, 0, 1, 1, 1])

        tokens_b = instance['tokens_b']
        self.assertListEqual(
            [t.text for t in tokens_b],
            ['ĠThis', 'Ġis', 'Ġnot', 'Ġa', 'Ġdifficult', 'Ġtest']
        )

        eval_mask_b = tensor_dict['eval_mask_b']
        self.assertListEqual(eval_mask_b.tolist(), [0, 0, 1, 1, 1, 1])

        metadata = tensor_dict['metadata']
        self.assertListEqual(metadata['prefix'], ['ĠThis', 'Ġis'])
        self.assertListEqual(metadata['suffix_a'], ['Ġa', 'Ġdifficult', 'Ġtest'])
        self.assertListEqual(metadata['suffix_b'], ['Ġnot', 'Ġa', 'Ġdifficult', 'Ġtest'])
