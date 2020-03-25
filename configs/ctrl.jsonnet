local model_name='ctrl';

{
    "dataset_reader": {
        "type": "prefix_lm",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "add_special_tokens": false
        },
        "token_indexers": {
            "tokens": {
                "type":  "pretrained_transformer",
                "model_name": model_name
            }
        }
    },
    "train_data_path": "acceptability/tests/fixtures/sentence_pairs.jsonl",
    "validation_data_path": "acceptability/tests/fixtures/sentence_pairs.jsonl",
    "model": {
        "type": "prefix_lm",
        "model_name": model_name
    },
    "data_loader": { },
    "trainer": {
        "type": "no_op"
    }
}