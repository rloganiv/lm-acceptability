model_names = ['gpt2-medium', 'gpt2-large', 'openai-gpt', 'transfo-xl-wt103', 'xlm-clm-enfr-1024', 'xlnet-base-cased', 'xlnet-large-cased']
tasks = ['prefix', 'suffix']
template = """local model_name = '{model_name}';
local task = '{task}';

{{
    "dataset_reader": {{
        "type": task,
        "tokenizer": {{
            "type": "pretrained_transformer",
            "model_name": model_name,
            "add_special_tokens": false
        }},
        "token_indexers": {{
            "tokens": {{
                "type":  "pretrained_transformer",
                "model_name": model_name
            }}
        }}
    }},
    "train_data_path": "acceptability/tests/fixtures/{task}_pairs.jsonl",
    "validation_data_path": "acceptability/tests/fixtures/{task}_pairs.jsonl",
    "model": {{
        "type": "pretrained_transformer_lm",
        "model_name": model_name
    }},
    "data_loader": {{ }},
    "trainer": {{
        "type": "no_op"
    }}
}}"""

for model_name in model_names:
    for task in tasks:
        with open(f'configs/{model_name}-{task}.jsonnet', 'w') as f:
            f.write(template.format(model_name=model_name, task=task))
