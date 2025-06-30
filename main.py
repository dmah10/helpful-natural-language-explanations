from pathlib import Path
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from parser import parse_args
import wandb

from train import train_model, get_tokenize_function
from data import load_dataset
from eval import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
print("running on", device)

# we repeat fives times, each on splits with disjoint test sets
reps = 1 if args.run_once else 5
for split_idx in range(reps):
    if args.project is not None:
        wandb.login(
            relogin=False,
        )
        wandb.init(
            project=args.project,
            config=vars(args),
            dir=Path(".").resolve(),
            reinit=True,
        )
        config = wandb.config
        config.model = args.model_str
        config.dataset = args.dataset_str

    trainset, testset = load_dataset(
        name=args.dataset_str,
        explanation_model=args.explanation_model,
        explanation_type=args.explanation_type,
        split_idx=split_idx,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_str)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_str, num_labels=3
    ).to(device)

    data_collator = DataCollatorWithPadding(tokenizer)

    tokenize_function = get_tokenize_function(tokenizer, args)

    tokenized_train = trainset.map(tokenize_function, batched=False)
    tokenized_test = testset.map(tokenize_function, batched=False)
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    trainer, model = train_model(model, tokenized_train, tokenizer, data_collator, args)

    model_name = args.model_str.split("/")[-1]
    if args.save_model:
        model_dir = f"{model_name}_{args.dataset_str}_{args.explanations[0]}_{args.explanation_model}_{args.explanation_type}"
        model.eval()
        model.save_pretrained(f"models/{model_dir}", save_config=True)
        tokenizer.save_pretrained(f"tokenizers/{model_dir}")

    metrics = evaluate_model(model, trainer, tokenized_test, args)

    if args.project is not None:
        wandb.log(metrics)
        wandb.finish()

    print(metrics)
