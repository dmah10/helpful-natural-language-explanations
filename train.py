from transformers import TrainingArguments, Trainer
from transformers import AdamW, get_scheduler


def sep_append(text, explanation):
    return text + " [SEP] " + explanation


def sep_prepend(text, explanation):
    return explanation + " [SEP] " + text


def get_tokenize_function(
    tokenizer,
    args,
):
    def tokenize_function(example, return_text=False):
        seq_1 = example["text_1"]
        seq_2 = example["text_2"]
        append_func = sep_append if args.method == "append" else sep_prepend

        # append human and llm explanations
        if "human" in args.explanations and example["human_explanation"] is not None:
            if "text_1" in args.explanation_targets:
                seq_1 = append_func(seq_1, example["human_explanation"])
            if "text_2" in args.explanation_targets:
                seq_2 = append_func(seq_2, example["human_explanation"])
        if "llm" in args.explanations and example["explanation"] is not None:
            if "text_1" in args.explanation_targets:
                seq_1 = append_func(seq_1, example["explanation"])
            if "text_2" in args.explanation_targets:
                seq_2 = append_func(seq_2, example["explanation"])

        if return_text:
            return seq_1 + seq_2
        else:
            return tokenizer(
                seq_1,
                seq_2,
                truncation=True,
                max_length=args.max_length,
            )

    return tokenize_function


def train_model(model, tokenized_train, tokenizer, data_collator, args):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=500,
        save_strategy="no",
        report_to=["wandb"] if args.project is not None else [],
    )

    # set up custom optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = (len(tokenized_train) // args.batch_size) * int(args.epochs)
    warmup_steps = int(0.1 * num_training_steps)  # 10% of training steps for warmup
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()
    return trainer, model
