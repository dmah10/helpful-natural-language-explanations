# helpful-explanations

Training runs can be started with the following command, filling in the values for the arguments. 

`main.py` automatically does five separate runs over 5-CV splits. 

```bash
python main.py
    --model-str <model> # model name on hf hub, default 'prajjwal1/bert-mini'
    --dataset-str <dataset> # dataset name [esnli (default), HFC]
    --epochs <epochs> # default 1
    --max-length <max_length> # max input length in tokens, default 512
    --batch-size <batch_size> # default 8
    --learning-rate <learning_rate> # default 0.05
    --weight-decay <weight_decay> # default 0.01
    --explanations <explanation_types> # one, two, or none (default) of [human, llm]
    --explanation-model <explanation_model> # gemma2, gpt4o (default), llama3, mixtral
    --explanation-type <type> # ZS (default), FS
    --explanation-targets <explanation_target> # text_1 (first part of e.g. premise and hypothesis), text_2 (second part)
    --project <project name> # optional; project name on wandb to log results
    --run-once # only do one split rather than 10
    --save-model # save the model after training
```