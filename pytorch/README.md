## Introduction

This directory contains our example PyTorch implementation of **Funnel-Transformer**.

- Currently, the PyTorch implementation <strong>only</strong> supports the **text classification** tasks. Hence, with the current code, you are able to replicate all classification tasks reported in the paper, including GLUE Benchmarks (except `STS-B`, which is regression) as well as the 7 addition text classification tasks .
- The PyTorch implementation currently <strong>does not </strong> support token-level NLP task, such as masked language modeling pretraining. If you are interested in it, please check our TensorFlow implementation.
- Though mathematically equivalent, there exist various the software (and hardware) differences between this PyTorch + GPU implementation and the original TensorFlow + TPU implementation:
  - Optimization: TensorFlow + TPU implementation clips the gradient on each TPU core while PyTorch + GPU implementation clips the cross-core-summed gradient.
  - `rel_shift` v.s. `factorized` implementation of the relative positional attention will inevitably have the numerical differences
  - The models were originally train under `bfloat16`, which is quite different from `float16`.



## Prerequisite

- [PyTorch](https://PyTorch.org/get-started/locally/) and [apex](https://github.com/NVIDIA/apex) (which is necessary for FP16 optimization)
- As of June 5, 2020, the code is tested under on PyTorch=1.5.0, apex=0.1 and Python3.



## Pretrained PyTorch Checkpoints

| Model Size     | PyTorch Checkpoint Link                                      |
| -------------- | ------------------------------------------------------------ |
| B10-10-10H1024 | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B10-10-10H1024-ELEC-PT.tar.gz) |
| B8-8-8H1024    | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B8-8-8H1024-ELEC-PT.tar.gz) |
| B6-6-6H768     | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-6-6H768-ELEC-PT.tar.gz) |
| B6-3x2-3x2H768 | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-3x2-3x2H768-ELEC-PT.tar.gz) |
| B4-4-4H768     | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B4-4-4H768-ELEC-PT.tar.gz) |



## Finetuning Funnel-Transformer

#### (A) GLUE classification on a single GPU

(1) Prepare data & pretrained model

- Download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://github.com/laiguokun/Funnel-Transformer/blob/master/data-scripts/download_glue.sh) and unpack it to some directory `${glue_dir}`.
- Download the PyTorch checkpoint and unpack it to `${pretrain_dir}` (see example [download script](https://github.com/laiguokun/Funnel-Transformer/blob/master/download_all_ckpts.sh)).

(2) Check out the bash script `scripts/classifier.sh`:

- Firstly, you need to change the `pretrain_dir` and `task_name` fields to control the (a) pretrained model to use and (b) the dataset to finetune on.
- In addition, you can also change the hyperparameters in this script, such as batch size, learning rate, and so on.

(3) After (1) & (2) are done, you can run the bash script to perform finetuning

```bash
bash scripts/classifier.sh
```

(4) See the [section](https://github.com/zihangdai/pretrain/tree/master/release/PyTorch#batch-size-guideline-v100-16gb) below for the batch size guide for large models.



#### (B) FP32 and FP16 optimization options

The FP16 optimizer can provide faster optimization and require less GPU memory without an obvious performance drop. So in default, we use it in the finetuning script. If you prefer to use FP32 optimizer, you can set `amp_opt=O0`. If you want FP32 optimizer and don't want to use the apex package, yon can simply remove the `--fp16` option.

- **B10-10-10H1024 Numerical Issue**: If you are using `B10-10-10H1024` checkpoint, you need to change the AMP optimizer version to `amp_opt=O1` from `amp_opt=02` option due to the numerical issue. For example,

```bash
task_name=CoLA
lr=1e-5
train_bsz=16
epochs=10
max_length=128

python classifier.py \
  --data_dir=${glue_dir}/glue/${task_name} \
  --output_dir=proc_data/glue/${task_name} \
  --model_dir=exp/${task_name} \
  --tokenizer_path=${pretrain_dir}/vocab.uncased.txt \
  --tokenizer_type=word_piece \
  --init_ckpt_config=${pretrain_dir}/net_config.pytorch.json \
  --init_ckpt=${pretrain_dir}/model.pt \
  --attn_type=rel_shift \
  --dataset=${task_name} \
  --lr=${lr} \
  --train_bsz=${train_bsz} \
  --epochs=${epochs} \
  --max_length=${max_legnth} \
  --fp16 \
  ##### Change to O1 for B10-10-10H1024
  --amp_opt=O2
```

If you are finetuning on your other datasets, we recommend you to check both "O1" and "O2" (which is the default choice) to find out which one works better. Usually "O2" is faster and "O1" is more numerically stable.



#### (C) Multi-GPU finetunning

If you want to use multi-GPUs for finetuning, you can use the `scripts/multigpu_classifier.sh`. While its usage is almost identical to the `scripts/classifier.sh`, there are some particular points worth mentioning:

- Under the multi-GPU setting, the `train_batch_size` refers to the **per-GPU** train batch size.
- Currently, we employ apex DDP as the default DDP backend. If you don't want to use the apex package, you can change the option `--ddp_backend=apex` to `--ddp_backend=PyTorch`.



#### (D) Recommanded Hyperparameters

For GLUE benchmark tasks, we recommend following hyperparameters:

| Task         | bsz  | epochs | lr             |
| ------------ | ---- | ------ | -------------- |
| RTE          | 16   | 10     | 1e-5,2e-5      |
| MRPC         | 16   | 10     | 1e-5,2e-5,3e-5 |
| CoLA         | 16   | 10     | 1e-5           |
| SST-2        | 16   | 10     | 1e-5           |
| QNLI         | 32   | 3      | 2e-5           |
| MNLI_matched | 64   | 3      | 1e-5           |
| QQP          | 64   | 5      | 2e-5           |

- Generally speaking, smaller datasets are more sensitive to hyper-parameters and usually involve larger performance variance. 
- Since we release multiple models, we will leave it to you to explore the optimal learning rate and other hyper-parameters for each specific model.



#### (E) Other options

- If you want to run PyTorch implementation TPU, take a look at `--attn_type=factorized`.



## Batch Size Guideline (V100-16GB)

As one can often run into OOM issue for large models, on a 16GB V100, we benchmark the maximum finetune batch size allowed for `B8-8-8H1024` and `B10-10-10H1024` with different sequence lengths.

| Model          | Sequence length | Batch size |
| -------------- | --------------- | ---------- |
| B8-8-8H1024    | 64              | 128        |
| ...            | 128             | 48         |
| ...            | 256             | 24         |
| ...            | 512             | 8          |
| B10-10-10H1024 | 64              | 64         |
| ...            | 128             | 32         |
| ...            | 256             | 12         |
| ...            | 512             | 4          |



## Custom Usage of Funnel-Transformer

To allow more flexible use of Funnel-Transformer, we here provide a high-level description of the model interface. If you want to utilize the `float16` optimizer, please check `classifier.py` for the implementation.

```python
import options
import modeling

arg = options.get_args()

##### Part 0: Set up tokenizer & special token ids (!!! Always do this !!!)
tokenizer = tokenization.get_tokenizer(args)
data_utils.setup_special_ids(args, tokenizer)

##### Part 1: Initialize model
if FLAGS.model_config:
  # option (a): initialize from an existing json file
  net_config = modeling.ModelConfig.init_from_json(args.config_path, args)
else:
  # option (b): initialize from args (see `modeling.py` for args needed)
  net_config = modeling.ModelConfig.init_from_flags(args)
# pass the net_config to the FunnelTFM class, you get the model
net_config = modeling.ModelConfig.init_from_args(args)
model = modeling.FunnelTFM(net_config, args)

##### Part 2: Get inputs
# Three common inputs
inputs = tokenizer.convert_text_to_ids(...)  # tokenized text
seg_ids = ...  # used to indicated different sequences
input_mask = ...  # 1 inidiates pad and 0 indicates real token

##### Part 3: Extract the sequence of hidden states
# Call `extract_hiddens`
"""
	`hiddens`: hiddens states of all encoder layers (including word embedding as the first element)
	`ret_dict`: a dict containing other structures of the model
"""
hiddens, ret_dict = model.extract_hiddens(
    inputs=inputs,
    seg_id=seg_id,
    input_mask=input_mask)
```



## Tensorflow Checkpoint Conversion

If you want to pretrain your own Funnel-Transformer with our TensorFlow implementation and convert it to this version PyTorch checkpoint, you can use `scripts/convert_ckpt2pt.sh`.

