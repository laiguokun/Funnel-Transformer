## Introduction

- As of June 5, 2020, the TF code is tested under TensorFlow 2.2.0 under Python3.
- All the TensorFlow code is written with TPU usage in mind. Hence, most implementations could be far from optimal for GPU usage.



## Pretrained TensorFlow Checkpoints

As the released Funnel-Transformers are trained with the ELECTRA objective, which additionally requires a "generator" component. In practice, there are common use cases:

- For finetuning-only purposes, the generator is directly discarded and only the core model ("discriminator") is finetuned.
- On the other hand, if one wants to further unsupervised tune the pretrained model with the ELECTRA objective on private data, the generator is needed.

Based on the two different use cases, we provide two types of TensorFlow checkpoints:

- **Core-only checkpoints**: This type only includes the parameters of the <u>core Funnel-Transformer model ("encoder" + "decoder")</u>. But all the corresponding optimizer states are removed.
- **Full checkpoints**: This type only includes the parameters of <u>both the discriminator and the generator</u>. In addition, all the corresponding optimizer states are retained in cases one hopes to reuse those states.

| Model Size     | Core-only checkpoints                                        | Full checkpoints                                             |
| -------------- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| B10-10-10H1024 | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B10-10-10H1024-ELEC-TF.tar.gz) | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B10-10-10H1024-ELEC-FULL-TF.tar.gz) |
| B8-8-8H1024    | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B8-8-8H1024-ELEC-TF.tar.gz) | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B8-8-8H1024-ELEC-FULL-TF.tar.gz) |
| B6-6-6H768     | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-6-6H768-ELEC-TF.tar.gz) | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-6-6H768-ELEC-FULL-TF.tar.gz) |
| B6-3x2-3x2H768 | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-3x2-3x2H768-ELEC-TF.tar.gz) | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B6-3x2-3x2H768-ELEC-FULL-TF.tar.gz) |
| B4-4-4H768     | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B4-4-4H768-ELEC-TF.tar.gz) | [Link](http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/B4-4-4H768-ELEC-FULL-TF.tar.gz) |



## Known Issues	

As the our code was developed for TPU platform, there are various known issues for running TensorFlow on GPUs. We currently do not have any clear clues to solving these issues.

#### GPU Memory Issue with TensorFlow

- From our preliminary experiment, TensorFlow is not very memory efficient on GPUs, espectially when compared to its memory performance on TPUs. As all the results in our paper were produced on TPUs, it is currently very costly to re-produce experiments involving long sequences or large models in the paper using GPUs with 12-16GB of RAM.
- Our current Tensorflow implementation does not support `float16` on GPUs. Alternatively, one may refer to [our PyTorch example](https://github.com/laiguokun/Funnel-Transformer/blob/master/pytorch/README.md), which enables `float16`.
- Finally, for unknown reasons, the largest model `B10-10-10H1024` cannot fit into a 16GB GPU with TensorFlow. However, with [PyTorch](https://github.com/laiguokun/Funnel-Transformer/blob/master/pytorch/README.md), the same model `B10-10-10H1024` can take 32 length-128 or 8 length-512 examples on a 16GB GPU.

Given the memory issues mentioned above, we benchmarked the <u>maximum classification batch size</u> for `B-8-8-8H1024` on a single **16GB** GPU (V-100) with **TensorFlow 2.2.0** and `float32` precision:

| System        | Seq Length | Max Batch Size |
| ------------- | ---------- | -------------- |
| `B8-8-8H1024` | 64         | 48             |
| ...           | 128        | 24             |
| ...           | 256        | 8              |
| ...           | 512        | 4              |

In most cases, it is possible to reduce the batch size `train_batch_size` or the maximum sequence length `max_seq_length` to fit in given hardware. The decrease in performance depends on the task and the available resources.

#### Multi-GPU Issuse with the AdamW Optimizer

- Our models are trained with a custom `AdamW` optimizer on TPUs. However, this optimizer does not run correctly in the multi-GPU setting (same issue in the original BERT TensorFlow implementation)
- As a result, the current code **does not support multi-GPU** training. For this moment, please refer to our [PyTorch](https://github.com/laiguokun/Funnel-Transformer/blob/master/pytorch/README.md) code for this functionality.



## Fine-tuning with Funnel-Transformer


### (A) Text Classification/Regression

The code used to perform classification/regression finetuning is in `classifier.py`. It was used to produce the GLUE benchmark results and the 7 text classification results in the paper.

From here on, we assume the TensorFlow checkpoint has been downloaded to `${MODEL_DIR}`.

#### (A.1) GLUE benchmark datasets

- Download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://github.com/laiguokun/Funnel-Transformer/blob/master/data-scripts/download_glue.sh)  and unpack it to some directory `${GLUE_DIR}`.

- For most released Funnel-Transformer ckpts (except `B10-10-10H1024`), **a single 16GB GPU** (V100) is enough for `CoLA`, `RTE`, `MRPC`, and `STS-B` (max sequence length 128 and train batch size 16).

  ```shell
  # Only one GPU in the CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=0 python classifier.py \
    --do_train=True \
    --do_eval=False \
    --task_name=cola \
    ##### Path related
    # `data_dir` is the directory for the "raw data".
    --data_dir=${GLUE_DIR}/CoLA \
    # `output_dir` is the directory for "preprocessed tfrecords".
    --output_dir=proc_data/cola \
    # `model_dir` is the working directory for saving checkpoints and tensorflow events. 
    --model_dir=exp/cola \
    ##### Tokenzation related
    --uncased=True \
    --tokenizer_type=word_piece \
    --tokenizer_path=${MODEL_DIR}/vocab.uncased.txt \
    ##### Initial checkpoint related
    --model_config_path=${MODEL_DIR}/net_config.json \
    --init_checkpoint=${MODEL_DIR}/model.ckpt \
    ##### Task-specific hyper-parameters
    # optimal for CoLA
    --max_seq_length=128 \
    --train_batch_size=16 \
    --learning_rate=1e-5 \
    # roughly train for 10 epochs with 10% warmup and save at the end of each epoch
    --train_steps=5350 \
    --warmup_steps=535 \
    --iterations=535 \
    --save_steps=535 \
    --num_hosts=1 \
    --num_core_per_host=1
  ```

- Evaluate the finetuning results with a single GPU by

  ```shell
  CUDA_VISIBLE_DEVICES=0 python classifier.py \
    # only do eval but not train
    --do_train=False \
    --do_eval=True \
    --task_name=cola \
    --data_dir=${GLUE_DIR}/CoLA \
    --output_dir=proc_data/cola \
    # the script will evaluate all ckpts saved in the `model_dir`
    --model_dir=exp/cola \
    --uncased=True \
    --tokenizer_type=word_piece \
    --tokenizer_path=${MODEL_DIR}/vocab.uncased.txt \
    --max_seq_length=128 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=1
  ```

**Notes**:

- Under the single-GPU setting, we can actually combine training and validation by setting `do_train=True` and `do_eval=True` at the same time. 
- However, in the multi-GPU setting, it is easier to separate the training and evaluation into "two phases", as using multi GPUs to perform evaluation is tricky (one has to correctly separate the data across GPUs).



#### (A.2) IMDB Sentiment classification (with TPU V3-8)

To obtain the beset result on the IMDB dataset, one needs to use the max sequence length 512. Therefore, we show how this can be done with a TPU V3-8.

- Download and unpack the IMDB dataset

  ```bash
  wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  tar zxvf aclImdb_v1.tar.gz
  ```
  
- Launch a Google cloud TPU V3-8 instance (see the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for how to set up Cloud TPUs).

- Set up your Google storage bucket path `${GS_ROOT}` and move the pretrained checkpoint into your Google storage (e.g. `${GS_ROOT}/pretrained_ckpt`).

- Perform TPU finetuning with Funnel-Transformer (any model size) by running

  ```bash
  # google storage path
    GS_ROOT=
    GS_INIT_CKPT_DIR=${GS_ROOT}/pretrained_ckpt
    GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
    GS_MODEL_DIR=${GS_ROOT}/exp/race
    
    # TPU name in google cloud
    TPU_NAME=
    
    # TPU v2-8: This is enough for all released Funnel-Transformer models
    NUM_HOSTS=1
    NUM_CORE_PER_HOST=8
    
    python run_classifier.py \
      --use_tpu=True \
      --tpu=${TPU_NAME} \
      --num_hosts=${NUM_HOST} \
      --num_core_per_host=${NUM_CORE_PER_HOST} \
      --do_train=True \
      --do_eval=True \
      --eval_all_ckpt=True \
      --task_name=imdb \
      # `data_dir` is local while `output_dir` and `model_dir` are on Google Storage
      --data_dir=${IMDB_DIR} \
    --output_dir=${GS_PROC_DATA_DIR} \
      --model_dir=${GS_MODEL_DIR} \
      --uncased=True \
      --tokenizer_type=word_piece \
      --tokenizer_path=${MODEL_DIR}/vocab.uncased.txt \
      --model_config_path=${GS_INIT_CKPT_DIR}/net_config.json \
      --init_checkpoint=${GS_INIT_CKPT_DIR}/model.ckpt \
      # IMDB contains long sequences 
      --max_seq_length=512 \
      --train_batch_size=32 \
      --eval_batch_size=8 \
      # For different model size, the optimal LR is different. Try [1e-5, 2e-5, 3e-5].
      --learning_rate=1e-5 \
      # roughly train for 5 epochs with 10% warmup and save at the end of each epoch
      --train_steps=4000 \
      --warmup_steps=400 \
      --save_steps=400 \
      --iterations=400
  ```

**Notes**:

- Notice that the `data_dir` and `tokenizer_path` both use a local path rather than a Google Storage path. The reason is that data preprocessing is actually performed locally. Hence, using local paths leads to a faster preprocessing speed.



### (B) RACE reading comprehension

The code for the reading comprehension task [RACE](https://www.cs.cmu.edu/~glai1/data/race/) is included in `race.py`.

- Notably, the average length of the passages in RACE is over 300 words (not pieces), which is <u>significantly longer</u> than other popular reading comprehension datasets such as SQuAD. Hence, we have to use the sequence length 512.
- Due to the long sequence, we only provide the instruction for running on TPUs

#####(1) Download the RACE dataset from the [official website](https://www.cs.cmu.edu/~glai1/data/race/) and unpack the raw data to `${RACE_DIR}`.

#####(2) Perform training and evaluation.

```bash
#!/bin/bash
# google storage path
GS_ROOT=
GS_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
GS_MODEL_DIR=${GS_ROOT}/exp/race

# TPU name in google cloud
TPU_NAME=

# TPU v2-8: This is enough for base models (B4-4-4H768, B6-3x2-3x2H768, B6-6-6H768)
NUM_HOSTS=1
NUM_CORE_PER_HOST=8

python run_race.py \
  # TPU settings
  --use_tpu=True \
  --tpu=$TPU_NAME \
  --use_bfloat16=True \
  --num_hosts=$NUM_HOSTS \
  --num_core_per_host=$NUM_CORE_PER_HOST \
  # Paths
  --data_dir=${RACE_DIR} \
  --output_dir=${GS_PROC_DATA_DIR} \
  --model_dir=${GS_MODEL_DIR} \
  # Tokenizer & initial checkpoint
  --uncased=True \
  --tokenizer_type=word_piece \
  --tokenizer_path=${MODEL_DIR}/vocab.uncased.txt \
  --model_config_path=${MODEL_DIR}/net_config.json \
  --init_checkpoint=${MODEL_DIR}/model.ckpt \
  # Other hyper-paramters
  --max_seq_length=512 \
  --max_qa_length=128 \
  --do_train=True \
  --do_eval=True \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --train_steps=24000 \
  --save_steps=2400 \
  --iterations=2400 \
  --warmup_steps=2400 \
  --learning_rate=2e-5 \
  $@
```

- For base Funnel-Transformer models (`B4-4-4H768`, `B6-3x2-3x2H768`, `B6-6-6H768`), TPU v2-8 can be used to replicate the results. But for large models, it requires TPU v3-16 for batch size 16 and TPU v3-32 for batch size 32.



### (D) SQuAD2.0

The code for the SQuAD dataset is included in `squad.py`.

##### (1) Data download and preprocessing

We provide a bash script `scripts/preprp_squad.sh` to do this. However, before running the script, you need to specify the following fields:

- `data_dir`: the **local** directory to which the raw data will be downloaded and saved
- `tokenizer_path`: the **local** or **google storage** path where the downloaded tokenizer file is stored
-  `output_dir`: the **google storage path** used to save the preprocessed tfrecords

After the fields above have been set, one can simply run the bash script:

```bash
bash scripts/prepro_squad.sh
```

- SQuAD preprocessing will take quite some time in order to accurately map character positions (raw data) to sentence piece positions (used for training).

- For faster parallel preprocessing, please refer to the flags `--num_proc` and `--proc_id` in `squad.py`.

#####(3) Perform training and evaluation.

In this work, we always use  <u>sequence length 512</u> and <u>batch size 48</u> for training.

- For the large models (`B10-10-10H1024` & `B8-8-8H1024`), it requires TPU v3-8 to reproduce the results. 

- For the base models (`B4-4-4H768`, `B6-3x2-3x2H768`, `B6-6-6H768`), TPU v2-8 is enough.

With the TPU set up, one can perform training and evaluate with the script `scripts/tpu_squad.sh`



## (D) Custom Usage of Funnel-Transformer

To allow more flexible use of Funnel-Transformer, we here provide a high-level description of the model interface. 

```python
import modeling

##### Part 0: Set up tokenizer & special token ids (!!! Always do this for pretrained ckpt !!!)
tokenizer = tokenization.get_tokenizer()
data_utils.setup_special_ids(tokenizer)

##### Part 1: Initialize model
if FLAGS.model_config:
  # option (a): initialize from an existing json file
  net_config = modeling.ModelConfig.init_from_json(FLAGS.model_config)
else:
  # option (b): initialize from FLAGS (see `modeling.py` for FLAGS needed)
  net_config = modeling.ModelConfig.init_from_flags()
# pass the net_config to the FunnelTFM class, you get the model
model = modeling.FunnelTFM(net_config)

##### Part 2: Get a single-vector representation with the "encoder" 
# Three common inputs 
inputs = tokenizer.convert_text_to_ids(...)  # tokenized text
seg_id = ...  # used to indicated different sequences
input_mask = ...  # 1 inidiates pad and 0 indicates real token 

# Call `get_pooled_output`
summary, _ = model.get_pooled_output(
   inputs,
   is_training,
   seg_id=seg_id,
   input_mask=input_mask,
   use_tpu=use_tpu,
   use_bfloat16=use_bfloat16)

##### Part 3: Extract the sequence of hidden states
# For extract hidden sequence, one can decide whether to use the decoder
use_decoder = False/True

# Call `extract_hiddens`
"""
	`last_hidden`: the decoder output if use_decoder == True else encoder output
	`encoder_hiddens`: the hidden states of all encoder layers
	`ret_dict`: a dict containing other structures of the model
"""
last_hidden, encoder_hiddens, ret_dict = model.extract_hiddens(
    inputs,
    is_training,
    seg_id=seg_id,
    input_mask=input_mask,
    use_decoder=use_decoder,
    use_tpu=use_tpu,
    use_bfloat16=use_bfloat16)
```



## Pretraining

There are three source files particularly related to pretraining on TPUs:

- `pretrain.py`: It implements the core logic of MLM and ELECTRA pretraining with `bfloat16` enabled for TPU.
- `create_pretrain_data.py`: It turns raw text data into tfrecords used for pretraining.
- `input_func_builder.py`: It creates the input function which takes the preprocessed tfrecords as input and provides a data pipeline for pretraining.

##### (1) Turning raw text into tfrecords

```shell
for pass_id in `seq 0 4`; do
  python create_pretrain_data.py \
    --seq_len=512 \
    --input_glob=*.txt \
    --save_dir=${SAVE_DIR} \
    --pass_id=${pass_id} \
    --tokenizer_type=word_piece \
    --tokenizer_path=${PATH_TO_WORD_PIECE} \
    --uncased=True
done
```

where `input_glob` defines all input text files, `save_dir` is the output directory for tfrecords, and `tokenizer_type`, `tokenizer_path` and `uncased` jointly define the tokenizer.

The input text files to `data_utils.py` must use the following format:
* Each line is a sentence.
* An empty line means End of Document.

For example, the text input file could be:
```
This is the first sentence.
This is the second sentence and also the end of the paragraph.<eop>
Another paragraph.

Another document starts here.
```

##### (2) Launch pretraining after preprocessing

After preprocessing, we are ready to pretrain an Funnel-Transformer on TPUs.

```shell
python train.py
  # TPU settings
  --use_tpu=True \
  --tpu=$TPU_NAME \
  # we always use bfloat16 for pretraining & finetuning
  --use_bfloat16=True \
  --num_hosts=$NUM_HOSTS \
  --num_core_per_host=$NUM_CORE_PER_HOST \
  # choose your loss type here
  --loss_type=ELECTRA/MLM \
  # path to the preprocess tfrecords
  --record_dir=${SAVE_DIR} \
  --train_batch_size=256 \
  --learning_rate=1e-4 \
  --seq_len=512 \
  --num_predict=85 \
  --train_steps=1000000 \
  --warmup_steps=10000 \
  # model size
  --block_size="6_6_6" \
  --decoder_size="2" \
  --d_model=1024 \
  --d_embed=1024 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --pool_q_only=True
```


