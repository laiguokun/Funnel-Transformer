"""Parse args."""

import argparse
import re


def get_args():
  """Parge args."""
  parser = argparse.ArgumentParser(description="")

  ##### Model argsments
  parser.add_argument("--overwrite_keys", default="", type=str,
                      help="Comma separated keys to indicate configs that "
                      "will always be overwritten by the args values.")
  parser.add_argument("--model_size", default="", type=str,
                      help="A compact repsentation of the model size.")
  parser.add_argument("--vocab_size", default=-1, type=int,
                      help="Vocabulary size.")
  parser.add_argument("--d_model", default=768, type=int,
                      help="Dimension of the model.")
  parser.add_argument("--d_embed", default=768, type=int,
                      help="Dimension of the embeddings.")
  parser.add_argument("--n_head", default=12, type=int,
                      help="Number of attention heads.")
  parser.add_argument("--d_head", default=64, type=int,
                      help="Dimension of each attention head.")
  parser.add_argument("--d_inner", default=3072, type=int,
                      help="Dimension of inner hidden size in FFN.")
  parser.add_argument("--dropout", default=0.1, type=float,
                      help="Model dropout.")
  parser.add_argument("--dropatt", default=0.1, type=float,
                      help="Attention dropout.")
  parser.add_argument("--dropact", default=0.0, type=float,
                      help="Activation dropout.")
  parser.add_argument("--attn_type", default="rel_shift", type=str,
                      choices=["rel_shift", "factorized"],
                      help="For GPU/CPU, `rel_shift` is much faster. "
                      "For TPU, `factorized` is faster instead.")
  ##### Data
  parser.add_argument("--data_dir", default="", type=str,
                      help="Path to the data directory.")
  parser.add_argument("--output_dir", default="", type=str,
                      help="Path to the processed data directory.")
  parser.add_argument("--dataset", default=None, type=str,
                      help="Name of the dataset.")
  parser.add_argument("--max_length", default=128, type=int,
                      help="Max sequence length allowed.")
  parser.add_argument("--tokenizer_type", default="word_piece", type=str,
                      choices=["word_piece", "sent_piece"],
                      help="Type of the tokenizer.")
  parser.add_argument("--tokenizer_path", default=None, type=str,
                      help="Path to the tokenizer model or vocab.")
  parser.add_argument("--cased", action="store_true",
                      help="Use cased tokenizer.")
  parser.add_argument("--overwrite_data", action="store_true",
                      help="Overwrite existing cache.")
  # keep the following unchanged
  parser.add_argument("--seg_id_a", default=0, type=int,
                      help="seg id for the first segment.")
  parser.add_argument("--seg_id_b", default=1, type=int,
                      help="seg id for the second segment.")
  parser.add_argument("--seg_id_cls", default=2, type=int,
                      help="seg id for the cls token.")
  parser.add_argument("--seg_id_pad", default=0, type=int,
                      help="seg id for the pad token.")
  ##### Training
  # path
  parser.add_argument("--resume", action="store_true",
                      help="Resume from existing ckpt.")
  parser.add_argument("--init_ckpt", default="", type=str,
                      help="Path to the initial model ckpt.")
  parser.add_argument("--init_ckpt_config", default="", type=str,
                      help="Path to the initial model ckpt json config")
  parser.add_argument("--model_dir", default="", type=str,
                      help="Path to the model directory.")
  parser.add_argument("--log_path", default="", type=str,
                      help="Path to log exp.")
  # optimization
  parser.add_argument("--lr", default=1e-5, type=float,
                      help="Initial learning rate.")
  parser.add_argument("--warmup_prop", default=0.10, type=float,
                      help="Warmup proportion.")
  parser.add_argument("--min_lr", default=0, type=float,
                      help="Minimum learning rate.")
  parser.add_argument("--weight_decay", default=0.01, type=float,
                      help="Weight decay rate.")
  parser.add_argument("--clip", default=1.0, type=float,
                      help="Gradient clip value.")

  parser.add_argument("--epochs", default=10, type=int,
                      help="Number of epochs to train the model.")
  parser.add_argument("--train_bsz", default=16, type=int,
                      help="Training batch size.")
  parser.add_argument("--valid_bsz", default=32, type=int,
                      help="Validation batch size.")
  parser.add_argument("--test_bsz", default=32, type=int,
                      help="Testing batch size.")
  # misc
  parser.add_argument("--seed", default=None, type=int,
                      help="Random seed.")
  parser.add_argument("--n_log_epoch", default=20, type=int,
                      help="Number of times to log per epoch.")
  parser.add_argument("--test_only", action="store_true",
                      help="Only run testing.")
  parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode.")
  parser.add_argument("--write_prediction", action="store_true",
                      help="Write prediction to files.")

  ##### Distributed
  parser.add_argument("--device_id", default=0, type=int,
                      help="cuda device id. set to -1 for cpu.")
  parser.add_argument("--distributed_init_method", default=None, type=str,
                      help="Distributed group initialization method.")
  parser.add_argument("--distributed_port", default=11111, type=int,
                      help="port number in the distributed training")
  parser.add_argument("--distributed_world_size", default=None, type=int,
                      help="world size in the distributed setting")
  parser.add_argument("--distributed_rank", default=0, type=int,
                      help="local rank in the distributed setting")
  parser.add_argument("--ddp_backend", default="apex", type=str,
                      choices=["pytorch", "apex"],
                      help="DDP backend")
  parser.add_argument("--distributed_backend", default="nccl", type=str,
                      help="distributed backend")
  parser.add_argument("--log_all_processes", action="store_true",
                      help="Logging on all processes.")

  ##### Funnel-Transformer specific
  parser.add_argument("--block_size", default="4_4_4", type=str,
                      help="Depth of blocks with potential parameter sharing")
  # down-sampling
  parser.add_argument("--pooling_type", default="mean", type=str,
                      choices=["mean", "max"],
                      help="Type of the pooling operation.")
  parser.add_argument("--pooling_size", default=2, type=int,
                      help="Kernel size for max and mean pooling.")
  parser.add_argument("--pool_q_only", action="store_true",
                      help="Only perform pooling on query")
  parser.add_argument("--separate_cls", action="store_true",
                      help="Whether to isolate the [cls]")
  parser.add_argument("--truncate_seq", action="store_true",
                      help="Truncate seq to speed up separate [cls].")

  ##### fp16
  parser.add_argument("--fp16", action="store_true",
                      help="whether use fp16")
  parser.add_argument("--amp_opt", default="O2", type=str,
                      choices=["O1", "O2"],
                      help="AMP opt level of the fp16 optimization manager")
  args = parser.parse_args()

  # model size
  if args.model_size:
    pattern = re.compile(r"^H(\d+)$")
    m = pattern.match(args.model_size)
    args.d_model = int(m.group(1))
    args.n_head = args.d_model // 64
    args.d_embed = args.d_model
    args.d_head = args.d_model // args.n_head
    args.d_inner = args.d_model * 4


  return args

def setup_device(args):
  # whether to use gpu
  if args.device_id >= 0:
    args.device = "cuda:{}".format(args.device_id)
  else:
    args.device = "cpu"
