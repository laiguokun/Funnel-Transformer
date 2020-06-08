"""Finetune for classification."""

import os
import math
import torch
import numpy as np

import data
from distributed_utils import is_master, distributed_init
import options
import utils
import modeling

def adjust_lr(args, curr_step, optimizer):
  """Adjust learning rate."""
  for param_group in optimizer.param_groups:
    if curr_step < args.warmup_steps:
      param_group["lr"] = args.lr * curr_step / args.warmup_steps
    else:
      decay_steps = args.train_steps - args.warmup_steps
      progress = (curr_step - args.warmup_steps) / decay_steps
      param_group["lr"] = (1 - progress) * args.lr + progress * args.min_lr

def confusion_matrix(pred, label):
  tp = torch.sum((pred == 1) * (label == 1))
  fp = torch.sum((pred == 0) * (label == 1))
  tn = torch.sum((pred == 0) * (label == 0))
  fn = torch.sum((pred == 1) * (label == 0))
  return tp, fp, tn, fn

def _compute_metric_based_on_keys(key, tp, fp, tn, fn):
  """Compute metrics."""

  if key == "corr":
    # because this is only used for threshold search,
    # we only care about matthews_corrcoef but  not pearsons
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
      corr = 0.0
    else:
      corr = (tp * tn - fp * fn) / math.sqrt(
          (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return corr
  else:
    if tp + fp + tn + fn == 0:
      accu = 0
    else:
      accu = (tp + tn) / (tp + fp + tn + fn)

    if key == "accu":
      return accu

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    if key == "accu_and_f1":
      return (accu + f1) / 2
    elif key == "f1":
      return f1

    raise NotImplementedError

def evaluate(args, model, va_loader):
  """Evaluate on validation data."""
  # Keep non-master processes waiting here
  if not is_master(args):
    accuracy = torch.zeros([1]).cuda()
    torch.distributed.barrier()

  # Only master perform evaluation
  if is_master(args):
    num_correct, num_example = 0, 0
    num_tp, num_fp, num_tn, num_fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
      for sent, seg_id, label in va_loader.get_iter(shuffle=False):
        _, ret_dict = model(sent, seg_id=seg_id, cls_target=label)
        cls_corr = ret_dict["cls_corr"]
        num_correct += cls_corr
        num_example += len(sent)
        tp, fp, tn, fn = confusion_matrix(ret_dict["cls_pred"], label)
        num_tp = num_tp + tp
        num_fp = num_fp + fp
        num_tn = num_tn + tn
        num_fn = num_fn + fn

    model.train()

    if args.dataset in ["CoLA"]:
      accuracy = _compute_metric_based_on_keys(
          "corr", num_tp.item(), num_fp.item(), num_tn.item(),
          num_fn.item())
      accuracy = torch.FloatTensor([accuracy]).cuda()
    else:
      accuracy = num_correct / num_example

    if args.distributed:
      torch.distributed.barrier()

  # sync accuracy
  if args.distributed:
    torch.distributed.all_reduce(
        accuracy, op=torch.distributed.ReduceOp.SUM)

  return accuracy.item()


def predict(args, model, loader, out_path, rev_label_dict):
  """Make prediction and write to file. This should only be called by master."""
  # Only master perform prediction
  if is_master(args):
    model.eval()
    with open(out_path, "w") as fo:
      with torch.no_grad():
        for sent, seg_id, label in loader.get_iter(shuffle=False):
          _, ret_dict = model(sent, seg_id=seg_id, cls_target=label)
          cls_pred = ret_dict["cls_pred"]
          for i in range(cls_pred.size(0)):
            label = rev_label_dict[cls_pred[i].item()]
            fo.write("{}\n".format(label))

    model.train()


def main(args):
  """Main training function."""
  torch.cuda.set_device(args.device_id)
  if args.distributed:
    args.distributed_rank = args.device_id
    distributed_init(args)
  if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
  options.setup_device(args)

  ############################################################################
  # Experiment & Logging
  ############################################################################
  if is_master(args):
    if args.resume:
      # rank-0 device creates experiment dir and log to the file
      logging = utils.get_logger(os.path.join(args.model_dir, "log.txt"),
                                 log_=not args.debug)
    else:
      # rank-0 device creates experiment dir and log to the file
      logging = utils.create_exp_dir(args.model_dir, debug=args.debug)
  else:
    # other devices only log to console (print) but not the file
    logging = utils.get_logger(log_path=None, log_=False)


  ############################################################################
  # Load data
  ############################################################################
  logging("Loading data..")
  loaded_data, label_dict = data.load_data(args)
  args.num_class = len(label_dict)
  logging("Loading finish")
  tr_data, va_data, te_data = loaded_data
  va_loader = data.BucketIterator(va_data, args.valid_bsz, args.pad_id,
                                  args.seg_id_pad, args.device, args.max_length)
  te_loader = data.BucketIterator(te_data, args.test_bsz, args.pad_id,
                                  args.seg_id_pad, args.device, args.max_length)

  options.setup_device(args)

  args.model_path = os.path.join(args.model_dir, "model.pt")
  args.var_path = os.path.join(args.model_dir, "var.pt")
  args.config_path = os.path.join(args.model_dir, "net_config.json")
  train_step = 0
  best_accuracy = -float("inf")

  # create model
  if args.resume:
    logging("Resuming from {}...".format(args.model_dir))
    net_config = modeling.ModelConfig.init_from_json(args.config_path, args)
    model = modeling.FunnelTFM(net_config, args)
    model_param, optimizer = torch.load(args.model_path, map_location="cpu")
    logging(model.load_state_dict(model_param, strict=False))
    model = model.to(args.device)
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(args.device)
    best_accuracy, train_step = torch.load(args.var_path)
  else:
    # create new model
    if args.init_ckpt:
      logging("Init from ckpt {}".format(args.init_ckpt))
      net_config = modeling.ModelConfig.init_from_json(args.init_ckpt_config, args)
      model = modeling.FunnelTFM(net_config, args)
      print(model.load_state_dict(torch.load(args.init_ckpt), strict=False))
    else:
      logging("init model")
      net_config = modeling.ModelConfig.init_from_args(args)
      model = modeling.FunnelTFM(net_config, args)
    net_config.to_json(args.config_path)
    model = model.to(args.device)

  # create new optimizer
  if args.fp16:
    from apex.optimizers import FusedAdam
    import apex.amp as amp
    optimizer = FusedAdam(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    amp_model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt)
  else:
    try:
      from apex.optimizers import FusedAdam
      optimizer = FusedAdam(model.parameters(),
                            lr=args.lr,
                            betas=(0.9, 0.99),
                            eps=1e-6,
                            weight_decay=args.weight_decay)
    except ImportError as e:
      logging("use pytorch optimizer")
      optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    betas=(0.9, 0.99),
                                    eps=1e-6,
                                    weight_decay=args.weight_decay)
    amp_model = model

  if args.distributed:
    if args.ddp_backend == "apex":
      from apex.parallel import DistributedDataParallel as DDP
      para_model = DDP(amp_model)
    else:
      from torch.nn.parallel import DistributedDataParallel as DDP
      para_model = DDP(amp_model, device_ids=[args.device_id], find_unused_parameters=True)
  else:
    para_model = amp_model

  ############################################################################
  # Log args
  ############################################################################
  logging("=" * 100)
  for k, v in args.__dict__.items():
    logging("  - {} : {}".format(k, v))
  logging("=" * 100)

  ############################################################################
  # Training
  ############################################################################
  if not args.test_only:
    tr_loader = data.BucketIterator(tr_data, args.train_bsz, args.pad_id,
                                    args.seg_id_pad, args.device,
                                    args.max_length)

    if args.distributed:
      num_data = len(tr_data) // args.distributed_world_size
    else:
      num_data = len(tr_data)
    num_tr_batch = (num_data + args.train_bsz - 1) // args.train_bsz
    args.train_steps = num_tr_batch * args.epochs
    args.warmup_steps = int(args.train_steps * args.warmup_prop)

    num_example = torch.Tensor([0]).to(args.device)
    num_correct = torch.Tensor([0]).to(args.device)

    if args.dataset in ["CoLA"]:
      num_tp = torch.Tensor([0]).to(args.device)
      num_fp = torch.Tensor([0]).to(args.device)
      num_tn = torch.Tensor([0]).to(args.device)
      num_fn = torch.Tensor([0]).to(args.device)

    for epoch in range(args.epochs):
      #### One epoch
      for i, (sent, seg_id, label) in enumerate(
          tr_loader.get_iter(epoch, distributed=args.distributed)):
        optimizer.zero_grad()
        _, ret_dict = para_model(sent, seg_id=seg_id, cls_target=label)
        cls_loss = ret_dict["cls_loss"]
        cls_corr = ret_dict["cls_corr"]

        if args.fp16:
          with amp.scale_loss(cls_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          cls_loss.backward()
        num_correct += cls_corr.detach()
        num_example += len(sent)
        if args.dataset in ["CoLA"]:
          tp, fp, tn, fn = confusion_matrix(ret_dict["cls_pred"], label)
          num_tp = num_tp + tp
          num_fp = num_fp + fp
          num_tn = num_tn + tn
          num_fn = num_fn + fn

        if args.clip > 0:
          if args.fp16:
            gnorm = torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), args.clip)
          else:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else:
          gnorm = 0
          for p in model.parameters():
            if p.grad is not None:
              param_gnorm = p.grad.data.norm(2)
              gnorm += param_gnorm.item() ** 2
          gnorm = gnorm ** (1. / 2)
        train_step += 1
        adjust_lr(args, train_step, optimizer)
        optimizer.step()

        ##### training stat
        if (i + 1) % (num_tr_batch // args.n_log_epoch) == 0:
          if args.distributed:
            torch.distributed.all_reduce(
                num_correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_example, op=torch.distributed.ReduceOp.SUM)
            if args.dataset in ["CoLA"]:
              torch.distributed.all_reduce(
                  num_tp, op=torch.distributed.ReduceOp.SUM)
              torch.distributed.all_reduce(
                  num_fp, op=torch.distributed.ReduceOp.SUM)
              torch.distributed.all_reduce(
                  num_tn, op=torch.distributed.ReduceOp.SUM)
              torch.distributed.all_reduce(
                  num_fn, op=torch.distributed.ReduceOp.SUM)

          if is_master(args):
            if args.dataset in ["CoLA"]:
              corref = _compute_metric_based_on_keys(
                  "corr", num_tp.item(), num_fp.item(), num_tn.item(),
                  num_fn.item())
              logging("[{:>02d}/{:>08d}] Train | corref {:.4f} | gnorm {:.2f} "
                      "| lr {:.6f}".format(epoch, train_step, corref, gnorm,
                                           optimizer.param_groups[0]["lr"]))
            else:
              accuracy = num_correct.item() / num_example.item()
              logging("[{:>02d}/{:>08d}] Train | accu {:.4f} | gnorm {:.2f} "
                      "| lr {:.6f}".format(epoch, train_step, accuracy, gnorm,
                                           optimizer.param_groups[0]["lr"]))
          num_example.zero_()
          num_correct.zero_()
          if args.dataset in ["CoLA"]:
            num_tp.zero_()
            num_fp.zero_()
            num_tn.zero_()
            num_fn.zero_()

        ##### validation
        if train_step % (args.train_steps // 10) == 0:
          accuracy = evaluate(args, model, va_loader)
          if is_master(args):
            if accuracy > best_accuracy:
              torch.save([model.state_dict(), optimizer], args.model_path)
              torch.save([best_accuracy, train_step], args.var_path)
            best_accuracy = max(accuracy, best_accuracy)
            logging("[{}] Valid | curr accu {:.4f} | best accu {:.4f}".format(
                train_step // (args.train_steps // 10), accuracy, best_accuracy))

  ##### make prediction
  if is_master(args) and args.write_prediction:
    rev_label_dict = dict((v, k) for k, v in label_dict.items())
    model.load_state_dict(torch.load(args.model_path, map_location="cpu")[0],
                          strict=False)
    model = model.to(args.device)
    predict(args, model, te_loader,
            os.path.join(args.model_dir, "test_results.txt"),
            rev_label_dict)
    predict(args, model, va_loader,
            os.path.join(args.model_dir, "valid_results.txt"),
            rev_label_dict)

if __name__ == "__main__":
  args = options.get_args()
  args.distributed_world_size = 1
  args.distributed = False
  if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.distributed_world_size = int(os.environ['WORLD_SIZE'])
    args.device_id = args.local_rank
  main(args)
