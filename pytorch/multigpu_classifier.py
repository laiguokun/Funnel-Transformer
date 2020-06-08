"""MultiGPU classification."""

import os
import random
import signal
import threading
import torch

from classifier import main as single_process_main
import options


def main(args):
  """MultiGPU main function."""
  # Set distributed training parameters for a single node.
  if args.distributed_world_size is None:
    args.distributed_world_size = torch.cuda.device_count()
  port = random.randint(10000, 20000)
  args.distributed_init_method = "tcp://localhost:{port}".format(port=port)

  mp = torch.multiprocessing.get_context("spawn")

  # Create a thread to listen for errors in the child processes.
  error_queue = mp.SimpleQueue()
  error_handler = ErrorHandler(error_queue)

  # Train with multiprocessing.
  procs = []
  for i in range(args.distributed_world_size):
    args.distributed_rank = i
    args.device_id = i
    procs.append(mp.Process(target=run, args=(args, error_queue),
                            daemon=True))
    procs[i].start()
    error_handler.add_child(procs[i].pid)
  for p in procs:
    p.join()


def run(args, error_queue):
  """Single process run."""
  try:
    single_process_main(args)
  except KeyboardInterrupt:
    pass  # killed by parent, do nothing
  except Exception:
    # propagate exception to parent process, keeping original traceback
    import traceback
    error_queue.put((args.distributed_rank, traceback.format_exc()))


class ErrorHandler(object):
  """A class that listens for exceptions in children processes."""

  def __init__(self, error_queue):
    self.error_queue = error_queue
    self.children_pids = []
    self.error_thread = threading.Thread(target=self.error_listener,
                                         daemon=True)
    self.error_thread.start()
    signal.signal(signal.SIGUSR1, self.signal_handler)

  def add_child(self, pid):
    self.children_pids.append(pid)

  def error_listener(self):
    (rank, original_trace) = self.error_queue.get()
    self.error_queue.put((rank, original_trace))
    os.kill(os.getpid(), signal.SIGUSR1)

  def signal_handler(self, signalnum, stackframe):
    for pid in self.children_pids:
      os.kill(pid, signal.SIGINT)  # kill children processes
    (rank, original_trace) = self.error_queue.get()
    msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
    msg += original_trace
    raise Exception(msg)


if __name__ == "__main__":
  args = options.get_args()
  args.distributed = True
  main(args)
