import pathlib
import os

class AverageMeter(object):
  """
  Computes and stores the average and
  current value.
  """
  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0.0
      self.sum = 0.0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val*n
      self.count += n

  @property
  def average(self):
    if self.count > 0:
      return float(self.sum) / self.count
    else:
      return 0

def tensor_is_set(tensor):
  try:
    tensor[0]
  except TypeError as te:
    # if give None for tensor
    return False
  except IndexError as ie:
    # if give empty list for tensor
    return False
  except Exception as e:
    # something else happened.
    raise e
  return True

def mkdir_p(path):
  """python version of mkdir -p
  stolen from: https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
  """
  pathlib.Path(path).mkdir(parents=True, exist_ok=True)
  print("Made path '%s'" % path)

def filepath_exists(file):
  if not file: return
  file_dir = os.path.dirname(file)
  path_exists(file_dir)

def path_exists(path, verbosity=0):
  if not path: return
  if not os.path.exists(path):
    mkdir_p(path)
