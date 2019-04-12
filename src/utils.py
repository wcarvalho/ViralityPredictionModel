import pathlib
import os


def shuffle_batch(t):
  idx = torch.randperm(t.shape[0])
  return t[idx]

def tensor_is_set(tensor):
  try:
    tensor[0]
  except TypeError as e:
    return False
  except Exception as ep:
    raise ep
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