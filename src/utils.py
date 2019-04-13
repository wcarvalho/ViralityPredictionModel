import pathlib
import os

def get_filenames(files, keep_suffix=False):
  names = [os.path.basename(fname) for fname in files]
  if keep_suffix: return names
  return [os.path.splitext(n)[0] for n in names]

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
