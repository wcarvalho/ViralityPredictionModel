import os

def get_overlapping_data_files(data_files, image_files, text_files, label_files):
  data_filenames = set(get_filenames(data_files))
  image_filenames = set(get_filenames(image_files))
  text_filenames = set(get_filenames(text_files))
  label_filenames = set(get_filenames(label_files))
  intersection = data_filenames.intersection(*[f for f in [image_filenames, text_filenames, label_filenames] if f])
  if not len(intersection) == len(data_files):
    import ipdb; ipdb.set_trace()
    data_files = [f for f in data_files if get_filename(f) in intersection]
    image_files = [f for f in image_files if get_filename(f) in intersection]
    text_files = [f for f in text_files if get_filename(f) in intersection]
    label_files = [f for f in label_files if get_filename(f) in intersection]

  return data_files, image_files, text_files, label_files

def get_filename(f, keep_suffix=False):
  basename = os.path.basename(f)
  if keep_suffix: return basename
  return os.path.splitext(basename)[0]

def get_filenames(files, keep_suffix=False):
  names = [os.path.basename(fname) for fname in files]
  if keep_suffix: return names
  return [os.path.splitext(n)[0] for n in names]


def close_h5py_filelist(file_list):
  for file in file_list:
    file.close()

def open_files(file_list, key_fn, opener, name):
  return [(key_fn(file), opener(file)) for file in tqdm(file_list, desc="opening %s files" % name)]

def open_data_files(text_files, image_files, label_files):
  h5_filename = lambda x:  os.path.splitext(os.path.basename(x))[0]
  text_files = open_files(text_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py text")
  image_files = open_files(image_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py image")
  label_files = open_files(label_files, lambda x: os.path.basename(x), lambda x: pd.read_csv(x, sep=",", names=['root_postID','tree_size','max_depth','avg_depth'], header=None), "label")
  return text_files, image_files, label_files

def binary_search(files, split_func, pid):
  if not len(files): return None, None

  mid = len(files)//2
  start, end = split_func(files[mid])

  #base case here
  if len(files) == 1:
    if pid >= start and pid <= end:
      return files[0]
    else: return None, None

  if pid > end:
    return binary_search(files[mid+1:], split_func, pid)
  elif pid < start:
    return binary_search(files[:mid], split_func, pid)
  else:
    return files[mid]

def split_h5file(filename):
  # remove suffix
  # split by "_"
  filename = os.path.splitext(filename)[0]
  return [int(x) for x in filename.split("_")]

def load_h5py_data(files, pid, data_type="text", default_size=1024):
  filename, h5py_file = binary_search(files, lambda x: split_h5file(x[0]), pid)
  if not filename:
    error = "Corresponding %s file for root_postID %d wasn't found..." % (data_type, pid)
    # raise RuntimeError(error)
    return torch.zeros(default_size)
  try:
    group = h5py_file[str(pid)]
    if data_type in group.keys():
      data = group.get(data_type)[()]
      return torch.from_numpy(data).float()
    else:
      return torch.zeros(default_size)
  except Exception as ke:
    return torch.zeros(default_size)
  # except TypeError as te:
  #   print(te)
  #   import ipdb; ipdb.set_trace()
  #   raise te
  # except Exception as e:
    # raise e
