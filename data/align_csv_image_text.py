import h5py
import subprocess
import pandas as pd
import argparse
import time
from tqdm import tqdm
from pprint import pprint
import os
import yaml
import itertools
from src.utils import path_exists

def start_end_from_filename(filename):
  name = os.path.splitext(os.path.basename(filename))[0]
  return [int(x) for x in name.split("_")]

class h5PyIterator(object):
  """docstring for h5PyIterator"""
  def __init__(self, files):
    super(h5PyIterator, self).__init__()
    self.files = files
    self.bounds = [start_end_from_filename(x) for x in self.files]
    self.iterator = 0

  def enclosing_file_indx(self, pid):
    pid_found = self.bounds[self.iterator][0] <= pid and pid <=self.bounds[self.iterator][1]
    tries = 0
    while not pid_found:
      self.iterator += 1
      if self.iterator >= len(self.bounds):
        self.iterator = 0
        tries += 1
        if tries > 1:
          return -1
      pid_found = self.bounds[self.iterator][0] <= pid and pid <=self.bounds[self.iterator][1]
    return self.iterator

def create_corresponding_h5py(csv_file, target_file, iterator, min_pid, max_pid, all_pids, datatype='img'):
  min_file_indx = iterator.enclosing_file_indx(min_pid)
  max_file_indx = iterator.enclosing_file_indx(max_pid)

  if min_file_indx == -1 and max_file_indx == -1:
    print(1)
    import ipdb; ipdb.set_trace()
  elif min_file_indx == -1:

    min_file_indx = max_file_indx
  elif max_file_indx == -1:

    max_file_indx = min_file_indx

  files = iterator.files[min_file_indx:max_file_indx+1]

  # filename = os.path.basename(csv_file)
  # target_file = os.path.join(outdir, filename)
  if not len(files): 
    print("nothing found :(")
    import ipdb; ipdb.set_trace()
  elif len(files) == 1:
    # create a symbolic link if simply have 1 file
    command='ln -s %s %s' % (files[0], target_file)
    subprocess.run(command, shell=True)
    tqdm.write('ln -s %s' % (os.path.basename(target_file)))

  else:
    out = h5py.File(target_file, 'w',  libver='latest')
    in_files = [h5py.File(file, 'r',  libver='latest') for file in files]

    for pid in tqdm(all_pids, "writing %s pids" % datatype):
      for in_file in in_files:
        try:
          value = in_file[str(pid)].get(datatype)[()]

          grp = out.create_group(str(pid))
          grp.create_dataset(datatype, data=value)

        except KeyError as ke:
          # tqdm.write(str(ke))
          continue
        except TypeError as te:
          import ipdb; ipdb.set_trace()
          continue

        except Exception as e:
          raise e

    [file.close for file in in_files]
    out.flush()
    out.close()
    tqdm.write("saved %s" % os.path.basename(target_file))



def chunks_between_pids(chunks, min_pid, max_pid):

  for chunk in tqdm(chunks, desc="chunks"):
    greater = chunk[chunk['root_postID'] >= min_pid]
    yield greater[greater['root_postID'] <= max_pid], chunk[chunk['root_postID'] > max_pid]

    # if mask.all():
    #   yield chunk, None, False # nothing outside
    # else:
    #   yield chunk[mask], chunk[chunk['root_postID'] > max_pid], True

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv-file', type=str, required=True)
  parser.add_argument('--text-files', type=str, nargs="+", required=True)
  parser.add_argument('--image-files', type=str, nargs="+", required=True)
  parser.add_argument('--label-file', type=str, required=True)
  parser.add_argument('--image-outdir', type=str, required=True)
  parser.add_argument('--data-outdir', type=str, required=True)
  parser.add_argument('--label-outdir', type=str, required=True)
  parser.add_argument('--csv-header', type=str, required=True)
  parser.add_argument('--label-header', type=str, required=True)
  parser.add_argument('--chunksize', type=int, default=1024)

  args, unknown = parser.parse_known_args()
  args = vars(args)

  path_exists(args['image_outdir'])
  # path_exists(args['text_outdir'])
  path_exists(args['label_outdir'])
  path_exists(args['data_outdir'])


  with open(args['csv_header']) as f:
    colnames = f.readlines()[0].strip().split(",")

  with open(args['label_header']) as f:
    label_colnames = f.readlines()[0].strip().split(",")

  image_iterator = h5PyIterator(args['image_files'])
  # text_iterator = h5PyIterator(args['text_files'])

  label_df = pd.read_csv(args['label_file'], sep=",", names=label_colnames, header=None)

  chunks = pd.read_csv(args['csv_file'], chunksize=args['chunksize'], names=colnames, header=None)

  relevant_chunks = []

  for text_file in tqdm(args['text_files']):
    min_pid, max_pid = start_end_from_filename(text_file)

    if relevant_chunks:
      # filter for parts of chunk from previous process that fit within bounds
      chunk = relevant_chunks[0]
      relevant_chunks[0] = chunk[chunk['root_postID'] >= min_pid][chunk['root_postID'] <= max_pid]

    # get all chunks within bounds.
    # if inside bounds is empty, go to next
    for chunk_within, chunk_outside in chunks_between_pids(chunks, min_pid, max_pid):
      if not chunk_within.empty:
        relevant_chunks.append(chunk_within)

      if not chunk_outside.empty:
        break

    filename = os.path.splitext(os.path.basename(text_file))[0]


    target_csv = os.path.join(args['data_outdir'], filename)+".csv"
    if os.path.exists(target_csv):
      tqdm.write("\n%s exists" % os.path.basename(target_csv))
    else:
      if relevant_chunks:
        # you have a valid csv
        # create df from those and write it to the path using the same filename
        main_df_chunk = pd.concat(relevant_chunks)
        main_df_chunk.to_csv(target_csv, header=None, index=False)
        tqdm.write("\nsaved %s" % os.path.basename(target_csv))
      else:
        # no valid csv
        tqdm.write("no valid csv chunk for %s" % text_file)
        continue


    target_image = os.path.join(args['image_outdir'], filename)+".h5"
    if os.path.exists(target_image):
      tqdm.write("%s exists" % os.path.basename(target_image))

    else:
      with h5py.File(text_file, 'r',  libver='latest') as text_input:
        all_pids = [key for key in text_input.keys()]
        create_corresponding_h5py(text_file, target_image, image_iterator, min_pid, max_pid, all_pids, 'img')

    # store chunk_outside as beginning of next chunk
    relevant_chunks = [chunk_outside]

    target_label = os.path.join(args['label_outdir'], filename)+".csv"
    if os.path.exists(target_label):
      tqdm.write("%s exists" % os.path.basename(target_label))
    else:
      greater = label_df[label_df['root_postID'] >= min_pid]
      sub_label_df = greater[greater['root_postID'] <= max_pid]
      sub_label_df.to_csv(target_label, header=None, index=False)
      tqdm.write("saved %s" % os.path.basename(target_label))
  # for csv_file in tqdm(args['csv_files']):
  #   min_pid = int(main_df['root_postID'].min())
  #   max_pid = int(main_df['root_postID'].max())

  #   create_corresponding_h5py(csv_file, args['image_outdir'], image_iterator, min_pid, max_pid)


if __name__ == '__main__':
  main()