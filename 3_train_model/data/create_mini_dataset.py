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
from data.align_csv_image_text import start_end_from_filename, h5PyIterator, create_corresponding_h5py, chunks_between_pids

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv-file', type=str, required=True)
  parser.add_argument('--text-file', type=str, required=True)
  parser.add_argument('--image-file', type=str, required=True)
  parser.add_argument('--label-file', type=str, required=True)
  parser.add_argument('--image-outdir', type=str, required=True)
  parser.add_argument('--data-outdir', type=str, required=True)
  parser.add_argument('--text-outdir', type=str, required=True)
  parser.add_argument('--label-outdir', type=str, required=True)
  parser.add_argument('--csv-header', type=str, required=True)
  parser.add_argument('--label-header', type=str, required=True)
  parser.add_argument('--chunksize', type=int, default=1024)
  parser.add_argument('--datasize', type=int, default=100)

  args, unknown = parser.parse_known_args()
  args = vars(args)

  path_exists(args['image_outdir'])
  path_exists(args['text_outdir'])
  path_exists(args['label_outdir'])
  path_exists(args['data_outdir'])


  with open(args['csv_header']) as f:
    data_header = f.readlines()[0].strip().split(",")

  with open(args['label_header']) as f:
    label_header = f.readlines()[0].strip().split(",")
  
  data        = pd.read_csv(args['csv_file'], sep=",", names=data_header, header=None)
  labels      = pd.read_csv(args['label_file'], sep=",", names=label_header, header=None)
  image_data  = h5py.File(args['image_file'], 'r')
  text_data   = h5py.File(args['text_file'], 'r')


  mini_batch = data.iloc[:100]
  mini_labels = []
  mini_image_file = h5py.File(os.path.join(args['image_outdir'], "mini_image.h5py"), 'w',  libver='latest')
  mini_text_file = h5py.File(os.path.join(args['text_outdir'], "mini_text.h5py"), 'w',  libver='latest')

  unique_pids = mini_batch['root_postID'].unique()
  for pid in unique_pids:
    # get relevant labels
    mini_labels.append(labels[labels['root_postID'] == pid])

    try:
      # save relevant image data
      value = image_data[str(pid)].get("img")[()]
      grp = mini_image_file.create_group(str(pid))
      grp.create_dataset(datatype, data=value)
    except Exception as e:
      pass

    try:
      # save relevant text data
      value = text_data[str(pid)].get("text")[()]
      grp = mini_text_file.create_group(str(pid))
      grp.create_dataset(datatype, data=value)
    except Exception as e:
      pass

  mini_labels = pd.concat(mini_labels)
  mini_labels.to_csv(os.path.join(args['label_outdir'], "labels.csv"), header=None, index=False)
  mini_batch.to_csv(os.path.join(args['data_outdir'], "data.csv"), header=None, index=False)

  mini_image_file.flush()
  mini_image_file.close()

  mini_text_file.flush()
  mini_text_file.close()
  image_data.close()
  text_data.close()


  # image_iterator = h5PyIterator(args['image_files'])
  # text_iterator = h5PyIterator(args['text_files'])

  # label_df = pd.read_csv(args['label_file'], sep=",", names=label_colnames, header=None)

  # chunks = pd.read_csv(args['csv_file'], chunksize=args['chunksize'], names=colnames, header=None)

  # relevant_chunks = []

  # for text_file in tqdm(args['text_files']):
  #   min_pid, max_pid = start_end_from_filename(text_file)

  #   if relevant_chunks:
  #     # filter for parts of chunk from previous process that fit within bounds
  #     chunk = relevant_chunks[0]
  #     relevant_chunks[0] = chunk[chunk['root_postID'] >= min_pid][chunk['root_postID'] <= max_pid]

  #   # get all chunks within bounds.
  #   # if inside bounds is empty, go to next
  #   for chunk_within, chunk_outside in chunks_between_pids(chunks, min_pid, max_pid):
  #     if not chunk_within.empty:
  #       relevant_chunks.append(chunk_within)

  #     if not chunk_outside.empty:
  #       break

  #   filename = os.path.splitext(os.path.basename(text_file))[0]
  #   if relevant_chunks: main_df_chunk = pd.concat(relevant_chunks)

  #   target_csv = os.path.join(args['data_outdir'], filename)+".csv"
  #   if os.path.exists(target_csv):
  #     tqdm.write("\n%s exists" % os.path.basename(target_csv))
  #   elif relevant_chunks and not main_df_chunk.empty:
  #     # you have a valid csv
  #     # create df from those and write it to the path using the same filename

  #     main_df_chunk.to_csv(target_csv, header=None, index=False)
  #     tqdm.write("\nsaved %s" % os.path.basename(target_csv))
  #   else:
  #     # no valid csv
  #     tqdm.write("no valid csv chunk for %s" % text_file)
  #     continue


  #   target_image = os.path.join(args['image_outdir'], filename)+".h5"
  #   if os.path.exists(target_image):
  #     tqdm.write("%s exists" % os.path.basename(target_image))

  #   else:
  #     with h5py.File(text_file, 'r',  libver='latest') as text_input:
  #       all_pids = [key for key in text_input.keys()]
  #       create_corresponding_h5py(text_file, target_image, image_iterator, min_pid, max_pid, all_pids, 'img')

  #   # store chunk_outside as beginning of next chunk
  #   relevant_chunks = [chunk_outside]

  #   target_label = os.path.join(args['label_outdir'], filename)+".csv"
  #   if os.path.exists(target_label):
  #     tqdm.write("%s exists" % os.path.basename(target_label))
  #   else:
  #     greater = label_df[label_df['root_postID'] >= min_pid]
  #     sub_label_df = greater[greater['root_postID'] <= max_pid]
  #     sub_label_df.to_csv(target_label, header=None, index=False)
  #     tqdm.write("saved %s" % os.path.basename(target_label))
  # # for csv_file in tqdm(args['csv_files']):
  # #   min_pid = int(main_df['root_postID'].min())
  # #   max_pid = int(main_df['root_postID'].max())

  # #   create_corresponding_h5py(csv_file, args['image_outdir'], image_iterator, min_pid, max_pid)


if __name__ == '__main__':
  main()