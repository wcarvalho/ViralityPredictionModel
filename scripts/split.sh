# first sort the large csv file by r_pid
# sort --parallel=20 -t, /mnt/brain4/datasets/Twitter/junghwan/rc_path.csv > data/rc_path_sorted.csv


ipython src/split_master_csv.py -- \
  --master-filename /mnt/brain4/datasets/Twitter/junghwan/tree_label.csv \
  --colnames r_pid tree_size max_depth avg_depth \
  --outdir data/labels/

ipython src/split_master_csv.py -- \
  --master-filename data/rc_path_sorted.csv \
  --outdir data/data/