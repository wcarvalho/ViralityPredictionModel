```

# sort files according to first column (root_postID)
sort --parallel=20 -t, /mnt/brain4/datasets/Twitter/junghwan/rc_path.csv > /mnt/brain4/datasets/Twitter/junghwan_chunked/rc_path_sorted.csv
sort --parallel=20 -t, /mnt/brain4/datasets/Twitter/junghwan/tree_label.csv > /mnt/brain4/datasets/Twitter/junghwan_chunked/tree_label_sorted.csv

# split files into (root_postID)
split -l 100000 -d -a 6 /mnt/brain4/datasets/Twitter/junghwan_chunked/rc_path_sorted.csv  /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/
split -l 100000 -d -a 6 /mnt/brain4/datasets/Twitter/junghwan_chunked/tree_label_sorted.csv /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/

# create a pid_start,pid_end map for the labels
python data/get_label_chunks_pid_map.py \
  --files /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
  --header /mnt/brain4/datasets/Twitter/junghwan/target_label_header.csv \
  --key root_postID \
  --outfile /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml

# get the file with the largest timestamp
sort -t, -nrk3,3 /mnt/brain4/datasets/Twitter/junghwan_chunked/rc_path_sorted.csv | head -1 > /mnt/brain4/datasets/Twitter/junghwan/newest.txt

# now split data based on root_pid and using labels file into train/validation/test
# ____NOT DONE_____
python data/create_train_valid_test.py \
  --files /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
  --header /mnt/brain4/datasets/Twitter/junghwan/target_label_header.csv \
  --key root_postID \
  --outfile /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml

```