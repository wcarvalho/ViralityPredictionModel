# first sort the large csv file by r_pid
# sort --parallel=20 -t, /mnt/brain4/datasets/Twitter/junghwan/rc_path.csv > data/rc_path_sorted.csv

python src/split_master_csv.py \
  --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/tree_label_sorted.csv \
  --header /mnt/brain4/datasets/Twitter/junghwan/target_label_header.csv \
  --outdir /mnt/brain4/datasets/Twitter/junghwan_chunked/label_buckets

# python src/split_master_csv.py \
#   --master-filename data/rc_path_sorted.csv \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --outdir data/data/
