# first sort the large csv file by r_pid
# sort --parallel=20 -t, /mnt/brain4/datasets/Twitter/junghwan/rc_path.csv > data/rc_path_sorted.csv


python src/split_master_csv.py \
  --master-filename /mnt/brain4/datasets/Twitter/junghwan/tree_label.csv \
  --header /mnt/brain4/datasets/Twitter/junghwan/target_label_header.csv \
  --outdir data/labels/

python src/split_master_csv.py \
  --master-filename data/rc_path_sorted.csv \
  --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
  --outdir data/data/