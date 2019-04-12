python data/get_label_chunks_pid_map.py \
  --files /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
  --header /mnt/brain4/datasets/Twitter/junghwan/target_label_header.csv \
  --key root_postID \
  --outfile /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml

# python data/find_max_month.py \
#   --files /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --outfile /mnt/brain4/datasets/Twitter/junghwan/max_month.txt

# python data/create_train_valid_test.py \
#   --files /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --max-month "$(< /mnt/brain4/datasets/Twitter/junghwan/max_month.txt)" \
#   --outdir /mnt/brain4/datasets/Twitter/junghwan_chunked/training_splits