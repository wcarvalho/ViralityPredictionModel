python src/create_train_valid_test.py \
  --master-filename /mnt/brain4/datasets/Twitter/junghwan/rc_path.csv \
  --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
  --key root_postID \
  --outdir data/splits/ \
  --batchsize 50000