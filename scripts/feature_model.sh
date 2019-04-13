# ipython data/twitter_chunk.py -- \
#   --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --dummy-user-vector

# ipython src/models/feature_model.py -- \
#   --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --dummy-user-vector


ipython src/trainer.py -- \
  --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
  --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
  --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
  --image-filenames /mnt/brain4/datasets/Twitter/final/image/* \
  --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
  --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
  --log-dir /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/tb \
  --checkpoint /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/ckpt.th \
  --key root_postID \
  --dummy-user-vector
