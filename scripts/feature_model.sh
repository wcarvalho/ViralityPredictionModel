# ipython data/twitter_chunk.py -- \
#   --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --dummy-user-vector

ipython data/dataloader.py -- \
  --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
  --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
  --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
  --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
  --split-map /mnt/brain4/datasets/Twitter/junghwan_chunked/split_map.yaml \
  --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
  --log-dir /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/tb \
  --checkpoint /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/ckpt.th \
  --key root_postID \
  --num-workers 20 \
  --dummy-user-vector "$1"



# ipython src/models/feature_model.py -- \
#   --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --key root_postID \
#   --dummy-user-vector


# 530 files
# 25% == 132 files
# ipython src/trainer.py -- \
#   --master-filenames $(ls /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* | head -132) \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --image-filenames /mnt/brain4/datasets/Twitter/final/image/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --split-map /mnt/brain4/datasets/Twitter/junghwan_chunked/split_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --log-dir /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/tb \
#   --checkpoint /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/ckpt.th \
#   --key root_postID \
#   --num-workers 20 \
#   --dummy-user-vector "$1"


# ipython src/trainer.py -- \
#   --master-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/data_chunks/* \
#   --label-filenames /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks/* \
#   --text-filenames /mnt/brain4/datasets/Twitter/final/text/* \
#   --label-map /mnt/brain4/datasets/Twitter/junghwan_chunked/label_chunks_map.yaml \
#   --split-map /mnt/brain4/datasets/Twitter/junghwan_chunked/split_map.yaml \
#   --header /mnt/brain4/datasets/Twitter/junghwan/rc_path_header.csv \
#   --log-dir /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/tb \
#   --checkpoint /mnt/brain4/datasets/Twitter/training_results/logs/feature_model/ckpt.th \
#   --key root_postID \
#   --num-workers 20 \
#   --dummy-user-vector "$1"
