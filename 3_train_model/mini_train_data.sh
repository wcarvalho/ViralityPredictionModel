
ipython src/trainer.py -- \
  --train-data-files sample_data/data/* \
  --train-image-files sample_data/image/* \
  --train-text-files sample_data/text/* \
  --train-label-files sample_data/labels/* \
  --data-header sample_data/header.csv \
  --label-header sample_data/label_header.csv \
  --key root_postID \
  --batch-size 20 \
  --seed 2 \
  --num-workers 1

