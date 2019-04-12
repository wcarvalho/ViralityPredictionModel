ipython src/models/feature_model.py -- \
  --master-filename 20170913.csv \
  --file-length $(wc -l 20170913.csv) \
  --text-filename /mnt/brain4/datasets/Twitter/final/text/836727684126375937_836830804802224128.h5 \
  --output /mnt/brain4/datasets/Twitter/merged/text/merged.h5 \
  --inner-key text

# --label-filename 20170913.csv \
# --image-filename /mnt/brain4/datasets/Twitter/final/text/836727684126375937_836830804802224128.h5 \