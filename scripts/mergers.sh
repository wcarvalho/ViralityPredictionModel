merge_image(){
  ipython src/merge_h5py.py -- \
    --files /mnt/brain4/datasets/Twitter/final/image/* \
    --output data/image/merged.h5 \
    --inner-key image
}

merge_text(){
  # ls -rt /mnt/brain4/datasets/Twitter/final/text/* > data/text_files.txt

  ipython src/merge_h5py.py -- \
    --files /mnt/brain4/datasets/Twitter/final/text/* \
    --output data/text/merged.h5 \
    --inner-key text
}

merge_text