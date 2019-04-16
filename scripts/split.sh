base_path=/data/wcarvalh/twitter/junghwan


# sort train/valid/test/labels files according to first column (root_postID)
sort_data(){
  sort --parallel=20 -t, ${base_path}/train.csv > ${base_path}/train_sorted.csv
  sort --parallel=20 -t, ${base_path}/val.csv > ${base_path}/val_sorted.csv
  sort --parallel=20 -t, ${base_path}/test.csv > ${base_path}/test_sorted.csv
  sort --parallel=20 -t, ${base_path}/label.csv > ${base_path}/label_sorted.csv
}


# split (train, valid, test) files by root_postID
split_data(){
  split -l 100000 -d -a 6 ${base_path}/train_sorted.csv  ${base_path}/chunks/train/
  split -l 100000 -d -a 6 ${base_path}/val_sorted.csv  ${base_path}/chunks/val/
  split -l 100000 -d -a 6 ${base_path}/test_sorted.csv  ${base_path}/chunks/test/
  # split -l 100000 -d -a 6 ${base_path}/label_sorted.csv  ${base_path}/chunks/label/

}

# align the train/test/valid data
align_csv_image_text_label_data(){
  python data/align_csv_image_text.py \
  --csv-file ${base_path}/${1}_sorted.csv \
  --text-files /data/wcarvalh/twitter/features_from_yunseok/text/* \
  --image-files /data/wcarvalh/twitter/features_from_yunseok/image/* \
  --label-file ${base_path}/label_sorted.csv \
  --image-outdir ${base_path}/aligned_chunks/image/${1} \
  --data-outdir ${base_path}/aligned_chunks/data/${1} \
  --label-outdir ${base_path}/aligned_chunks/label/${1}\
  --csv-header ${base_path}/header.csv\
  --label-header ${base_path}/label_header.csv 
}

align_csv_image_text_label_data 'train'

# # create a pid_start,pid_end map for the labels
# create_label_map(){
#   python data/get_label_chunks_pid_map.py \
#     --files ${base_path}/chunks/label/* \
#     --header ${base_path}/label_header.csv \
#     --key root_postID \
#     --outfile ${base_path}/label_chunks_map.yaml
# }


# # split_data
# create_label_map