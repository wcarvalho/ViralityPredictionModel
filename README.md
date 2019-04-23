# ViralityPredictionModel

## Input

### 1. graph
set of rows that contains r_pid, r_uid, r_t, p_pid, p_uid, p_t, c_pid, c_uid, c_t (where r=root, p=parent, c=child, pid=post_id, uid=user_id, t=timestamp)

### 2. Image array

(key,value dictionary, created with hdf5 format):  key=r_pid, value={"img(or vid)": image or video_vector}, #512/1024, pass through few layers of NN

- Extracted from ResNet-152
- dim for each vector: 2048 (final fully-connected layer before the last one)

### 3. Text array

(key,value dictionary, created with hdf5 format):    key=r_pid, value={"text": text_vector} #768, pass through few layers of NN

- Extracted from BERT
- Specific model name: bert-base-multilingual-cased
- dim for each vector: 768 (final hidden of 'CLS' token)

## Files
- /mnt/brain4/datasets/Twitter/final/image
- /mnt/brain4/datasets/Twitter/final/text
- /mnt/brain4/datasets/Twitter/junghwan
