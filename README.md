# ViralityPredictionModel

## Input

### 1. graph
set of rows that contains r_pid, r_uid, r_t, p_pid, p_uid, p_t, c_pid, c_uid, c_t (where r=root, p=parent, c=child, pid=post_id, uid=user_id, t=timestamp)

### 2. Image array

(key,value dictionary, created with hdf5 format):  key=r_pid, value={"img(or vid)": image or video_vector}

### 3. Text array

(key,value dictionary, created with hdf5 format):    key=r_pid, value={"text": text_vector}

- Extracted from BERT
- Specific model name: bert-base-multilingual-cased
- dim for each vector: 768

## Files
- brain4/datasets/Twitter/final/image
- brain4/datasets/Twitter/final/text

## TODO:
- Jungwhan: for each r_pid, p_pid, c_pid, have an extra column with a unique_id uid. uid should be within [0,V-1], where V is the total number of unique ids.
  - use HDf5
  - separate JSON, YAML file with V. maybe other useful, global values and statistics?
  - for each row, add the length between (r_pid, p_pid)
  - wiener index for tree starting at (r_pid, p_pid, c_id)?
- Wilka: use negative sampling to use another random batch
  - 2: better way to shuffle. Yunseok is recommending I change the format of the CSV to a HD5. Use h5py
