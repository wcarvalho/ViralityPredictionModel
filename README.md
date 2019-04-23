# ViralityPredictionModel

## Structure of the repository
- We put number for each subtask in chronological order. The data pipeline flows from 1 to 3.
- In each folder, there is **'run.sh' (not MAKEFILE)** file. You can run each part of the codes just by executing it.


## Input data for our model

### 1. Graph
- Consists of a set of rows that contains r_pid, r_uid, r_t, p_pid, p_uid, p_t, c_pid, c_uid, c_t 
  (r=root, p=parent, c=child, pid=post_id, uid=user_id, t=timestamp)


### 2. Text 

- HDF5 Data format: key=r_pid, value={"text": text_vector} 
- Extracted from BERT (Specific model name: bert-base-multilingual-cased)
- Dim for each vector: 768 (final hidden of 'CLS' token)
- Pass through four layers of NN in '3_train_model'


### 3. Image 

- HDF5 Data format: key=r_pid, value={"img(or vid)": image or video_vector}
- Features are extracted from ResNet-152
- Dim for each vector: 2048 (final fully-connected layer before the last one)
- Pass through four layers of NN in '3_train_model'

