# ViralityPredictionModel

## Input
### 1. set of rows that contains r_pid, r_uid, r_t, p_pid, p_uid, p_t, c_pid, c_uid, c_t (where r=root, p=parent, c=child, pid=post_id, uid=user_id, t=timestamp)
### 2. Image array (key,value dictionary, created with hdf5 format):  key=r_pid, value={"vector": image or video_vector, "label": image/video/(gif?)}
### 3. Text array (key,value dictionary, created with hdf5 format):    key=r_pid, value={"vector": text_vector}
