import pandas as pd
import dateutil.parser

df = pd.read_csv("20170913.csv", sep=",", names=["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t", "text", "data"], header=None)
df['month'] = df['r_t'].transform(lambda x: dateutil.parser.parse(x).month)

max_ = 5
df_train = df[df['month'] < max_]
df_val_test = df[df['month'] >= max_]
df_val = df_val_test[df_val_test['r_pid'] % 2 == 0]
df_test = df_val_test[df_val_test['r_pid'] % 2 == 1]
# dfs = dict(tuple(df.groupby('month')))
# sorted_list = sorted(dfs.items(), key=operator.itemgetter(0))

# train = sorted_list[:-2]
# val_test = sorted_list[-2:]
# dfs = dict(tuple(df.groupby('r_pid')))

import ipdb; ipdb.set_trace()