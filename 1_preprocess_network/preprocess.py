'''
input: raw Twitter data from 2017-03 ~ 2017-09
output: preprocessed Twitter data in csv form
    [root_postID, root_userID, root_time,
    parent_postID, parent_userID, parent_time,
    curr_postID, curr_userID, curr_time, 
    path_length_to_root,
    root_followers_cnt, root_friends_cnt, root_listed_cnt, root_favourites_cnt, root_statuses_cnt,
    root_protected, root_verified, root_contributors_enabled, root_default_profile, root_profile_image
    parent_followers_cnt, parent_friends_cnt, parent_listed_cnt, parent_favourites_cnt, parent_statuses_cnt,
    parent_protected, parent_verified, parent_contributors_enabled, parent_default_profile, parent_profile_image
    curr_followers_cnt, curr_friends_cnt, curr_listed_cnt, curr_favourites_cnt, curr_statuses_cnt,
    curr_protected, curr_verified, curr_contributors_enabled, curr_default_profile, curr_profile_image
]

* Twitter has three types of posts: (1) original, (2) retweeted post, and (3) quoted post.
We consider (1) as the original post and (2) and (3) as the shared post.
* We store the original posts from 2017-03 ~ 2017-06 and those posts shared until 2017-09.
* As for the definition of the previous user, we look up the friend network we created and 
find friends of the current user. If one of them have shared the post before the current user,
we define that user as the previous user of the current user. If there is none, we define
the root user as the previous user.
* We only save the post written in English.
* Only the shared posts are saved into a file. That is, we do not save the information of
original posts separately. Every entry has the information of the original post shared by
a current user, a friend of the user who also shared the same post, and the current user.
'''
import sys
import os
import gzip
import json
import csv
import random
import pandas as pd
import twit_parser as tp

days = ['%.2d' % (i+1) for i in range(31)]

def load_edgeList(friend_dir):
    ### Load the edge list of the friend network ###
    edge_list = set()
    cnt = 0
    with open(friend_dir + 'edge_list.csv', 'r') as f:
        for edge in csv.reader(f):
            if cnt % 1000000 == 0:
                print(cnt)
            edge_list.add((int(edge[0]),int(edge[1])))
            cnt += 1
        return edge_list

def add_rtwt(rtwt, root_postID, curr_postID, curr_userID, curr_time, level):
    ### Update the 'rtwt' dictionary used in get_features function ###
    if curr_userID in rtwt:
        rtwt[str(root_postID)].append((curr_postID, curr_userID, curr_time, level))
    else:
        rtwt[str(root_postID)] = [(curr_postID, curr_userID, curr_time, level)]

def get_parent(rtwt, root_postID, curr_userID, edge_list):
    ### Return the information of the previous post ###
    rtwts = rtwt.get(str(root_postID))
    if rtwts is not None:
        for rt in reversed(rtwts):
            if (rt[1], curr_userID) in edge_list:
                return rt
    return 0,0,0,0

def update_user_features(usr_feats, user_id, features):
    usr_feats[user_id] = features

def save_textNmedia(root_postID, text, media, write_dir):
    ### Save two files: textual.csv and visual.csv ###
    with open(write_dir + 'textual.csv', 'a+') as fp:
        csv.writer(fp, delimiter=',').writerow([root_postID, text])
    
    if media:
        mlinks = []
        for m in media:
            mlinks.append(m[-1])
        with open(write_dir + 'visual.csv', 'a+') as fp:
            csv.writer(fp, delimiter=',').writerow([root_postID] + mlinks)

def get_features(read_dir, write_dir, edge_list):
    rtwt = dict()
    usr_feats = dict()
    pids = set()

    for m in range(3,10):
        for d in days:
            # Read a raw data file
            rfile = read_dir + str(m) + '/' + 'gardenhose.2017-0' + str(m) + '-' + d + '.gz'
            if os.path.exists(rfile):
                print(rfile)
                with gzip.open(rfile, 'rb') as f:
                    for cnt, line in enumerate(f):
                        if cnt % 10000 == 0:
                            print(cnt, len(rtwt))
                        try:
                            tweet = json.loads(bytes.decode(line))

                            # Skip if it's a deleted tweet
                            # or a status_withheld tweet
                            if tp.skip_data(tweet):
                                continue

                            # Skip if the same tweet is already saved
                            if tweet.get('id') in pids:
                                continue

                            rtweet = tp.get_retweet(tweet)
                            qtweet = tp.get_quotedtweet(tweet)
                            
                            # Not original post
                            if rtweet is not None or qtweet is not None: 
                                # Retweet of quoted tweet
                                if rtweet is not None and qtweet is not None: 
                                    tweet_ = qtweet
                                else:
                                    tweet_ = rtweet if rtweet is not None else qtweet

                                if tp.wanted_data(tweet_):
                                    # Root post info
                                    root_postID, root_userID, root_time = tp.get_idNtimestamp(tweet_)
                                    
                                    # Current post info
                                    curr_postID, curr_userID, curr_time = tp.get_idNtimestamp(tweet)

                                    if root_userID == curr_userID:
                                        continue

                                    # Root user info
                                    root_feats = tp.get_counts(tweet) \
                                            + tp.get_booleans(tweet)
                                    update_user_features(usr_feats, root_userID, root_feats)

                                    # Current user info
                                    curr_feats = tp.get_counts(tweet) \
                                            + tp.get_booleans(tweet)
                                    update_user_features(usr_feats, curr_userID, curr_feats)
                                    pids.add(curr_postID)

                                    # Previous post (parent node) info
                                    parent_postID, parent_userID, parent_time, parent_level \
                                            = get_parent(rtwt, root_postID, curr_userID, edge_list)
                                    if parent_postID == 0:
                                        parent_postID, parent_userID, parent_time \
                                                = root_postID, root_userID, root_time
                                        parent_feats = root_feats
                                        curr_depth = 1
                                        add_rtwt(rtwt, root_postID,\
                                                curr_postID, curr_userID, curr_time, curr_depth)
                                    else:
                                        parent_feats = usr_feats[parent_userID]
                                        curr_depth = parent_level + 1
                                        add_rtwt(rtwt, root_postID,\
                                                curr_postID, curr_userID, curr_time, curr_depth)

                                    # Content info
                                    text, _, media = tp.get_content(tweet_)
                                else:
                                    continue
                            else: # Original post
                                continue

                            preprocessed = [root_postID, root_userID, root_time,\
                                        parent_postID, parent_userID, parent_time, \
                                        curr_postID, curr_userID, curr_time, curr_depth] \
                                        + root_feats + parent_feats + curr_feats

                            ### Write the result to a file ###
                            with open(write_dir + 'tmp01.csv', 'a+') as fp:
                                csv.writer(fp, delimiter=',').writerow(preprocessed)
                            save_textNmedia(root_postID, text, media, write_dir)

                        except:
                            print(rfile, cnt, sys.exc_info()[0])
                            sys.exit()
    usr_feats = None
    edge_list = None
    pids = None
    return rtwt

def get_sampledIDs_and_targetLabel(rtwt, write_dir):
    twos = []
    tree_size = dict()
    for root_pid, children in rtwt.items():
        tsize = len(children) + 1
        if tsize == 2:
            twos.append(root_pid)
        
        max_depth = 0
        sum_depth = 0
        for child in children:
            if child[-1] > max_depth:
                max_depth = child[-1]
            sum_depth += child[-1]
        tree_size[root_pid] = [tsize, max_depth, sum_depth/(tsize-1)]

    print('The number of trees with size 2: ', len(twos))

    removed = set(random.sample(twos, int(len(twos)*0.9)))
    print('The number of trees to be sampled: ', len(twos)-len(removed))

    # Write the target label file
    with open(write_dir + 'target_label.csv', 'w') as wfile:
        for root_pid, labels in tree_size.items():
            csv.writer(wfile, delimiter=',').writerow([root_pid] + labels)
    print('Wrote target_label.csv in ', write_dir)

    twos = None
    rtwt = None
    tree_size = None
    return removed

def sample_twos(removed, write_dir):
    tweets = pd.read_csv(write_dir + 'tmp01.csv', header = None)
    tweets = tweets[~tweets[0].isin(removed)]
    print('Sampled 10% of cascades with tree-size 2')
    removed = None
    return tweets

def reassign_uids(tweets):
    ### Reassign user ids ###
    user_ids = set(tweets[1].tolist() + tweets[4].tolist() + tweets[7].tolist())
    
    new_uids = dict()
    for idx, uid in enumerate(user_ids):
        new_uids[uid] = idx

    for i in range(3):
        tweets[3*i+1] = tweets[3*i+1].replace(new_uids)
    new_uids = None
    user_ids = None
    return tweets

def divide_twts(tweets, write_dir):
    ### Divide the dataset into train, validation and test sets ###

    tweets = reassign_uids(tweets)
    print('Reassigned user IDs')

    tweets_ = tweets.copy()
    tweets_ = pd.concat([tweets_[0], tweets_[2]], axis = 1)
    tweets_ = tweets_.sort_values(by = 2)
    print('Sorted the root postID by timestamp')

    # First 80%, next 10%, and the rest becomes train, val and test sets, respectively
    pid_num = len(tweets_.index)
    train_idx = int(pid_num * 0.8)
    val_idx = int(pid_num * 0.1)

    train_pids = set(tweets_.iloc[:train_idx, 0])
    val_pids = set(tweets_.iloc[train_idx:train_idx+val_idx, 0])
    test_pids = set(tweets_.iloc[train_idx+val_idx:, 0])
    tweets_ = None

    train = tweets[tweets[0].isin(train_pids)]
    val = tweets[tweets[0].isin(val_pids)]
    test = tweets[tweets[0].isin(test_pids)]
    print('Divided the data into train, val and test sets')
    tweets = None

    means = []
    stds = []
    for i in range(3):
        means.append(train.iloc[:, 10*(i+1): 10*(i+1)+5].mean())
        stds.append(train.iloc[:, 10*(i+1): 10*(i+1)+5].std())
    
    for i in range(3):
        idx = 10 * (i + 1)
        train.iloc[:, idx:idx+5] = (train.iloc[:, idx:idx+5] - means[i])/stds[i]
        val.iloc[:, idx:idx+5] = (val.iloc[:, idx:idx+5] - means[i])/stds[i]
        test.iloc[:, idx:idx+5] = (test.iloc[:, idx:idx+5] - means[i])/stds[i]
    print('Normalized the counts in user features')
    
    with open(write_dir + 'train_set.csv', 'w') as wfile:
        train.to_csv(wfile, header=False, index=False)

    with open(write_dir + 'val_set.csv', 'w') as wfile:
        val.to_csv(wfile, header=False, index=False)

    with open(write_dir + 'test_set.csv', 'w') as wfile:
        test.to_csv(wfile, header=False, index=False)
    print('Wrote train_set.csv, val_set.csv, test_set.csv in ', write_dir)

    os.remove(write_dir + 'tmp01.csv')

def main(args):
    if len(args) != 3:
        print('Enter friend network directory, raw data directory and output directory.')
        print('e.g. python preprocess.py ../friend_dir/ ../raw/ ../output/')
        sys.exit()

    print('loading...')
    edge_list = load_edgeList(args[0])
    print('Edge list loaded')

    rtwt = get_features(args[1], args[2], edge_list)
    print('Content and user features extracted')
    print('Also wrote textual.csv and visual.csv')

    # Sample 10% of cascades with tree size 2
    # And save target_label.csv
    removed = get_sampledIDs_and_targetLabel(rtwt, args[2])
    tweets = sample_twos(removed, args[2])

    # Reassign userIDs, normalize the counts in user features
    # and divide the data into train, val, and test sets
    divide_twts(tweets, args[2])

if __name__ == '__main__':
    main(sys.argv[1:])
