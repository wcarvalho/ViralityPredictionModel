'''
Generate a friend's network
input: raw Twitter data from 2017-01 ~ 2017-02
output: set of edge table

* edge A->B if (1) B retweets A or (2) A mentions B
'''
import sys
import os
import gzip
import json
import csv
import twit_parser as tp

edge_list = set()
days = ['%.2d' % i for i in range(1,32)]

def create_edge(u,v):
    # create an edge between u and v
    # (u,v) means u->v
    return (tp.get_id(u), tp.get_id(v))

def gen_edge_table(read_dir, write_dir):
    for m in range(2):
        for d in days:
            rfile = read_dir + str(m+1) +'/gardenhose.2017-0' + str(m+1) + '-' + d + '.gz'
            if os.path.exists(rfile):
                print(rfile)
                with gzip.open(rfile, 'rb') as f:
                    for cnt, line in enumerate(f):
                        if cnt % 100000 == 0 and cnt != 0:
                            print(cnt, len(edge_list))

                        try:
                            tweet = json.loads(bytes.decode(line))
                        except:
                            print(rfile, cnt, sys.exc_info()[0])
                            break

                        if tp.skip_data(tweet):
                            continue

                        rtweet = tweet.get('retweeted_status')
                        qtweet = tweet.get('quoted_status')
                        if rtweet is not None or qtweet is not None:
                            tweet_ = rtweet if rtweet is not None else qtweet
                            # create an edge between the retweeter and retweetee
                            # only if the edge does not already exists
                            edge_list.add(create_edge(tweet_.get('user'), tweet.get('user')))
                        else: # original post
                            # create an edge if the current user mentions anyone
                            if tweet['entities']['user_mentions']:
                                for user in tweet['entities']['user_mentions']:
                                    edge_list.add(create_edge(tweet.get('user'), user))

    with open(write_dir + 'edge_list.csv', 'w') as f:
       csv.writer(f, delimiter=',').writerows(edge_list) 

def main(args):
    if len(args) != 2:
        print('Enter an input directory and output directory')
        sys.exit() 
    gen_edge_table(args[0], args[1])

if __name__ == '__main__':
    main(sys.argv[1:])
