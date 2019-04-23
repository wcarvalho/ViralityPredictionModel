import math
from datetime import datetime

months = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr': '04', 'May':'05', 'Jun':'06',
        'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

def get_id(tweet):
    return tweet.get('id')

def get_time(tweet):
    return tweet.get('created_at')

def get_timestamp(tweet):
    return tweet.get('timestamp_ms')

def convert_to_timestamp(tweet):
    ### e.g. str_time: 'Sun Jan 01 04:55:49 +0000 2017'
    str_time = get_time(tweet)
    str_time_ = str_time.split()
    y = int(str_time_[-1])
    m = int(months[str_time_[1]])
    d = int(str_time_[2])
    time = str_time_[3].split(':')
    time_ = int(str_time_[4])
    return int(datetime(y,m,d,int(time[0]),int(time[1]),int(time[2]),time_).timestamp())

def get_idNtime(tweet):
    ### Return postID, userID, time ###
    return get_id(tweet), get_id(tweet.get('user')), get_time(tweet)

def get_idNtimestamp(tweet):
    ### Return postID, userID, timestamp ###
    timestamp = get_timestamp(tweet)
    if timestamp is None:
        timestamp = convert_to_timestamp(tweet)
    return get_id(tweet), get_id(tweet.get('user')), timestamp

def get_retweet(tweet):
    return tweet.get('retweeted_status')

def get_quotedtweet(tweet):
    return tweet.get('quoted_status')

def get_media(tweet):
    ### Get media(photos, video, animate GIF) ###
    mlist = []
    media = tweet.get('extended_entities')
    if media is not None:
        for m in media.get('media'):
            m_type = m['type']
            if m_type == 'photo':
                m_url = m['media_url']
            else: # video or animated_gif
                m_url = m['video_info']['variants'][0]['url']

            mlist.append([m_type, m_url])
        return mlist
    else:
        return []

def get_content(tweet):
    ### Get content info: text, hashtags and media ###
    # if a truncated tweet, get them in 'extended_tweet'
    if tweet.get('truncated') == True:
        tweet_ = tweet.get('extended_tweet')
        text = tweet_.get('full_text')
    else:
        tweet_ = tweet
        text = tweet.get('text')

    hashtags = [h['text'] for h in tweet_.get('entities').get('hashtags')]
    media = get_media(tweet_)
    return text, hashtags, media

def get_language(tweet):
    return tweet.get('lang')

def get_ymd(tweet):
    ### Convert time string to year, month, day ###
    # Example: 'Sun Jan 01 04:55:49 +0000 2017' -> '2017','01','01'
    # input: tweet 
    # output: year, month, day
    time = get_time(tweet).split()
    return time[-1], months[time[1]], time[2]

def get_log_count_of(twtuser, category):
    if twtuser[category] == -1 or twtuser[category] is None:
        return 0
    else:
        return math.log(twtuser[category] + 1)

def get_counts(tweet):
    cnts = []
    twtuser = tweet.get('user')
    cnts.append(get_log_count_of(twtuser,'followers_count'))
    cnts.append(get_log_count_of(twtuser,'friends_count'))
    cnts.append(get_log_count_of(twtuser,'listed_count'))
    cnts.append(get_log_count_of(twtuser,'favourites_count'))
    cnts.append(get_log_count_of(twtuser,'statuses_count'))
    return cnts

def get_booleans(tweet):
    bools = []
    twtuser = tweet.get('user')
    bools.append(int(twtuser['protected']))
    bools.append(int(twtuser['verified']))
    bools.append(int(twtuser['contributors_enabled']))
    bools.append(int(twtuser['default_profile']))
    bools.append(int(twtuser['default_profile_image']))
    return bools

def wanted_data(tweet):
    ### Returns true if the given tweet was posted in 2017/03~06
    ### and is written in English
    y, m, d = get_ymd(tweet)
    lang = get_language(tweet)

    orig_months = ['%.2d' % i for i in range(3,7)]
    if y == '2017' and m in orig_months and lang == 'en':
        return True
    return False

def skip_data(tweet):
    if tweet.get('delete') is not None \
            or tweet.get('status_withheld') is not None:
                return True
    return False
