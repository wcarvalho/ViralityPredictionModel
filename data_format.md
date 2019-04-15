## train.csv, val.csv, test.csv
 - *header info is at header.csv*
 - _First 80% / Mid 10% / Last 10%_    of original (root) posts as _training / validation / test_ set, respectively
 - *40 columns*, 41,756,048 / 5,494,375 / 5,700,758 rows
 
### Column descriptions
  - root_postID(int): post ID of the root node
  - root_userID(int): user ID of the root node, which is an integer from 0-13649797
  - root_timestamp(int): timestamp of the post of the root node
  - parent_postID(int): post ID of the parent node; if the parent is the root node, then 4-6th column will be identical to 1-3rd column.
  - parent_userID(int): user ID of the parent node, which is an integer from 0-13649797
  - parent_timestamp(int): timestamp of the post of the parent node
  - curr_postID(int): post ID of the current node
  - curr_userID(int): user ID of the current node, which is an integer from 0-13649797
  - curr_timestamp(int): timestamp of the post of the current node
  - path_length_to_root(int): path length from current node to the root node. e.g., if the root node is the parent of the current node, then the path length is 1.
  - root_followers_count (float): the number of followers of the root node; logged and then normalized
  - root_friends_count(float): the number of friends of the root node; logged and then normalized
  - root_listed_count(float): the number of public lists that the root node is a member of; logged and then normalized
  - root_favourites_count(float): the number of favorites of the root node; logged and then normalized
  - root_statuses_count(float): the number of tweets issued by the root node; logged and then normalized
  - root_protected(int): 1 if the root node¡¯s user has chosen to protect it¡¯s tweets else 0
  - root_verified(int): 1 if the root node¡¯s user has a verified account else 0
  - root_contributors_enabled(int): 1 if the root node¡¯s user has an account with ¡°contributor mode¡± enabled else 0
  - root_default_profile: 1 if the root node¡¯s user has not altered the theme or background of their user profile else 0
  - root_default_profile_image: 1 if the root node¡¯s user has not uploaded their own profile image and a default image is use instead, else 0
  - parent_followers_count (float)
  - parent_listed_count(float)
  - parent_favourites_count(float)
  - parent_statuses_count(float)
  - parent_protected(int)
  - parent_verified(int)
  - parent_contributors_enabled(int)
  - parent_default_profile
  - parent_default_profile_image
  - current_followers_count (float)
  - current_listed_count(float)
  - current_favourites_count(float)
  - current_statuses_count(float)
  - current_protected(int)
  - current_verified(int)
  - current_contributors_enabled(int)
  - current_default_profile
  - current_default_profile_image


## textual.csv
### Column descriptions:
  - root_postID(int): post ID of the root node of a cascade
  - text(string): a text in the content of the root node


## visual.csv
### Column descriptions: 
  - root_postID(int): post ID of the root node of a cascade
  - media_link(string): a media link in the content of the root node (there can be multiple links)

* User ids: an integer from 0-13649797


[General statistics]
- Number of posts: 52,951,180
- Number of original posts: 9,770,787
- Number of users: 13,649,798
- Number of users who post original tweets: 1,858,505
- Max number of posts per user: 17,637
- Max number of original posts per user: 17,637
- Average number of posts per user: 4.595083897944863
- Average number of original posts per user: 0.7158191645033868
- Median of number of posts per user: 1
- Median of number of original posts per user: 0