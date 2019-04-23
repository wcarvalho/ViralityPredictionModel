## train.csv, val.csv, test.csv
 - *header info is at header.csv*
 - _First 80% / Mid 10% / Last 10%_    of original (root) posts as _training / validation / test_ set, respectively
 - **40 columns**, 41,756,048 / 5,494,375 / 5,700,758 rows
 
### Column descriptions
#### (*, mean, standard_deviation) Given x>=0, calculated (log(x+1) - mean)/standard_deviation. If x=-1, then (0 - mean)/standard_deviation (same as x=0).
#### all values in 64-bit format
#### In last 10 columns, features are repeated as root for parent and child
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
  - root_followers_count(float): the number of followers of the root node; logged and then normalized. (*, 6.219427088578182, 1.983826050230269)
  - root_friends_count(float): the number of friends of the root node; logged and then normalized. (*, 6.132543112565718, 1.558552156740943)
  - root_listed_count(float): the number of public lists that the root node is a member of. (*, 1.9908226618572418, 1.6484266550681868)
  - root_favourites_count(float): the number of favorites of the root node. (*, 7.842045184756162, 2.343776207989388)
  - root_statuses_count(float): the number of tweets issued by the root node (*, 8.982325811315071, 2.020504540118892)
  - root_protected(int): 1 if the root node's user has chosen to protect it's tweets else 0
  - root_verified(int): 1 if the root node's user has a verified account else 0
  - root_contributors_enabled(int): 1 if the root node's user has an account with "contributor mode" enabled else 0
  - root_default_profile(int): 1 if the root node's user has not altered the theme or background of their user profile else 0
  - root_default_profile_image(int): 1 if the root node's user has not uploaded their own profile image and a default image is use instead, else 0
  - parent_followers_count(float) (*, 6.081952464222046, 2.084475122394664)
  - parent_friends_count(float) (*, 6.023603359412002, 1.6376557397870526)
  - parent_listed_count(float) (*, 1.8721212194613213, 1.643730253573648)
  - parent_favourites_count(float) (*, 7.67355860301163, 2.408259657814672)
  - parent_statuses_count(float) (*, 8.791976864422656, 2.0817206962676327)
  - parent_protected(int)
  - parent_verified(int)
  - parent_contributors_enabled(int)
  - parent_default_profile(int)
  - parent_default_profile_image(int)
  - current_followers_count(float) (*, 10.721344928922445, 2.936430640591761)
  - current_friends_count(float) (*, 6.460655094634361, 2.705486877033605)
  - current_listed_count(float) (*, 5.468024835575275, 2.5874046405964775)
  - current_favourites_count(float) (*, 7.117646595204013, 2.9615802212388167)
  - current_statuses_count(float) (*, 9.247121989643325, 2.1551669975808427)
  - current_protected(int)
  - current_verified(int)
  - current_contributors_enabled(int)
  - current_default_profile(int)
  - current_default_profile_image (int)


## textual.csv
### Column descriptions:
  - root_postID(int): post ID of the root node of a cascade
  - text(string): a text in the content of the root node


## visual.csv
### Column descriptions: 
  - root_postID(int): post ID of the root node of a cascade
  - media_link(string): a media link in the content of the root node (there can be multiple links)

* User ids: an integer from 0-13649797


## General statistics
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
