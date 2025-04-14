import json
import pandas as pd
import matplotlib.pyplot as plt

# open input file:
ifile = open('dataset/yelp_review.json', encoding='utf-8')

# read the first 100k entries
# set to -1 to process everything
stop = 100000

all_data = list()
for i, line in enumerate(ifile):
    if i%10000==0:
        print(i)
    if i==stop:
        break
    # convert the json on this line to a dict
    data = json.loads(line)
    # extract what we want
    text = data['text']
    stars = data['stars']
    useful = data['useful']
    funny = data['funny']
    cool = data['cool']
    # add to the data collected so far
    all_data.append([stars, text,useful, funny, cool])
# create the DataFrame
df = pd.DataFrame(all_data, columns=['stars','text', 'useful', 'funny', 'cool'])
print(df)
df.to_hdf('data/reviews.h5','reviews')
#numpy version cannot be high

ifile.close()