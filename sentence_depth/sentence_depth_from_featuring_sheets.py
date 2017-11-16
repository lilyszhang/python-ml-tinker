import pandas as pd
import csv
import os
from HTMLParser import HTMLParser
import spacy
from spacy.lang.en import English
import nltk
import numpy as np
import matplotlib.pyplot as plt


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        t = nltk.Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        return t
    else:
        return node.orth_

def get_depths(doc):
    depths = []
    for sent in doc.sents:
        depth = []
        t = to_nltk_tree(sent.root)
        n_leaves = len(t.leaves())
        leavepos = set(t.leaf_treeposition(n) for n in range(n_leaves))
        for pos in t.treepositions():
            if pos not in leavepos:
                depth.append(len(pos))
        depths.append(max(depth))
    return depths

# data = csv.reader(open("storage/featuring/answer-only-featuring.csv", 'rU'))
# writer = csv.writer(open("storage/featuring/answer-only-featuring-raw.csv", 'w'))
# fields = data.next()
# toWrite = [['body','featured']]
# for row in data:
#     item = row
#     item[0] = strip_tags(item[0])
#     toWrite.append(item)
# writer.writerows(toWrite)

df = pd.read_csv("storage/featuring/answer-only-featuring.csv")
X = df.loc[:, 'body'].values
y = df.loc[:, 'featured'].values

featured_posts = []
not_featured_posts = []

for i in range(len(X)):
    if y[i] == 1:
        featured_posts.append(X[i])
    else:
        not_featured_posts.append(X[i])

#calculate depths
nlp = spacy.load('en_core_web_sm')
featured_posts = featured_posts[:278]
featured_max_depths = []
featured_total_depth = []
featured_avg_depths = []
for post in featured_posts:
    doc = nlp(strip_tags(post).decode('utf-8'))
    try:
        post_depths = get_depths(doc)
    except:
        continue
    try:
        featured_max_depths.append(max(post_depths))
        featured_total_depth.append(sum(post_depths))
        featured_avg_depths.append(sum(post_depths)/float(len(post_depths)))
    except:
        continue

not_featured_max_depths = []
not_featured_total_depth = []
not_featured_avg_depths = []
# writer = csv.writer(open("storage/featuring/not-featured-high-max-depth-fall.csv", 'w'))
# toWrite = []
for post in not_featured_posts:
    doc = nlp(strip_tags(post).decode('utf-8'))
    try:
        post_depths = get_depths(doc)
    except:
        continue
    try:
        not_featured_max_depths.append(max(post_depths))
        not_featured_total_depth.append(sum(post_depths))
        not_featured_avg_depths.append(sum(post_depths)/float(len(post_depths)))
    except:
        continue
#     if max(post_depths) > 13:
#         print doc
#         toWrite.append(doc)
# writer.writerows(toWrite)

print len(not_featured_posts)
print len(featured_posts)

# visualize data
# plt.subplot(221)
# x = not_featured_avg_depths
# y = not_featured_max_depths
# plt.scatter(x,y,label="Not Featured",marker=".")
# x = featured_avg_depths
# y = featured_max_depths
# plt.scatter(x,y,label="Featured",marker=".")
# plt.xlabel('Average Sentence Depth')
# plt.ylabel('Max Sentence Depth')
# plt.legend()
#
# plt.subplot(222)
# x = not_featured_total_depth
# y = not_featured_max_depths
# plt.scatter(x,y,label="Not Featured",marker=".")
# x = featured_total_depth
# y = featured_max_depths
# plt.scatter(x,y,label="Featured",marker=".")
# plt.xlabel('Total Sentence Depth')
# plt.ylabel('Max Sentence Depth')
# plt.legend()
#
# plt.subplot(223)
# x = not_featured_total_depth
# y = not_featured_avg_depths
# plt.scatter(x,y,label="Not Featured",marker=".")
# x = featured_total_depth
# y = featured_avg_depths
# plt.scatter(x,y,label="Featured",marker=".")
#
# plt.xlabel('Total Sentence Depth')
# plt.ylabel('Average Sentence Depth')
# plt.legend()

# histograms
x = [not_featured_max_depths, featured_max_depths]
plt.hist(x, label=["Not Featured", "Featured"])
plt.xlabel('Max Sentence Depth')
plt.xticks(range(21))
plt.ylabel('Number of Posts')

plt.show()
