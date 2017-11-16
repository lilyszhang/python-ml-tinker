import nltk
import numpy as np
import matplotlib.pyplot as plt
import re
import spacy
from spacy.lang.en import English
import MySQLdb
from HTMLParser import HTMLParser
import sys
import csv
from oauth2client.service_account import ServiceAccountCredentials
from os.path import dirname, join
import dotenv
from os import getenv
import json

#clean HTML text to raw feature_extraction
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

# pull questions from database
dotenv.load_dotenv(join(dirname(__file__),'.','.env'))
db = MySQLdb.connect(host=getenv('QUESTIONS_DB_HOST'), user=getenv('QUESTIONS_DB_USERNAME'), passwd=getenv('QUESTIONS_DB_PASSWORD'), db=getenv('QUESTIONS_DB_DATABASE'), sql_mode='NO_ENGINE_SUBSTITUTION')

cursor = db.cursor(MySQLdb.cursors.DictCursor)

featured_query = "SELECT body from `answers` WHERE is_featured=1 AND deleted_at IS NULL AND `created_at` >= '2017-03-01 00:00:00' ORDER BY RAND() LIMIT 1000"
not_featured_query = "SELECT body from `answers` WHERE is_featured=0 AND deleted_at IS NULL AND `created_at` >= '2017-03-01 00:00:00' ORDER BY RAND() LIMIT 1000"
moderated_query = "SELECT body from `answers` WHERE status_id=2 AND deleted_at IS NULL AND `created_at` >= '2017-03-01 00:00:00' ORDER BY RAND() LIMIT 1000"

cursor.execute(featured_query)
db.commit()
featured_posts = cursor.fetchall()

cursor.execute(not_featured_query)
db.commit()
not_featured_posts = cursor.fetchall()

cursor.execute(moderated_query)
db.commit()
moderated_posts = cursor.fetchall()

#calculate depths
nlp = spacy.load('en_core_web_sm')
featured_max_depths = []
featured_total_depth = []
featured_avg_depths = []
for post in featured_posts:
    doc = nlp(strip_tags(post['body']).decode('utf-8'))
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
for post in not_featured_posts:
    doc = nlp(strip_tags(post['body']).decode('utf-8'))
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

moderated_max_depths = []
moderated_total_depth = []
moderated_avg_depths = []
for post in moderated_posts:
    doc = nlp(strip_tags(post['body']).decode('utf-8'))
    try:
        post_depths = get_depths(doc)
    except:
        continue
    try:
        moderated_max_depths.append(max(post_depths))
        moderated_total_depth.append(sum(post_depths))
        moderated_avg_depths.append(sum(post_depths)/float(len(post_depths)))
    except:
        continue

print moderated_posts

# visualize data

# scatter plots
# plt.subplot(221)
# x = not_featured_avg_depths
# y = not_featured_max_depths
# plt.scatter(x,y,label="Not Featured",marker=".")
# x = featured_avg_depths
# y = featured_max_depths
# plt.scatter(x,y,label="Featured",marker=".")
# x = moderated_avg_depths
# y = moderated_max_depths
# plt.scatter(x,y,label="Moderated", marker=".")
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
# x = moderated_total_depth
# y = moderated_max_depths
# plt.scatter(x,y,label="Moderated", marker=".")
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
# x = moderated_total_depth
# y = moderated_avg_depths
# plt.scatter(x,y,label="Moderated", marker=".")
# plt.xlabel('Total Sentence Depth')
# plt.ylabel('Average Sentence Depth')
# plt.legend()

# histograms
x = [not_featured_max_depths, featured_max_depths, moderated_max_depths]
plt.hist(x, label=["Not Featured", "Featured", "Moderated"])
plt.xlabel('Max Sentence Depth')
# plt.xticks(range(21))
plt.ylabel('Number of Posts')

# x = featured_max_depths
# plt.hist(x, label="Featured")
# plt.xlabel('Max Sentence Depth')
# plt.ylabel('Number of Posts')
#
# x = moderated_max_depths
# plt.hist(x, label="Moderated")
# plt.xlabel('Max Sentence Depth')
# plt.ylabel('Number of Posts')
plt.legend()

plt.show()
