import nltk
import re
import spacy
import pattern.en as pattern

# grammar = nltk.CFG.fromstring("""
#     S -> NP VP
#     PP -> P NP
#     NP -> Det N | Det N PP | 'I'
#     VP -> V NP | VP PP
#     Det -> 'an' | 'my'
#     N -> 'elephant' | 'pajamas'
#     V -> 'shot'
#     P -> 'in'
#   """)

sentence = "I shot an elephant in my pajamas"
post = "Personally I dont do that because I know better even when there are times when I do, but dont realize it. I usually dont like giving people the benefit of the doubt just by how they looked because you could be 100% wrong about who you thought they were. I have no recount from an experience but from observation, for example, I used to work at a fast food business andmy manager would always apply this halo effect to the new attractive employees. I noticed that he was much harder on those who werent attractive and were males, rather than the ones who were most attractive. I also observed that since they were given many loose benefits, their performance in the work environment decreased because they werent micro managed, whereas those who werent given loose benefits, had the best work attitude and performance do to the discipline held against them. After noticing this, I made a short promise to myself to never give anyone special privileges or benefits based on how they looked rather than who they are, because it could also be unfair to those who have no control over how they look. This halo effect is probably why surgeries, fillers, and changing your looks is now a trend."

# nltk parser using specific grammar

# sent = sentence.split()
# parser = nltk.ChartParser(grammar)
# trees = parser.parse(sent)
# for tree in trees:
#     print tree

# this just counts point of view pronouns and number of sentences (periods)
# count_i = 0
# count_p = 0
# count_third = 0
# third = ['he', 'she', 'his', 'hers', 'it']
# for word in post.split():
#     if word == 'I':
#         count_i += 1
#     elif word == '.':
#         count_p += 1
#     elif word in third:
#         count_third += 1
# print count_i
# print count_p
# print count_third
# print post.split()

# using spacy
#lemma = root form, tag = fine-grained POS, pos = coarse_grained POS

nlp = spacy.load('en')
doc = nlp(post.decode('utf-8'))
for word in doc:
    print(word.text, word.lemma_, word.tag_, word.pos_)

for np in doc.noun_chunks:
    print(np.text, np.root.text, np.root.dep_, np.root.head.text)

#visualize spacy results as a nltk tree!

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return nltk.Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

# for pos in t.treepositions():
#         if pos not in leavepos:
#             print(t[pos].label(), len(pos))

# using pattern, good for finding chunks?

p = pattern.parsetree(post, relations=True, lemmata=True)
print repr(p)

for sentence in p:
    for chunk in sentence.chunks:
        print chunk
        print chunk.type, [(w.string, w.type) for w in chunk.words]

print p.sentences
