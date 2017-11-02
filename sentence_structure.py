import nltk
import re
import spacy

grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
  """)

sentence = "I shot an elephant in my pajamas"
sent = sentence.split()
parser = nltk.ChartParser(grammar)
trees = parser.parse(sent)
for tree in trees:
    print tree

nlp = spacy.load('en')
doc = nlp(sentence.decode('utf-8'))
for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

for np in doc.noun_chunks:
    print(np.text, np.root.text, np.root.dep_, np.root.head.text)
