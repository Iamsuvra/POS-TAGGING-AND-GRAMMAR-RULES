 import nltk
>>> text1="We had a nice party yesterday"
>>> text2="She came to visit me two days ago"
>>> text3="You may go now"
>>> text4="Their kids are not always naive"
>>> textsplit1 = nltk.sent_tokenize(text1)
>>> textsplit1
['We had a nice party yesterday']
>>> tokentext1 = [nltk.word_tokenize(sent) for sent in textsplit1]
>>> tokentext1
[['We', 'had', 'a', 'nice', 'party', 'yesterday']]
>>> t0 = nltk.DefaultTagger('NN')
>>> t1 = nltk.UnigramTagger(treebank_train, backoff=t0)

>>> treebank_tagged = treebank.tagged_sents()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'treebank' is not defined
>>> from nltk.corpus import treebank
>>> treebank_tagged = treebank.tagged_sents()
>>> size = int(len(treebank_tagged) * 0.9)
>>> treebank_train = treebank_tagged[:size]
>>> treebank_test = treebank_tagged[size:]
>>> t1 = nltk.UnigramTagger(treebank_train, backoff=t0)
>>> t2 = nltk.BigramTagger(treebank_train, backoff=t1)
>>> t2.evaluate(treebank_test)
0.8905852417302799
>>> taggedtext1 = [t2.tag(tokens) for tokens in tokentext1]
>>> taggedtext1
[[('We', 'PRP'), ('had', 'VBD'), ('a', 'DT'), ('nice', 'JJ'), ('party', 'NN'), ('yesterday', 'NN')]]
>>> textsplit2 = nltk.sent_tokenize(text2)
>>> tokentext2 = [nltk.word_tokenize(sent) for sent in textsplit2]
>>> taggedtext = [t2.tag(tokens) for tokens in tokentext]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'tokentext' is not defined
>>> taggedtext2 = [t2.tag(tokens) for tokens in tokentext2]
>>> taggedtext2
[[('She', 'PRP'), ('came', 'VBD'), ('to', 'TO'), ('visit', 'NN'), ('me', 'PRP'), ('two', 'CD'), ('days', 'NNS'), ('ago', 'IN')]]
>>> textsplit3= nltk.sent_tokenize(text3)

>>> tokentext3= [nltk.word_tokenize(sent) for sent in textsplit3]
>>> taggedtext3= [t2.tag(tokens) for tokens in tokentext3]
>>> taggedtext3
[[('You', 'PRP'), ('may', 'MD'), ('go', 'VB'), ('now', 'RB')]]
>>> textsplit4= nltk.sent_tokenize(text4)
>>> tokentext4= [nltk.word_tokenize(sent) for sent in textsplit4]
>>> taggedtext4= [t2.tag(tokens) for tokens in tokentext4]
>>> taggedtext4
[[('Their', 'PRP$'), ('kids', 'NNS'), ('are', 'VBP'), ('not', 'RB'), ('always', 'RB'), ('naive', 'NN')]]

>>> groucho_grammar = nltk.CFG.fromstring("""
...  S -> NP VP
...  NP -> Prop | Det VP | Det ADJ NP | V NP | Prop Det NP | N Prop
...  VP -> V VP |  V AP | Prop VP | V NP
...  AP -> ADV AP | ADJ| ADV
...  ADV ->  'now' | 'not'| 'always'
...  ADJ -> 'nice' | 'naive'
...  Det -> 'to' | 'last' | 'a' | 'two'
...  N -> 'party' | 'days'
...  V -> 'came' | 'had' | 'may' | 'go' | 'visit' | 'are'
...  Prop -> 'We' | 'She' | 'You' | 'kids' | 'yesterday'| 'me'| 'Their'| 'ago'
...  """)
>>> rd_parser = nltk.RecursiveDescentParser(groucho_grammar)
>>> trees = rd_parser.parse(sentlist4)

>>> treelist4= list(trees)
>>> for tree in treelist4:
...     print(tree)
...
(S
  (NP (Prop Their))
  (VP
    (Prop kids)
    (VP (V are) (AP (ADV not) (AP (ADV always) (AP (ADJ naive)))))))

        (NP (Prop me) (Det two) (NP (N days) (Prop ago)))))))
>>> senttext5="She may go now"
>>> senttext6="You had a nice party yesterday"
>>> senttext6="You came to visit me two days ago"
>>> sentlist5=senttext5.split()
>>> rd_parser = nltk.RecursiveDescentParser(groucho_grammar)
>>> trees5= rd_parser.parse(sentlist5)
>>> treelist5=list(trees5)
>>> for tree in treelist5:
...     print(tree)
...
(S (NP (Prop She)) (VP (V may) (VP (V go) (AP (ADV now)))))
>>> sentlist6=senttext6.split()
>>> trees6= rd_parser.parse(sentlist6)

>>> treelist6=list(trees6)
>>> for tree in treelist6:
...     print(tree)
...
(S
  (NP (Prop You))
  (VP
    (V came)
    (NP
      (Det to)
      (VP
        (V visit)
        (NP (Prop me) (Det two) (NP (N days) (Prop ago)))))))
>>> grammar = PCFG.fromstring("""
...  S -> NP VP [1.0]
...
...  NP -> Prop [0.45] | Det VP[0.11] | Det ADJ NP[0.11] | Prop Det NP[0.11] | N Prop[0.22]
...
... VP -> V VP [0.15]|  V AP [0.28]| Prop VP [0.15] | V NP [0.42]
...
...  AP -> ADV AP [0.34] | ADJ [0.33] | ADV [0.33]
...
...  ADV  ->    'not' [0.34] | 'always' [0.33] | 'now' [0.33]
...
...  ADJ ->  'nice' [0.50] | 'naive' [0.50]
...
...  Det ->  'to' [0.25] | 'last' [0.25] | 'a' [0.25] | 'two' [0.25]
...
...  N ->  'party' [0.50] | 'days' [0.50]
...
...  V ->  'came' [0.16] | 'had' [0.16] | 'may' [0.17] | 'go' [0.16] | 'visit' [0.17] | 'are' [0.17]
...
...  Prop ->  'We' [0.125] | 'She' [0.125] | 'You' [0.125] | 'kids' [0.125] | 'yesterday' [0.125] | 'me' [0.125] | 'Their' [0.125] | 'ago' [0.125]
...  """)
>>> rd_parser = nltk.RecursiveDescentParser(grammar)
>>> for tree in treelist:
...     print(tree)
...
(S
  (NP (Prop She))
  (VP
    (V came)
    (NP
      (Det to)
      (VP
        (V visit)
        (NP (Prop me) (Det two) (NP (N days) (Prop ago)))))))

...     print(tree)
...
(S (NP (Prop You)) (VP (V may) (VP (V go) (AP (ADV now)))))
>>> for tree in treelist1:
...     print(tree)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'treelist1' is not defined
>>> for tree in treelist4:
...     print(tree)
...
(S
  (NP (Prop Their))
  (VP
    (Prop kids)
    (VP (V are) (AP (ADV not) (AP (ADV always) (AP (ADJ naive)))))))
>>> for tree in treelist:
...     print(tree)
...
(S
  (NP (Prop She))
  (VP
    (V came)
    (NP
      (Det to)
      (VP
        (V visit)
        (NP (Prop me) (Det two) (NP (N days) (Prop ago)))))))