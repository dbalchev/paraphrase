Paraphrase identification in Python 3 using scikit-learn
===

A solution to the [Semeval Paraphrase identification task](http://alt.qcri.org/semeval2015/task1/)

It's a reimplementation and extension of [ASOBEK](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval011.pdf).

Currently only adds as a feature, the dot product of the sums of [word2vec](https://code.google.com/p/word2vec/) vectors
from both tweets and replaces SVC with AdaBoost-ed decision trees.  

Currently the performance of the method is unstable, sometimes yielding an F1
score of 0.6903 (beating ASOBEK's 0.674) and sometimes as low as 0.63.

The word2vec database is  "pre-trained vectors trained on part of Google News"
from the word2vec website.
