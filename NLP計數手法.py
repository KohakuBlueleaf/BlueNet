from functions import *
from txt import txt
corpus, word_to_id, id_to_word=preprocess(txt)
C = Co_Matrix(corpus,len(word_to_id))
M = ppmi(C,verbose = True)
most_similar(input(), word_to_id, id_to_word, M)
