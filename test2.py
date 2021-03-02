from nltk.corpus import wordnet as wn

#tha
tha_lemmas = [x for x in wn.all_lemma_names(lang='tha')]
t = wn.synsets(tha_lemmas[0], lang='tha')
t[0].definition()

# eng
lemmas_in_words = list(set(i for i in wn.words()))
t = wn.synsets(lemmas_in_words[0])
t[0].definition()

for i, ss in enumerate(wn.all_synsets()):
    names = ss.lemma_names()
    definition = ss.definition()
