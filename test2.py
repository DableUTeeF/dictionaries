from nltk.corpus import wordnet as wn


def syn(word, lch_threshold=2.26):
    for net1 in wn.synsets(word):
        for net2 in wn.all_synsets():
            try:
                lch = net1.lch_similarity(net2)
            except:
                continue
            # The value to compare the LCH to was found empirically.
            # (The value is very application dependent. Experiment!)
            if lch >= lch_threshold:
                yield net1, net2, lch

if __name__ == '__main__':
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
