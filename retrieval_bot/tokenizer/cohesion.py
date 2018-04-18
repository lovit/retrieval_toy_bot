import numpy as np

class CohesionScore:
    
    def __init__(self, max_length=10, min_count=5, 
                 verbose=True, debug=False):

        self.max_length = max_length
        self.min_count = min_count
        self.verbose = verbose
        self.debug = debug
        self._L = {}

        self._check_points = 1000

    def train_and_scores(self, sentences):
        self.train(sentences)
        return self.scores()
    
    def train(self, sentences):
        for i_sent, sent in enumerate(sentences):
            if self.debug and i_sent >= 1000:
                print('Debug mode stop at 1000 sentences') 
                break

            if self.verbose and i_sent % self._check_points == 0:
                print('\rtraining cohesion {} sents'.format(
                    i_sent), flush=True, end='')

            for token in sent.split():
                for e in range(1, min(len(token), self.max_length)+1):
                    l_part = token[:e]
                    self._L[l_part] = self._L.get(l_part, 0) + 1

        if self.verbose:
            print('\rtraining cohesion was done. {} sents'.format(
                i_sent+1), flush=True)

        return self

    def scores(self):
        scores = {word:pow(count / self._L[word[0]], 1/(len(word)-1)) 
                  for word, count in self._L.items() 
                  if len(word) >= 2 and count >= self.min_count}
        return scores

    def score(self, word):
        if not word or len(word) <= 1:
            return 0
        
        base_freq = self._L.get(word[0])
        if not base_freq:
            return 0
        
        word_freq = self._L.get(word, 0)
        return pow(word_freq / base_freq, 1/(len(word)-1))

    def frequency(self, l_word):
        return self._L.get(l_word, 0)
    
    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            for l_part, count in self._L.items():
                f.write('{}\t{}\n'.format(l_part, count))

    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            for i_line, line in enumerate(f):
                try:
                    l_part, count = line.strip().split('\t')
                    self._L[l_part] = int(count)
                except Exception as e:
                    print('{} at line ({}), {}'.format(e, i_line+1, line.strip()))
                    break