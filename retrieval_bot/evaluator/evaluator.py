import scipy
from scipy.io import mmread

class TermPairEvaluator:

    def __init__(self, score):
        if type(score) == str:
            self._x = mmread(score).todok()
        else:
            assert type(score) == scipy.sparse.dok.dok_matrix
            self._x = score

    def evaluate(self, send_terms, reply_terms, topk=4):
        scores = [(si, ri, self._x[si,ri]) for si in send_terms for ri in reply_terms]
        scores = [s for s in scores if s[2] > 0]
        if not scores:
            return 0
        
        scores = sorted(scores, key=lambda x:-x[2])
        if topk > 0:
            scores = scores[:topk]
        return sum([s[2] for s in scores]) / len(scores)