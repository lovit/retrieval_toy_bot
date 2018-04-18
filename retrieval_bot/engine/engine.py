import scipy
from .utils import DefaultMessage

class InstanceBasedRetrievalEngine:

    def __init__(self, vectorizer, send2reply, send_indexer, reply_x,
                 evaluator=None, preprocessor=None, default_message=None):

        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.send2reply = send2reply
        self.send_indexer = send_indexer
        self.reply_x = reply_x
        if type(reply_x) != scipy.sparse.csr.csr_matrix:
            self.reply_x = self.repyl_x.tocsr()
        self.evaluator = evaluator
        self.default_message = default_message if default_message else DefaultMessage()

    def process(self, send, n_reply=5, n_similar_sends=5,
                max_send_distance=0.5, debug=False):

        # Preprocessing
        if self.preprocessor:
            send = preprocessor.process(send)

        # Tokenization & Vectorization
        send_vector = self.vectorizer.transform([send])[0]
        if send_vector.sum() == 0:
            return self.default_message.get_random_message()

        # Get similar sends
        similar_dist, similar_idxs = self.send_indexer.kneighbors(
            send_vector, n_similar_sends, max_send_distance)
        if similar_idxs.shape[0] == 0:
            return self.default_message.get_random_message()

        # Get reply candidates
        candidate_text = {}
        candidate_count = {}
        for similar_idx in similar_idxs:
            # format [(idx, text, count), ... ]
            replies = self.send2reply.get_replies(similar_idx)
            for reply_idx, text, count in replies:
                candidate_text[reply_idx] = text
                candidate_count[reply_idx] = count

        # Return default message if exists no candidates
        if not candidate_count:
            return self.default_message.get_random_message()

        # Evaluating candidates
        if not self.evaluator:
            # sort by frequency
            # format {idx:text, ... }
            replies = sorted(candidate_text.items(),
                             key=lambda x:-candidate_count[x[0]])
            replies = [(reply_idx, text, candidate_count[reply_idx])
                       for reply_idx, text in replies]
        else:
            scores = []
            send_terms = send_vector.nonzero()[1]
            for reply_idx in candidate_text:
                reply_terms = self.reply_x[reply_idx,:].nonzero()[1]
                scores.append((reply_idx, self.evaluator.evaluate(send_terms, reply_terms)))
            replies = sorted(scores, key=lambda x:-x[1])
            replies = [(reply_idx, candidate_text[reply_idx], score)
                       for reply_idx, score in replies]

        if n_reply > 0:
            replies = replies[:n_reply]

        if debug:
            return replies
        return [text for reply_idx, text, score in replies]