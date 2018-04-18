from .utils import DefaultMessage

class InstanceBasedRetrievalEngine:

    def __init__(self, vectorizer, send2reply, send_indexer,
                 scoring=None, preprocessor=None, default_message=None):

        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.send2reply = send2reply
        self.send_indexer = send_indexer
        self.scoring = scoring
        self.default_message = default_message if default_message else DefaultMessage()

    def process(self, send, n_reply=5, n_similar_sends=5,
                max_send_distance=0.5):

        # Preprocessing
        if self.preprocessor:
            send = preprocessor.process(send)

        # Tokenization & Vectorization
        send_vector = self.vectorizer.transform([send])[0]
        if send_vector.sum() == 0:
            return self.default_message.get_random_message()

        # Get similar sends
        send_dist, send_idxs = self.send_indexer.kneighbors(
            send_vector, n_similar_sends, max_send_distance)
        if send_idxs.shape[0] == 0:
            return self.default_message.get_random_message()

        # Get reply candidates
        candidate_text = {}
        candidate_count = {}
        for send_idx in send_idxs:
            # format [(idx, text, count), ... ]
            replies = self.send2reply.get_replies(send_idx)
            for idx, text, count in replies:
                candidate_text[idx] = text
                candidate_count[idx] = count

        # Scoring
        if not candidate_count:
            return self.default_message.get_random_message()

        # Sorting
        # format {idx:text, ... }
        replies = sorted(candidate_text.items(), 
                         key=lambda x:-candidate_count[x[0]])

        if n_reply > 0:
            replies = replies[:n_reply]

        return [text for idx, text in replies]
        