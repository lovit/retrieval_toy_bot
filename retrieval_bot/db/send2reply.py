from collections import defaultdict
import pickle

class Send2Reply:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.s2r = {}
        self.sends = []
        self.replies = []
        
        self._check_points = 1000

    def __len__(self):
        return len(self.sends)

    def train(self, pairs):
        unique_send = defaultdict(lambda: len(unique_send))
        unique_reply = defaultdict(lambda: len(unique_reply))
        s2r_ = defaultdict(lambda: defaultdict(lambda: 0))

        for i_pair, (send, reply) in enumerate(pairs):
            if self.verbose and i_pair % self._check_points == 0:
                print('\rtraining send2reply graph {} pairs'.format(
                    i_pair), flush=True, end='')

            send = send.strip()
            reply = reply.strip()
            send_idx = unique_send[send]
            reply_idx = unique_reply[reply]
            s2r_[send_idx][reply_idx] += 1

        if self.verbose:
            print('\rtraining send2reply graph was done. {} pairs'.format(
                i_pair), flush=True)

        for s, rdict in s2r_.items():
            self.s2r[s] = dict(rdict)

        for send, _ in sorted(unique_send.items(), key=lambda x:x[1]):
            self.sends.append(send)

        for reply, _ in sorted(unique_reply.items(), key=lambda x:x[1]):
            self.replies.append(reply)
    
    def get_replies(self, send_idx, n_replies=-1):
        """It return [(idx, text, count), ...]"""

        replies = self.s2r.get(send_idx, {})
        if not replies:
            return []

        replies = sorted(replies.items(), key=lambda x:-x[1])
        if n_replies > 0:
            replies = replies[:n_replies]

        replies = [(reply[0], self.replies[reply[0]], reply[1]) for reply in replies]
        return replies

    def save(self, fname):
        params = {
            's2r': self.s2r, 
            'sends': self.sends,
            'replies': self.replies
        }
        with open(fname, 'wb') as f:
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        self.s2r = params['s2r']
        self.sends = params['sends']
        self.replies = params['replies']