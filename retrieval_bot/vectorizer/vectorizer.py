from collections import Counter
from scipy.sparse import csr_matrix

class Vectorizer:

    def __init__(self, tokenizer=lambda x:x.split(), min_tf=0,
        max_tf=99999999, min_df=0, max_df=1.0, stopwords=None,
        lowercase=True, verbose=True):

        assert 0 <= min_df < 1
        assert 0 < max_df <= 1

        self.tokenizer = tokenizer
        self.min_tf = min_tf
        self.max_tf = max_tf
        self.min_df = min_df
        self.max_df = max_df
        self.stopwords = stopwords if stopwords else {}
        self.lowercase = lowercase
        self.verbose = verbose

        self._check_points = 500

        self.vocabulary_ = {}
        self.idx2vocab = []
        self.n_vocabs = 0
    
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
    
    def fit(self, docs):
        # counting
        df = {}
        tf = {}

        for i_doc, doc in enumerate(docs):
            if self.verbose and i_doc % self._check_points == 0:
                print('\rscanned {} docs'.format(i_doc), flush=True, end='')

            counter = Counter((token for token in self.tokenizer(doc)))
            for term, freq in counter.items():
                df[term] = df.get(term, 0) + 1
                tf[term] = tf.get(term, 0) + 1

        if self.verbose:
            print('\rscanning was done{}'.format(' '*40), flush=True)

        # filtering
        n_docs = i_doc + 1
        min_df = int(n_docs * self.min_df)
        max_df = int(n_docs * self.max_df)
        df = {term:df_t for term, df_t in df.items() if min_df <= df_t <= max_df}
        tf = {term:tf_t for term, tf_t in tf.items() if self.min_tf <= tf_t <= self.max_tf}

        # build vocabulary_
        vocabs = {term:tf_t for term, tf_t in tf.items() if term in df}
        self.vocabulary_ = {term:idx for idx, (term, _) in enumerate(
            sorted(vocabs.items(), key=lambda x:-x[1]))}
        self.idx2vocab = [term for term, _ in sorted(vocabs.items(), key=lambda x:x[1])]
        self.n_vocabs = len(self.idx2vocab)
        
        if self.verbose:
            print('{} terms are recognized'.format(self.n_vocabs), flush=True)

        return self

    def __len__(self):
        return self.n_vocabs

    def encode_a_doc_to_list(self, doc):
        return [self.vocabulary_[term] for term in self.tokenizer(doc) if term in self.vocabulary_]

    def decode_from_list(self, doc):
        return [self.idx2vocab[idx] for idx in doc if 0 <= idx < self.n_vocabs]

    def encode_a_doc_to_bow(self, doc):
        bow = Counter(self.tokenizer(doc))
        bow = {self.vocabulary_[term]:count for term, count in bow.items() if term in self.vocabulary_}
        return bow

    def decode_from_bow(self, bow):
        bow = {self.idx2vocab[idx]:count for idx, count in bow.items() if 0 <= idx < self.n_vocabs}
        return bow

    def transform(self, docs):
        rows = []
        cols = []
        data = []
        for i_doc, doc in enumerate(docs):
            if self.verbose and i_doc % self._check_points == 0:
                print('\rtransformed {} docs'.format(i_doc), flush=True, end='')

            bow = self.encode_a_doc_to_bow(doc)
            for term, count in bow.items():
                rows.append(i_doc)
                cols.append(term)
                data.append(count)

        if self.verbose:
            print('\rtransforming docs to term frequency marix was done', flush=True)

        return csr_matrix((data, (rows, cols)), shape=(i_doc+1, self.n_vocabs))

    def save(self, fname):
        if fname[-6:] != '.vocab':
            fname += '.vocab'
        with open(fname, 'w', encoding='utf-8') as f:
            for vocab in self.idx2vocab:
                f.write('{}\n'.format(vocab))
    
    def load(self, fname):
        if fname[-6:] != '.vocab':
            fname += '.vocab'
        with open(fname, encoding='utf-8') as f:
            self.idx2vocab = [term.strip() for term in f]
        self.vocabulary_ = {term:idx for idx, term in enumerate(self.idx2vocab)}
        self.n_vocabs = len(self.idx2vocab)

    def vocabs(self):
        return [term for term in sorted(self.vocabulary_, key=lambda x:self.vocabulary_[x])]