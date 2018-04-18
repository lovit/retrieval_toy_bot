class PairCorpus:

    def __init__(self, corpus_path, iter_pair=True, 
                 verbose=False, limit_pairs = -1):
        self.corpus_path = corpus_path
        self.iter_pair = iter_pair
        self.verbose = verbose
        self.n_pairs = 0
        self.limit_pairs = limit_pairs
        
        self._check_points = 1000

    def __len__(self):
        if self.n_pairs == 0:
            with open(self.corpus_path, encoding='utf-8') as f:
                for n_pair, _ in enumerate(f):
                    continue
            self.n_pairs = n_pair + 1

        if self.iter_pair:
            return self.n_pairs
        else:
            return 2 * self.n_pairs

    def __iter__(self):
        with open(self.corpus_path, encoding='utf-8') as f:
            for i_pair, pair in enumerate(f):
                if self.limit_pairs > 0 and i_pair >= self.limit_pairs:
                    break

                if self.verbose and i_pair % self._check_points == 0:
                    print('\ryield from {} / {} pairs'.format(
                        i_pair, self.n_pairs), flush=True, end='')
                
                if self.iter_pair:
                    first, second = pair.split('\t')[:2]
                    yield first, second
                else:
                    for message in pair.split('\t'):
                        yield message

        if self.verbose:
            print('\ryielding was done{}'.format(' '*40), flush=True)