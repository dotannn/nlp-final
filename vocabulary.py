from collections import defaultdict, Counter


class Vocabulary:
    def __init__(self, idx_to_token):
        self._idx_to_token = idx_to_token
        self._token_to_idx = defaultdict( lambda: 0, {v: k for k, v in enumerate( idx_to_token )} )
        self._n_tokens = len( idx_to_token )

    @staticmethod
    def from_text(texts, max_vocab=60000):
        freq = Counter(p for o in texts for p in o )

        idx_to_token = [o for o, c in freq.most_common( max_vocab )]
        idx_to_token.insert( 0, '_pad_' )
        idx_to_token.insert( 0, '_unk_' )

        return Vocabulary(idx_to_token)

    @property
    def size(self):
        return self._n_tokens

    def numericalize(self, text):
        return np.array( [[self._token_to_idx[o] for o in p] for p in text] )
