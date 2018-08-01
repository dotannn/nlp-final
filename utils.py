import ast
import re
import html
import math
from fastai.text import Tokenizer, partition_by_cores

from vocabulary import Vocabulary

BOS = 'xbos' # begining of string

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts_and_tokenize(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values
    labels = labels.tolist()
    labels = list(map(lambda x: ast.literal_eval(x[0]), labels))
    texts = f'\n{BOS} ' + df[n_lbls].astype(str)

    # perform common fixup to text:
    texts = list(texts.apply(fixup).values)
    labels = list(labels)

    # tokenize texts
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return texts, tok, labels


def get_all_tokenized(df, n_lbls):
    texts, tok, labels = [], [], []
    for i, r in enumerate(df):
        print(i)
        text_, tok_, labels_ = get_texts_and_tokenize(r, n_lbls)
        texts += text_
        tok += tok_
        labels += labels_
    return texts, tok, labels


def np_sigmoid(x):
    return 1. / (1. + math.exp( -x ))

