import ast
import re
import html
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


def get_texts_and_tokenize(df, n_lbls=1, do_tokenize=True):
    labels = df.iloc[:,range(n_lbls)].values
    labels = labels.tolist()
    labels = list(map(lambda x: ast.literal_eval(x[0]), labels))
    texts = f'\n{BOS} ' + df[n_lbls].astype(str)

    # perform common fixup to text:
    texts = list(texts.apply(fixup).values)
    labels = list(labels)

    # tokenize texts
    if do_tokenize:
        tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
        return tok, labels
    return texts, labels


def get_all_tokenized(df, n_lbls, do_tokenize=True):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts_and_tokenize(r, n_lbls, do_tokenize=do_tokenize)
        tok += tok_;
        labels += labels_
    return tok, labels

