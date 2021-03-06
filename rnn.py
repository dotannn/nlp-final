from fastai.text import LanguageModelLoader, LanguageModelData, accuracy, TextDataset, SortSampler, SortishSampler, \
    ModelData, TextModel, RNN_Learner, to_gpu, DataLoader, get_rnn_classifier, LoggingCallback, accuracy_multi
from fastai.lm_rnn import *
from torch.nn.functional import binary_cross_entropy, sigmoid

from metrics import jaccard_index, precision, recall, f1
from utils import *
from genre_classifier import GenreClassifier


class RNNGenreClassifier(GenreClassifier):
    THRESH = 0.5
    OPT_FN = partial( optim.Adam,
                      betas=(0.75, 0.99) )  # defaults for Adam dont work well for NLP so we change to this number...

    def __init__(self, embedding_size=250, n_hidden_activations=640, n_layers=3, drop_mul_lm=0.8, drop_mul_classifier=0.6,
                 bptt=70, wd=1e-7, n_classes=31, vocab=None, batch_size=128):#0.7):
        super( RNNGenreClassifier, self ).__init__(n_classes)

        self._vocab = vocab
        self._dropouts_lm = np.array( [0.25, 0.1, 0.2, 0.02, 0.15] ) * drop_mul_lm
        self._dropouts_classifier = np.array( [0.4, 0.5, 0.05, 0.3, 0.4] ) * drop_mul_classifier

        self._embedding_size = embedding_size
        self._n_hidden_activations = n_hidden_activations
        self._n_layers = n_layers
        self._bptt = bptt
        self._wd = wd
        self._batch_size = batch_size

    @staticmethod
    def func_metric(preds, gts, thresh=0.5, func=jaccard_index):
        if len( preds ) != len( gts ):
            raise RuntimeError(
                "predicted and gt lists must have same size! predicted = %d, gt = %d" % (len( preds ), len( gts )) )
        def _process(p, g):
            g = np.where( g > thresh )[0].tolist()
            p = np.where( sigmoid( p ) > thresh )[0].tolist()
            return func(p, g)

        return np.array( [_process(p, g) for p, g in zip( preds, gts )] ).mean()

    def _train_lm(self, train_ids, batch_size=4, val_ids=None):
        train_dataloader = LanguageModelLoader( np.concatenate( train_ids ), batch_size, self._bptt )
        val_dataloader = LanguageModelLoader( np.concatenate( val_ids ), batch_size, self._bptt )

        md = LanguageModelData( "tmp", 1, self._vocab.size, train_dataloader, val_dataloader, bs=batch_size, bptt=self._bptt )


        self._language_model = md.get_model( self.OPT_FN, self._embedding_size, self._n_hidden_activations, self._n_layers,
                                       dropouti=self._dropouts_lm[0], dropout=self._dropouts_lm[1], wdrop=self._dropouts_lm[2],
                                       dropoute=self._dropouts_lm[3], dropouth=self._dropouts_lm[4] )

        self._language_model.metrics = [accuracy]
        self._language_model.unfreeze()

        lr = 1e-3
        self._language_model.lr_find(start_lr=lr/10, end_lr=lr*50, linear=True)
        self._language_model.fit(lr / 2, 1, wds=self._wd, use_clr=(32,2), cycle_len=1, callbacks=[LoggingCallback(save_path="./tmp/log")])

        self._language_model.lr_find( start_lr=lr / 10, end_lr=lr * 10, linear=True )

        self._language_model.fit( lr, 1, wds=self._wd, use_clr=(32, 2), cycle_len=20, callbacks=[LoggingCallback(save_path="./tmp/log")] )

        self._language_model.save_encoder("enc_weights")

    def _train_classifier(self, train_ids, train_labels, batch_size=4, val_ids=None, val_labels=None):
        # change from multi-label to multi-class:

        def one_hot_idxs(idxs, n_classes):
            res = np.zeros( n_classes )
            res[idxs] = 1.
            return res

        onehot_train_labels = np.array( [one_hot_idxs(l, self._n_classes) for l in train_labels])
        onehot_val_labels = np.array( [one_hot_idxs(l, self._n_classes) for l in val_labels] )

        train_ds = TextDataset( train_ids, onehot_train_labels)
        val_ds = TextDataset( val_ids, onehot_val_labels )

        train_sampler = SortishSampler( train_ids, key=lambda x: len( train_ids[x] ), bs=batch_size)
        val_sampler = SortSampler( val_ids, key=lambda x: len( val_ids[x] ) )

        train_dl = DataLoader( train_ds, batch_size, num_workers=1, transpose=True, pad_idx=1, sampler=train_sampler )
        val_dl = DataLoader( val_ds, batch_size, num_workers=1, transpose=True, pad_idx=1, sampler=val_sampler )

        md = ModelData( "tmp", train_dl, val_dl )

        m = get_rnn_classifier( self._bptt, 20 * 70, self._n_classes, self._vocab.size, emb_sz=self._embedding_size,
                                n_hid=self._n_hidden_activations, n_layers=self._n_layers, pad_token=1,
                                layers=[self._embedding_size* 3, 128, self._n_classes],
                                drops=[self._dropouts_classifier[4], 0.1],
                                dropouti=self._dropouts_classifier[0], wdrop=self._dropouts_classifier[1],
                                dropoute=self._dropouts_classifier[2], dropouth=self._dropouts_classifier[3] )

        self._classifier_model = RNN_Learner( md, TextModel( to_gpu( m ) ), opt_fn=self.OPT_FN )
        self._classifier_model.reg_fn = partial( seq2seq_reg, alpha=2, beta=1 )
        self._classifier_model.clip = 25.  # or 0.3 ?!

        def binary_ce_wrapper(predicted, gt):
            out = F.sigmoid(predicted)
            return binary_cross_entropy(out, gt)

        self._classifier_model.crit = binary_ce_wrapper
        jaccard_0_5 = partial( self.func_metric, func=jaccard_index)
        jaccard_0_5.__name__ = "jaccard_0_5"
        precision_0_5 = partial( self.func_metric, func=precision )
        precision_0_5.__name__ = "precision_0_5"
        recall_0_5 = partial( self.func_metric, func=recall)
        recall_0_5.__name__ = "recall_0_5"
        f1_0_5 = partial( self.func_metric, func=f1 )
        f1_0_5.__name__ = "f1_0_5"

        self._classifier_model.metrics = [jaccard_0_5, precision_0_5, recall_0_5, f1_0_5]

        lr = 3e-3
        lrm = 2.6
        lrs = np.array( [lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr] )

        self._classifier_model.load_encoder( 'enc_weights' )

        self._classifier_model.freeze_to( -1 )
        self._classifier_model.fit( lrs, 1, cycle_len=1, use_clr=(8, 3), callbacks=[LoggingCallback(save_path="./tmp/log")] )
        self._classifier_model.freeze_to( -2 )
        self._classifier_model.fit( lrs, 1, cycle_len=1, use_clr=(8, 3), callbacks=[LoggingCallback(save_path="./tmp/log")] )
        self._classifier_model.unfreeze()
        self._classifier_model.fit( lrs, 1, cycle_len=24, use_clr=(32, 10),callbacks=[LoggingCallback(save_path="./tmp/log")] )

        self._classifier_model.save( 'classifier_weights' )

    def train(self, train_data, train_labels, val_data=None, val_labels=None):
        train_ids = self._vocab.numericalize( train_data )
        val_ids = self._vocab.numericalize(val_data)
        self._train_lm(train_ids, batch_size=self._batch_size, val_ids=val_ids)

        self._train_classifier(train_ids, train_labels, batch_size=self._batch_size, val_ids=val_ids,
                               val_labels=val_labels)

    def predict_lm(self, tokens):
        ids = self._vocab.numericalize(tokens)
        self._language_model.predict_array(ids)

    def predict(self, summaries_tokens):
        summaries_ids = self._vocab.numericalize(summaries_tokens)
        pp = []

        for x in summaries_ids:
            x = np.array(x)
            x = np.expand_dims(x, 1)
            res = self._classifier_model.predict_array(x)[0]
            p = np.apply_along_axis(np_sigmoid, 0, res)
            p = np.where( p > self.THRESH )[0].tolist()
            pp.append(p)
        return pp

