from fastai.text import LanguageModelLoader, LanguageModelData, accuracy, TextDataset, SortSampler, SortishSampler, \
    ModelData, TextModel, RNN_Learner, to_gpu, DataLoader, get_rnn_classifier
from fastai.lm_rnn import *

from utils import *
from classifier import GenreClassifier


class RNNGenreClassifier(GenreClassifier):
    OPT_FN = partial( optim.Adam,
                      betas=(0.75, 0.99) )  # defaults for Adam dont work well for NLP so we change to this number...

    def __init__(self, embedding_size=200, n_hidden_activations=500, n_layers=3, drop_mul_lm=0.7, drop_mul_classifier=0.5,
                 bptt=70, wd=1e-7, n_classes=31, vocab=None):#0.7):
        self._n_classes = n_classes
        self._vocab = vocab
        self._dropouts_lm = np.array( [0.25, 0.1, 0.2, 0.02, 0.15] ) * drop_mul_lm
        self._dropouts_classifier = np.array( [0.4, 0.5, 0.05, 0.3, 0.4] ) * drop_mul_classifier

        self._embedding_size = embedding_size
        self._n_hidden_activations = n_hidden_activations
        self._n_layers = n_layers
        self._bptt = bptt
        self._wd = wd
        pass

    def _train_lm(self, train_ids, batch_size=4, val_ids=None):
        train_dataloader = LanguageModelLoader( np.concatenate( train_ids ), batch_size, self._bptt )
        val_dataloader = LanguageModelLoader( np.concatenate( val_ids ), batch_size, self._bptt )

        md = LanguageModelData( "tmp", 1, self._vocab.size, train_dataloader, val_dataloader, bs=batch_size, bptt=self._bptt )


        language_model = md.get_model( self.OPT_FN, self._embedding_size, self._n_hidden_activations, self._n_layers,
                                       dropouti=self._dropouts_lm[0], dropout=self._dropouts_lm[1], wdrop=self._dropouts_lm[2],
                                       dropoute=self._dropouts_lm[3], dropouth=self._dropouts_lm[4] )

        language_model.metrics = [accuracy]
        language_model.unfreeze()

        lr = 1e-3
        # language_model.lr_find(start_lr=lr/10, end_lr=lr*50, linear=True)
        # lr = language_model.sched.lrs[np.argmin(language_model.sched.losses)]
        language_model.fit(lr / 2, 1, wds=self._wd, use_clr=(32,2), cycle_len=1)

        # language_model.lr_find( start_lr=lr / 10, end_lr=lr * 10, linear=True )

        # lr = language_model.sched.lrs[np.argmin( language_model.sched.losses )]

        language_model.fit( lr, 1, wds=self._wd, use_clr=(32, 2), cycle_len=15 )

        language_model.save_encoder("enc_weights")

    def _train_classifier(self, train_ids, train_labels, batch_size=4, val_ids=None, val_labels=None):
        # change from multi-label to multi-class:
        reduced_train_labels = np.array( [l[0] for l in train_labels] )
        reduced_val_labels = np.array( [l[0] for l in val_labels] )

        train_ds = TextDataset( train_ids, reduced_train_labels)
        val_ds = TextDataset( val_ids, reduced_val_labels )

        train_sampler = SortishSampler( train_ids, key=lambda x: len( train_ids[x] ), bs=batch_size) # TODO understand why bs// 2
        val_sampler = SortSampler( val_ids, key=lambda x: len( val_ids[x] ) )

        train_dl = DataLoader( train_ds, batch_size, num_workers=1, transpose=True, pad_idx=1, sampler=train_sampler ) # TODO understand why bs// 2
        val_dl = DataLoader( val_ds, batch_size, num_workers=1, transpose=True, pad_idx=1, sampler=val_sampler )

        md = ModelData( "tmp", train_dl, val_dl )

        m = get_rnn_classifier( self._bptt, 20 * 70, self._n_classes, self._vocab.size, emb_sz=self._embedding_size,
                                n_hid=self._n_hidden_activations, n_layers=self._n_layers, pad_token=1,
                                layers=[self._embedding_size* 3, 128, self._n_classes],
                                drops=[self._dropouts_classifier[4], 0.1],
                                dropouti=self._dropouts_classifier[0], wdrop=self._dropouts_classifier[1],
                                dropoute=self._dropouts_classifier[2], dropouth=self._dropouts_classifier[3] )

        classifier_model = RNN_Learner( md, TextModel( to_gpu( m ) ), opt_fn=self.OPT_FN )
        classifier_model.reg_fn = partial( seq2seq_reg, alpha=2, beta=1 )
        classifier_model.clip = 25.  # or 0.3 ?!
        classifier_model.metrics = [accuracy]  # total_jaccard]

        lr = 3e-3
        lrm = 2.6
        lrs = np.array( [lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr] )
        # lrs = np.array( [1e-4, 1e-4, 1e-4, 1e-3, 1e-2] )

        wd = 1e-6
        classifier_model.load_encoder( 'enc_weights' )

        classifier_model.freeze_to( -1 )
        # TODO: should we use wds?
        classifier_model.fit( lrs, 1, cycle_len=1, use_clr=(8, 3) )
        classifier_model.freeze_to( -2 )
        classifier_model.fit( lrs, 1, cycle_len=1, use_clr=(8, 3) )
        classifier_model.unfreeze()
        classifier_model.fit( lrs, 1, cycle_len=14, use_clr=(32, 10) )

        classifier_model.save( 'classifier_weights' )

        pass

    def fit(self, train_ids, train_labels, batch_size=4, val_ids=None, val_labels=None):
        self._train_lm(train_ids, batch_size=batch_size, val_ids=val_ids)

        self._train_classifier(train_ids, train_labels, batch_size=batch_size, val_ids=val_ids,
                               val_labels=val_labels)

    def predict_lm(self):
        pass

    def predict(self):
        pass