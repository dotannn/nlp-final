# Predicting movie genres using plot summary

## how to run this code:

### Install fastai env:
```bash
git clone https://github.com/fastai/fastai.git
cd fastai
conda env update
source activate fastai
python setup.py install
```

### Install other dependencies:
```sh
pip install pandas tabulate spacy nltk
python -m spacy download en
```

### Get project’s code: 
```bash

git clone https://github.com/dotannn/nlp-final.git
cd nlp-final
```

### Download the data:
```sh
mkdir data
cd data
wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/plot.list.gz
wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/genres.list.gz
gzip -d plot.list.gz
cd ..
```

### Run the code:
```bash
python main.py -bs 64 -nc
```
The code will run training process of best hyper-parameters configurations of our model as well as the baseline model and return the results in a table:

```console
Name                     Precision    Recall    F-score    Jaccard
---------------------  -----------  --------  ---------  ---------
Naive-bayes(baseline)        0.543     0.355      0.405      0.355
Ours[250, 720]               0.723     0.642      0.638      0.557
```

## recommended env:
We ran this project on AWS p3.8xlarge servers. with Deep Learning AMI (Amazon Linux) Version 11.0 (ami-ca4464b5) image. the training process takes about 6-7 hours.

