#!/usr/bin/env bash

#######get models
pwd
if [ ! -d "./models" ]
then
    echo "make models directory"
    mkdir ./models
fi

cd ./models/
#1. context2vec
if [ ! -d ./context2vec ]
then
    git clone https://github.com/qianchu/context2vec.git
fi

if [ ! -d "./context2vec/model_dir" ]
    then
        echo "make context2vec model"
        mkdir ./context2vec/model_dir
    fi

#load test context2vec model
if [ ! -f ./context2vec/model_dir/MODEL-wiki.params.14 ]
then
     scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/context2vec/model_dir/MODEL-wiki.params.14 ./context2vec/model_dir/
fi

if [ ! -f ./context2vec/model_dir/WORDS-wiki.targets.14 ]
then
     scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/context2vec/model_dir/WORDS-wiki.targets.14 ./context2vec/model_dir/
fi

if [ ! -f ./context2vec/model_dir/MODEL-wiki.14  ]
then
      scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/context2vec/model_dir/MODEL-wiki.14 ./context2vec/model_dir/
fi

# add context2vec as package
cd context2vec/
python ./setup.py install

cd ../

#2. load test skipgram model
if [ ! -d ./wiki_all.model ]
then
    mkdir ./wiki_all.model/
fi

if [ ! -f ./wiki_all.model/wiki_all.sent.split.model ]
then
        scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/wiki_all.model/wiki_all.sent.split.model ./wiki_all.model/wiki_all.sent.split.model
        scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/wiki_all.model/wiki_all.sent.split.model.wv.syn0.npy ./wiki_all.model/
        scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/models/wiki_all.model/wiki_all.sent.split.model.syn1neg.npy ./wiki_all.model/
fi


# 3. load test alacarte
if [ ! -d ./ALaCarte ]
then
     git clone  https://github.com/qianchu/ALaCarte.git

fi

cd ..

# 4. load w2saliance_f
if [ ! -d ./corpora/corpora ]
then
    mkdir ./corpora/corpora

fi

if [ ! -f ./corpora/corpora/WWC_norarew.txt.tokenized.vocab ]
then
    scp ql261@mml0223.rceal.private.cam.ac.uk:/home/ql261/rare_we/corpora/corpora/WWC_norarew.txt.tokenized.vocab ./corpora/corpora/
fi
