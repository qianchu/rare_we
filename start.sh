#!/usr/bin/env bash


#environment config
ENV_NAME="rare_we"
if conda env list | grep -q $ENV_NAME
then
    echo $ENV_NAME found
else
    conda env create -f environment.yml
fi

activate $ENV_NAME
conda env export > environment.yml

#git config
git config --global user.email "hey_flora@126.com"
git config --global user.name "qianchu"

# pack python project
cd ./models/context2vec
python ./setup.py install
