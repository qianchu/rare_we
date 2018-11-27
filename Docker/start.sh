#!/usr/bin/env bash


#environment config
ENV_NAME="rare_we_clean"
if conda env list | grep -q $ENV_NAME
then
    echo $ENV_NAME found
    conda env update -f environment_$ENV_NAME.yml
else
    echo "new environemnt"$ENV_NAME
    conda env create -f environment_$ENV_NAME.yml
fi

source activate $ENV_NAME
#conda env export --no-builds > environment_$ENV_NAME.yml

#git config
git config --global user.email "hey_flora@126.com"
git config --global user.name "qianchu"

## pack python project
cd ../models/context2vec
python ./setup.py install

cd ../../
rare_we_dir=$(pwd)
echo $rare_we_dir
export PYTHONPATH="$rare_we_dir" >> $HOME/.bashrc