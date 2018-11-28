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

#git config
git config --global user.email "hey_flora@126.com"
git config --global user.name "qianchu"

## pack python project and setup pythonpath
cd ../models/context2vec
python ./setup.py install

cd ../../
rare_we_dir="$(pwd | sed 's/ /\\ /g')"
echo $rare_we_dir

if [ -z "${PYTHONPATH}" ]; then echo export PYTHONPATH="$rare_we_dir" >> $HOME/.bashrc; else echo "pythonpath found"; fi

bash