__author__ = 'qianchu_liu'

from eval.Context_represent import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr,spearmanr

def evaluate_simlex(simlex_file,model,model_type):
    predicts = []
    golds = []
    line_num = 0
    with open(simlex_file) as f:
        for line in f:
            if line_num == 0:
                line_num += 1
                continue

            line = line.split('\t')
            try:
                if model_type==CONTEXT2VEC:
                    w=model[0]
                    word2index=model[1]
                    predict = cosine_similarity([w[word2index[line[0].split('-')[0]]]],
                                                [w[word2index[line[1].split('-')[0]]]])
                elif model_type==SKIPGRAM:
                    predict = cosine_similarity([model[line[0]]], [model[line[1]]])
            except KeyError as e:
                print(e)
                # predict = cosine_similarity([context_dependent([line[0]])],[context_dependent([line[1]])])

            print(line[3], predict[0][0])

            predicts.append(predict[0][0])

            golds.append(float(line[3]))

            line_num += 1

    print('simlex sim is {0}'.format(pearsonr(predicts, golds)))


def evaluate_men(men_dataset,model,model_type):
    predicts = []
    golds = []
    line_num = 0
    with open(men_dataset, 'r') as f:
        for line in f:

            #         if line_num==0:
            #             line_num+=1
            #             continue
            line = line.split()

            if model_type==SKIPGRAM:
                predict = cosine_similarity([model[line[0].split('-')[0]]],[model[line[1].split('-')[0]]])
            elif model_type==CONTEXT2VEC:
                w = model[0]
                word2index = model[1]
                predict = cosine_similarity([w[word2index[line[0].split('-')[0]]]],
                                            [w[word2index[line[1].split('-')[0]]]])
            print(line[2], predict[0][0])

            predicts.append(predict[0][0])

            golds.append(float(line[2]))

            line_num += 1

    print('MEN sim is {0}'.format(pearsonr(predicts, golds)))

if __name__=='__main__':
    skipgram_param_file='../models/wiki_all.model/wiki_all.sent.split.model'
    context2vec_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params'
    context2vec_modelreader, model_skipgram, elmo_model = load_model_fromfile(context2vec_param_file=context2vec_param_file,skipgram_param_file=skipgram_param_file,elmo_param_file=None,gpu=-1)

    CM=ContextModel(model_type=CONTEXT2VEC,skipgram_model=model_skipgram,context2vec_modelreader=context2vec_modelreader)
    evaluate_simlex(simlex_file='./eval_data/Simlex-999/Simlex-999.txt',model=(CM.context2vec_w,CM.context2vec_word2index),model_type=CONTEXT2VEC)
    evaluate_men(men_dataset='./eval_data/MEN/MEN_dataset_lemma_form_full',model=(CM.context2vec_w,CM.context2vec_word2index),model_type=CONTEXT2VEC)

