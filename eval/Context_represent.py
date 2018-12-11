
# coding: utf-8

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import math
from math import sqrt


import argparse
import sys
import numpy as np
from chainer import cuda
from context2vec.common.model_reader import ModelReader
import gensim
import unittest
from gensim.models import KeyedVectors
from allennlp.modules.elmo import Elmo, batch_to_ids
from functools import lru_cache
from copy import deepcopy

# define models
CONTEXT2VEC='context2vec'
CONTEXT2VEC_SUB='context2vec-skipgram'
A_LA_CARTE='alacarte'
SKIPGRAM='skipgram'
SKIPGRAM_ISF='skipgram_isf'
CONTEXT2VEC_SUB__SKIPGRAM=CONTEXT2VEC_SUB+'__'+SKIPGRAM
CONTEXT2VEC_SUB__SKIPGRAM_ISF=CONTEXT2VEC_SUB+'__'+SKIPGRAM_ISF
DTYPE = 'float64'
ALACARTE_FLOAT = np.float32
ELMO='elmo'
CONTEXT2VEC_SUB_ELMO='context2vec-elmo'
ELMO_WITH_TARGET='elmo_with_target'
CONTEXT2VEC_SUB___SKIPGRAM='context2vec-skipgram___skipgram'
CONTEXT2VEC_SUB_ELMO__ELMO_WITH_TARGET=CONTEXT2VEC_SUB_ELMO+'__'+ELMO_WITH_TARGET

stopw = stopwords.words('english')
stopw = [word.encode('utf-8') for word in stopw]


# helper function
def remove_stopword(word):

    if word not in stopw:
        return word
    else:
        return None

def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func

# general related functions
@lru_cache(maxsize=10000)
def read_context2vec_model(context2vec_param_file, gpu):
    model=ModelReader(context2vec_param_file, gpu)
    return model

def load_model_fromfile(context2vec_param_file,skipgram_param_file,elmo_param_file,gpu):
    context2vec_modelreader_out=None
    model_skipgram_out=None
    elmo=None
    if type(context2vec_param_file) == str:
        context2vec_modelreader_out = read_context2vec_model(context2vec_param_file, gpu)
    if type(skipgram_param_file) == str:
        model_skipgram_out = read_skipgram(skipgram_param_file)
    if type(elmo_param_file) == list:
        options_file =elmo_param_file[0]
        weight_file = elmo_param_file[1]
        elmo = Elmo(options_file, weight_file, 1, dropout=0)
    return context2vec_modelreader_out,model_skipgram_out,elmo


def read_skipgram(model_param_file):
        if not model_param_file.endswith('txt'):
            model_skipgram_out = gensim.models.Word2Vec.load(model_param_file)
        else:
            model_skipgram_out = KeyedVectors.load_word2vec_format(model_param_file)
       
        return model_skipgram_out


def read_context2vec(model_reader):
    w = model_reader.w
    index2word = model_reader.index2word
    word2index=model_reader.word2index
    model = model_reader.model

    return w,index2word,word2index,model


def process_sent(test_s,test_w):
    # test_s=test_s.replace(test_w, ' '+test_w+' ')
    words=test_s.split()
    # pos=words.index(test_w)
    return words,test_w


def load_w2salience(model_w2v,w2salience_f):
    w2salience={}
    with open(w2salience_f) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('sentence total'):
                sent_total = int(line.split(':')[1])
                continue
            w,w_count,s_count=line.split('\t')
            if model_w2v.wv.__contains__(w):
                    w2salience[w]=math.log(1+sent_total/float(s_count))
    return w2salience


def load_transform(Afile,model_dimension):
    """loads the transform from a text file
    Args:
    Afile: string; transform file name
    Returns:
    numpy array
    """
    if Afile is None:
        return None
    elif Afile.endswith('bin'):
        M = np.fromfile(Afile, dtype=ALACARTE_FLOAT)
        d = int(np.sqrt(M.shape[0]))
        assert d == model_dimension, "induction matrix dimension and word embedding dimension must be the same"
        M = M.reshape(d, d)
        M=M
        return M
    elif Afile.endswith('txt'):
        with open(Afile, 'r') as f:
            return np.vstack([np.array([ALACARTE_FLOAT(x) for x in line.split()]) for line in f])

# main class

class ContextModel():
    def __init__(self, model_type, gpu=-1, context2vec_modelreader=None,skipgram_model=None, elmo_model=None,n_result=20, ws_f=None,matrix_f=None):
        self.gpu=-1 if gpu is None else gpu
        self.xp = cuda.cupy if self.gpu >= 0 else np

        self.model_type=model_type
        self.n_result=20 if n_result is None else n_result
        
        self.load_model(context2vec_modelreader,skipgram_model,elmo_model)
        self.word_weight=load_w2salience(self.model_skipgram,ws_f) if ws_f is not None else None
        self.alacarte_m=self.xp.array(load_transform(matrix_f,self.model_dimension)) if matrix_f is not None else None

    def load_model(self, context2vec_modelreader,skipgram_model,elmo_model):
        if context2vec_modelreader is not None:
            self.context2vec_w,self.context2vec_index2word,self.context2vec_word2index,self.context2vec_model=read_context2vec(context2vec_modelreader)
            self.context2vec_w=self.xp.array(self.context2vec_w)
            self.model_dimension=self.context2vec_w[0].shape[0]
        if skipgram_model is not None:
            self.model_skipgram=skipgram_model
            self.model_skipgram_word2index = {key: self.model_skipgram.wv.vocab[key].index for key in self.model_skipgram.wv.vocab}
            self.model_dimension=self.model_skipgram.wv.vectors[0].shape[0]
        if elmo_model is not None:
            #To do elmo vocav
            self.model_elmo=elmo_model
            self.model_dimension=self.model_elmo.get_output_dim()
        elif self.model_type==CONTEXT2VEC_SUB___SKIPGRAM:
            self.model_dimension=self.model_skipgram.wv.vectors[0].shape[0]*2
     
    def compute_context_rep(self,words,pos,model):
        if model==CONTEXT2VEC:
            context_rep= self.context2vec_model.context2vec(words, pos)
        elif model==ELMO:
            context_rep=self.elmo_context_batch([words],[pos],[''])[0]
        elif model==CONTEXT2VEC_SUB:
            context_rep=self.context2vec_sub(words,pos)
        elif model==SKIPGRAM or model==SKIPGRAM_ISF:
            context_rep=self.skipgram_context(model,words, pos)
        elif model==A_LA_CARTE:
            context_rep=self.skipgram_context(model,words=words, pos=pos,stopw_rm=False)
        elif model==CONTEXT2VEC_SUB_ELMO:
            context_rep=self.context2vec_sub_elmo(words,pos)
        elif model==ELMO_WITH_TARGET:
            context_rep=self.elmo_context_batch([words],[pos],None)[0]

        else:
            print ('WARNING: incorrect model type{0}'.format(model))
        return context_rep
    
    def compute_context_reps(self,test_ss,test_w,model):  
        print ('model type is :{0}'.format(model))
        context_out=[]
        for test_id in range(len(test_ss)):
            test_s=test_ss[test_id]
            context_rep=self.compute_context_rep(test_s, test_w[test_id], model)
            if context_rep is not None:
                context_out.append(context_rep)
        return context_out
    
    def compute_context_reps_ensemble(self,test_ss,test_w_lst):
        context_out_concate=[]
        for ensemble_model in self.model_type.split('___'):
            context_out_ensemble = []
            for model in ensemble_model.split('__'):
                contexts_out=self.compute_context_reps(test_ss,test_w_lst,model)
                if contexts_out:
                    context_out=self.compute_context_reps_aggregate(contexts_out,model)
                    context_out_ensemble.append(context_out)
            context_out=sum(context_out_ensemble)/len(context_out_ensemble) if len(context_out_ensemble)!= 0 else None
            context_out_concate.append(context_out)
            if context_out is None:
                return None
        context_out_concate=self.xp.concatenate(context_out_concate)
        print ('model dimension is {0}'.format(context_out_concate.shape))

        return context_out_concate
    
    def compute_context_reps_aggregate(self,contexts_out,model):

        if model in [SKIPGRAM, SKIPGRAM_ISF, A_LA_CARTE]:
            context_weights, contexts_out = zip(*contexts_out)

        context_sum = sum(contexts_out)

        if self.alacarte_m is not None:
            context_out=self.alacarte_m.dot(context_sum)
        elif model ==SKIPGRAM or model==SKIPGRAM_ISF:
            context_out = context_sum / sum(context_weights) #if sum(context_weights)!=0 else context_sum
        else:
            context_out=context_sum/len(contexts_out)
        return context_out


    def elmo_context_batch(self,words_lsts,pos_lst,replace_w_lst):
        if replace_w_lst is not None:
            for i  in range(len(words_lsts)):
                 words_lsts[i][pos_lst[i]]=replace_w_lst[i]
        character_ids = batch_to_ids(words_lsts)
        context_reps = [context_per_sentence[pos_lst[i]] for i,context_per_sentence in enumerate(self.model_elmo(character_ids)['elmo_representations'][0].detach().numpy())]
        return context_reps


    def skipgram_context(self,model,words,pos,stopw_rm=True):
        context_wvs=[]
        weights=[]
        for i,word in enumerate(words):
            if i != pos: #surroudn context words
                if word in self.model_skipgram:
                    if stopw_rm==True and remove_stopword(word)==None:
                        continue
                    if model==SKIPGRAM_ISF:
                        weights.append(self.word_weight[word])
                        context_wvs.append(self.model_skipgram[word])
                    else:
                        #equal weights per word
                        context_wvs.append(self.model_skipgram[word])
                        weights.append(1.0)
                else:
                    pass
        context_embed=sum(self.xp.array(context_wvs)*self.xp.array(weights).reshape(len(weights),1))#/sum(weights)
        if sum(weights)==0:
            return None
        return sum(weights),context_embed


    def context2vec_sub_elmo(self,words,pos):
        context_rep=self.context2vec_model.context2vec(words, pos)
        top_vec,sim_scores,top_words=self.produce_top_n_simwords(self.context2vec_w,context_rep, self.context2vec_index2word, debug=False)
        contexts_sub,pos_lst,replace_w_lst=zip(*[(deepcopy(words), pos, top_words[i]) for i in range(len(top_words))])
        elmo_sub=self.elmo_context_batch(words_lsts=contexts_sub,pos_lst=pos_lst,replace_w_lst=replace_w_lst)
        elmo_sub_weighted_avg=self.xp.array(sum(elmo_sub*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))),dtype=elmo_sub[0].dtype)
        return elmo_sub_weighted_avg

    def context2vec_sub(self,words,pos):
        context_rep=self.context2vec_model.context2vec(words, pos)
        top_vec,sim_scores,top_words=self.produce_top_n_simwords(self.context2vec_w,context_rep, self.context2vec_index2word, debug=False)
        top_vec,index_list=self.lg_model_out_w2v(top_words,self.model_skipgram.wv.vectors,self.model_skipgram_word2index)
        if top_vec.shape[0]==0:
            return None
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        context_rep=self.xp.array(sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))))

        return context_rep

    def produce_top_n_simwords(self,w_filter, context_embed, index2word, debug=False):
        # assume that w_filter is already normalized
        context_embed = context_embed / sqrt((context_embed * context_embed).sum())
        similarity_scores = []
        #         print('producing top {0} simwords'.format(n_result))
        similarity = (w_filter.dot(context_embed) + 1.0) / 2
        top_words_i = []
        top_words = []
        count = 0
        for i in (-similarity).argsort():
            if self.xp.isnan(similarity[i]):
                continue
            if debug == True:
                try:
                    print('{0}: {1}'.format(str(index2word[int(i)]), str(similarity[int(i)])))
                except UnicodeEncodeError as e:
                    print (e)

            count += 1
            top_words_i.append(int(i))
            top_words.append(index2word[int(i)])
            similarity_scores.append(float(similarity[int(i)]))
            if count == self.n_result:
                break

        top_vec = w_filter[top_words_i, :]
        return top_vec, self.xp.array(similarity_scores), top_words

    def lg_model_out_w2v(self,top_words, w_target, word2index_target):
        # lg model substitutes in skipgram embedding
        top_vec = []
        index_list = []
        for i, word in enumerate(top_words):
            try:
                top_vec.append(w_target[word2index_target[word]])
                index_list.append(i)
            except KeyError as e:
                pass
        #                 print ('lg subs {0} not in w2v'.format(e))
        if top_vec == []:
            print ('no lg subs in w2v space')
            return self.xp.array([]), []
        else:
            return self.xp.stack(top_vec), index_list


# read in parameters and setup
class ArgTest(dict):
    def __init__(self, *args):
            super(ArgTest, self).__init__(*args)
            self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in self:
            return None
        else:
            return self[attr]


def parse_args(test_files):

    if len(sys.argv)==1:
        print ('missing arguments..')
        print ('load test')
        args=ArgTest(test_files)
    else:
        parser = argparse.ArgumentParser(description='Evaluate on long tail emerging ner')
        parser.add_argument('--cm',  type=str,
                            help='context2vec_param_file',dest='context2vec_param_file',default=None)
        parser.add_argument('--sm',type=str, default=None, dest='skipgram_param_file',help='skipgram_param_file')
        parser.add_argument('--m', dest='model_type', type=str,
                            help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
        parser.add_argument('--d', dest='data', type=str, help='data file',default=None)
        parser.add_argument('--g', dest='gpu',type=int, default=-1,help='gpu, default= -1')
        parser.add_argument('--ws', dest='w2salience_f',type=str, default=None,help='word2salience file, optional')
        parser.add_argument('--n_result',default=20,dest='n_result',type=int,help='top n result for language model substitutes')
        parser.add_argument('--ma', dest='matrix_f',type=str,default=None,help='matrix file for a la carte')
        parser.add_argument('--elmo',type=str, default=None, dest='elmo filename',nargs='+',help='elmo_param_file')

        args = parser.parse_args()
        print (args)

    return args  


def gpu_config(gpu):
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    

class TestCases2(unittest.TestCase):


    def test_skipgram(self):
        CM = ContextModel(skipgram_model=model_skipgram, gpu=-1,
                          model_type=SKIPGRAM)
        weight, context_embed = CM.compute_context_rep('enjoy fantastic ___', '___', SKIPGRAM)
        context_final=CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'], test_w='___')

        self.assertEquals(weight,2.0)
        self.assertSequenceEqual(list(context_embed[:10]),
                [0.15572980046272278, 0.391385093331337, -0.367725882679224, -0.4390959292650223, 0.46511872485280037, -0.08213193714618683, 0.11067408323287964, 0.05755297793075442, 0.06802653707563877, 0.2959321141242981])
        self.assertSequenceEqual(list(context_final[:10]),[-0.002902872860431671, 0.060669697355479, -0.16335702035576105, -0.14053642936050892, 0.2371379965916276, -0.014948914060369134, -0.023856326937675476, -0.15455589082557708, 0.18280773470178246, 0.0625985823571682]
)


    #
    def test_skipgram_weight(self):
        CM = ContextModel(skipgram_model=model_skipgram, gpu=-1,
                          model_type=SKIPGRAM_ISF,ws_f=args.w2salience_f)
        context_final=CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'], test_w='___')
        self.assertSequenceEqual(list(context_final[:10]),[-0.01189851459631592, 0.05283826293922492, -0.1513647235980044, -0.13441699829444592, 0.23343511758950736, -0.016666721591838407, -0.03688059408256895, -0.16563471932318508, 0.1904964859077803, 0.06445073124726468]
)




    def test_context2vec(self):
        CM = ContextModel(model_type=CONTEXT2VEC,context2vec_modelreader=context2vec_modelreader)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')

        self.assertSequenceEqual(xp.array(context_final[:10],dtype=DTYPE).tolist(),[0.25684037804603577, -0.2518651485443115, 0.8957875967025757, -0.264762818813324, -0.7319275140762329, 0.5076766610145569, 0.5815377831459045, -0.8884226679801941, -0.19761797785758972, 0.26606521010398865])




        
    def test_context_sub(self):
        CM = ContextModel(model_type=CONTEXT2VEC_SUB,skipgram_model=model_skipgram,context2vec_modelreader=context2vec_modelreader)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')

        self.assertSequenceEqual(xp.array(context_final[:10],dtype=DTYPE).tolist(),[0.03617460706238733, -0.007820711145649039, 0.019549923907292616, -0.039503757222290006, 0.09304707848039491, -0.17399982461609498, -0.1510802085625014, -0.05707189624693221, 0.2049098161107866, 0.02262262570894833])



    def test_context_sub__skipgram_isf(self):
        CM = ContextModel(model_type=CONTEXT2VEC_SUB__SKIPGRAM_ISF,skipgram_model=model_skipgram,context2vec_modelreader=context2vec_modelreader,ws_f=args.w2salience_f)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')
        print (xp.array(context_final[:10],dtype=DTYPE).tolist())
        self.assertSequenceEqual(xp.array(context_final[:10],dtype=DTYPE).tolist(),[0.012138046233035705, 0.022508775896787937, -0.06590739984535589, -0.08696037775836796, 0.16324109803495113, -0.09533327310396669, -0.09398040132253517, -0.11135330778505864, 0.19770315100928343, 0.04353667847810651]
)

    def test_zeroembed_context2vec_sub__skipgram_isf(self):
            CM = ContextModel(model_type=CONTEXT2VEC_SUB__SKIPGRAM_ISF, skipgram_model=model_skipgram,
                              context2vec_modelreader=context2vec_modelreader, ws_f=args.w2salience_f)
            context_final = CM.compute_context_reps_ensemble(test_ss=['___ !!!', 'hate horrible ___'],
                                                             test_w='___')
            print (xp.array(context_final[:10], dtype=DTYPE).tolist())
            self.assertSequenceEqual(xp.array(context_final[:10], dtype=DTYPE).tolist(),[-0.09380595129530858, -0.04850364374896774, -0.0945980830475423, -0.06501718103148113, 0.16707455899194756, -0.08880966052254462, -0.1424605893357314, -0.1779511408373386, 0.2257873964278589, -0.011529974631702328]
)
            context_final = CM.compute_context_reps_ensemble(test_ss=['___ !!!'],
                                                             test_w='___')
            print (xp.array(context_final[:10], dtype=DTYPE).tolist())
            self.assertSequenceEqual (xp.array(context_final[:10], dtype=DTYPE).tolist(),[-0.17036895272432656, 0.002701252657169536, -0.11705178024697765, -0.08133737505526306, 0.06545005713556959, -0.18760140187941493, -0.1258137400099445, 0.05292429965888347, 0.011674809238628125, -0.02235100849392118]
)

    def test_zeroembed_skipgram_isf(self):
            CM = ContextModel(model_type=SKIPGRAM_ISF, skipgram_model=model_skipgram,
                              context2vec_modelreader=context2vec_modelreader, ws_f=args.w2salience_f)

            context_final = CM.compute_context_reps_ensemble(test_ss=['___ !!!', 'hate horrible ___'],
                                                             test_w='___')
            print (xp.array(context_final[:10], dtype=DTYPE).tolist())
            self.assertSequenceEqual(xp.array(context_final[:10], dtype=DTYPE).tolist(),[-0.0864773801172783, -0.07905169760413748, -0.12716803108544453, -0.05863167144488575, 0.24123352123807362, 0.012780462113240625, -0.1150145902789262, -0.34118844846907126, 0.33422660234170193, -0.012813033840733106]
)
            context_final = CM.compute_context_reps_ensemble(test_ss=['___ !!!'],
                                                             test_w='___')
            self.assertIsNone(context_final)

    def test_alacarte(self):
        CM = ContextModel(model_type=A_LA_CARTE,skipgram_model=model_skipgram,matrix_f=args.matrix_f)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')
        self.assertSequenceEqual(xp.array(context_final[:10],dtype=DTYPE).tolist(),xp.array([-0.26741879756799725, 0.1904742187106585, -0.20207293493171546, -0.035881038638455404, 0.031569901468545686, -0.02004283636814247, 0.04682512920053014, -0.6029695352643941, 0.24818600094342189, 0.18152202352953248],dtype=DTYPE).tolist())

    def test_elmo(self):
        CM = ContextModel(model_type=ELMO,elmo_model=elmo_model)
        CM.elmo_context_batch([['i', 'do'], ['i', 'hate','you']], [0, 1], ['unk', 'unk'])
        CM.context2vec_sub_elmo(['i', 'hate','you'], 1)

if __name__=='__main__':
    print(sys.argv)
    args = parse_args(test_files=
        {
            'context2vec_param_file':'../models/context2vec/model_dir/MODEL-wiki.params.14',
         # 'skipgram_param_file':'../models/wiki_all.model/wiki_all.sent.split.model',
         'elmo_param_file':["https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json","https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"],

         # 'ws_f':'../corpora/corpora/WWC_norarew.txt.tokenized.vocab',
         # 'matrix_f':'../models/ALaCarte/transform/wiki_all_transform.bin'
            'gpu':-1
         })
    #gpu setup

    gpu_config(args.gpu)
    xp = cuda.cupy if args.gpu >= 0 else np


    #read in model

    context2vec_modelreader, model_skipgram, elmo_model = load_model_fromfile(skipgram_param_file=args.skipgram_param_file,context2vec_param_file=args.context2vec_param_file, elmo_param_file=args.elmo_param_file,gpu=args.gpu)
    CM = ContextModel(model_type=CONTEXT2VEC_SUB_ELMO, elmo_model=elmo_model,context2vec_modelreader=context2vec_modelreader,gpu=args.gpu)
    CM.context2vec_sub_elmo(['i', 'hate', 'you'], 1)

    # unittest.main(argv=['first-arg-is-ignored'], exit=False)

