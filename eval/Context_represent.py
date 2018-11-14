
# coding: utf-8

# In[12]:
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



#define models
CONTEXT2VEC='context2vec'
CONTEXT2VEC_SUB='context2vec-skipgram'
A_LA_CARTE='alacarte'
SKIPGRAM='skipgram'
SKIPGRAM_ISF='skipgram_isf'
CONTEXT2VEC_SUB__SKIPGRAM_ISF='context2vec-skipgram?skipgram'
DTYPE = 'float64'
ALACARTE_FLOAT = np.float32

stopw = stopwords.words('english')
stopw = [word.encode('utf-8') for word in stopw]

# In[2]:


#helper function
def remove_stopword(word):

    if word not in stopw:
        return word
    else:
        return None
    
   


# In[3]:



#general related functions
def load_model_fromfile(context2vec_param_file,skipgram_param_file,gpu):
    context2vec_modelreader=None
    model_skipgram=None
    if context2vec_param_file != None:
        context2vec_modelreader = ModelReader(context2vec_param_file, gpu)
    if skipgram_param_file != None:
        model_skipgram = read_skipgram(skipgram_param_file)
    return context2vec_modelreader,model_skipgram

def read_skipgram(model_param_file):
        if not model_param_file.endswith('txt'):
            model_skipgram = gensim.models.Word2Vec.load(model_param_file)
        else:
            model_skipgram = KeyedVectors.load_word2vec_format(model_param_file)
       
        return model_skipgram

def read_context2vec(model_reader):
    w = xp.array(model_reader.w)
    index2word = model_reader.index2word
    word2index=model_reader.word2index
    model = model_reader.model

    return w,index2word,word2index,model

def process_sent(test_s,test_w):
    test_s=test_s.replace(test_w, ' '+test_w+' ')
    words=test_s.split()
    pos=words.index(test_w)
    return words,pos

def load_w2salience(model_w2v,w2salience_f):
    w2salience={}
    with open(w2salience_f) as f:
        for line in f:
            line=line.strip()
            if line=='':
                continue
            if line.startswith('sentence total'):
                sent_total=int(line.split(':')[1])
                continue
            w,w_count,s_count=line.split('\t')
            if model_w2v.wv.__contains__(w):
                    w2salience[w]=math.log(1+sent_total/float(s_count))
    return w2salience

def produce_top_n_simwords(w_filter,context_embed,n_result,index2word,debug=False):
        #assume that w_filter is already normalized
        context_embed = context_embed / sqrt((context_embed * context_embed).sum())
        similarity_scores=[]
#         print('producing top {0} simwords'.format(n_result))
        similarity = (w_filter.dot(context_embed)+1.0)/2
        top_words_i=[]
        top_words=[]
        count = 0
        for i in (-similarity).argsort():                
                    if xp.isnan(similarity[i]):
                        continue
                    if debug==True:
                        try:
                            print('{0}: {1}'.format(str(index2word[int(i)]), str(similarity[int(i)])))
                        except UnicodeEncodeError as e:
                            print (e)
                            
                    count += 1
                    top_words_i.append(int(i))
                    top_words.append(index2word[int(i)])
                    similarity_scores.append(float(similarity[int(i)]))
                    if count == n_result:
                        break

        top_vec=w_filter[top_words_i,:]
        return top_vec,xp.array(similarity_scores),top_words
 

def lg_model_out_w2v(top_words,w_target,word2index_target):
        # lg model substitutes in skipgram embedding
        top_vec=[]
        index_list=[]
        for i,word in enumerate(top_words):
            try :
                top_vec.append(w_target[word2index_target[word]])
                index_list.append(i)
            except KeyError as e:
                pass
#                 print ('lg subs {0} not in w2v'.format(e))
        if top_vec==[]:
            print ('no lg subs in w2v space')
            return xp.array([]),[]
        else:
            return xp.stack(top_vec),index_list
    
def load_transform(Afile,model_dimension):
    '''loads the transform from a text file
    Args:
    Afile: string; transform file name
    Returns:
    numpy array
    '''
    if Afile==None:
        return None
    elif Afile.endswith('bin'):
        M = np.fromfile(Afile, dtype=ALACARTE_FLOAT)
        d = int(np.sqrt(M.shape[0]))
        assert d == model_dimension, "induction matrix dimension and word embedding dimension must be the same"
        M = M.reshape(d, d)
        M=xp.array(M)
        return M
    elif Afile.endswith('txt'):
        with open(Afile, 'r') as f:
            return xp.array(np.vstack([np.array([ALACARTE_FLOAT(x) for x in line.split()]) for line in f]))
 


# In[4]:


# main class

class ContextModel():
    def __init__(self, model_type, gpu=-1, context2vec_modelreader=None,skipgram_model=None, n_result=20, ws_f=None,matrix_f=None):
        self.gpu=gpu
        self.model_type=model_type
        self.n_result=20 if n_result ==None else n_result
        
        self.load_model(context2vec_modelreader,skipgram_model)
        self.word_weight=load_w2salience(self.model_skipgram,ws_f) if ws_f!=None else None
        self.alacarte_m=load_transform(matrix_f,self.model_dimension)

   
    def load_model(self, context2vec_modelreader,skipgram_model):
        if type(context2vec_modelreader)!=type(None):
            self.context2vec_w,self.context2vec_index2word,self.context2vec_word2index,self.context2vec_model=read_context2vec(context2vec_modelreader)
            self.model_dimension=self.context2vec_w[0].shape[0]
        if type(skipgram_model) !=type(None):
            self.model_skipgram=skipgram_model
            self.model_skipgram_word2index = {key: self.model_skipgram.wv.vocab[key].index for key in self.model_skipgram.wv.vocab}
            self.model_dimension=self.model_skipgram.wv.vectors[0].shape[0]
     
    def compute_context_rep(self,test_s,test_w,model):
        words,pos=process_sent(test_s,test_w)
        if model==CONTEXT2VEC:
            context_rep= self.context2vec_model.context2vec(words, pos)
        elif model==CONTEXT2VEC_SUB:
            context_rep=self.context2vec_sub(words,pos)
        elif model==SKIPGRAM or model==SKIPGRAM_ISF:
            context_rep=self.skipgram_context(words, pos)
        elif model==A_LA_CARTE:
            context_rep=self.skipgram_context(words=words, pos=pos,stopw_rm=False)
        else:
            print ('WARNING: incorrect model type{0}'.format(model))
        return context_rep
    
    def compute_context_reps(self,test_ss,test_w,model):  
        print ('model type is :{0}'.format(model))
        context_out=[]
        for test_id in range(len(test_ss)):
            test_s=test_ss[test_id]
            test_s=test_s.lower().strip()
            context_out.append(self.compute_context_rep(test_s,test_w,model))
        
        return context_out
    
    def compute_context_reps_ensemble(self,test_ss,test_w):
        context_out_ensemble=[]
        for model in self.model_type.split('?'):
            contexts_out=self.compute_context_reps(test_ss,test_w,model)
            context_out=self.compute_context_reps_aggregate(contexts_out,model)
            context_out_ensemble.append(context_out)
        context_out=sum(context_out_ensemble)/len(context_out_ensemble)
        return context_out
    
    def compute_context_reps_aggregate(self,contexts_out,model):

        if model in [SKIPGRAM, SKIPGRAM_ISF, A_LA_CARTE]:
            context_weights, contexts_out = zip(*contexts_out)

        context_sum = sum(contexts_out)

        if type(self.alacarte_m)!=type(None):
            context_out=self.alacarte_m.dot(context_sum)
        elif model ==SKIPGRAM or model==SKIPGRAM_ISF:
            context_out = context_sum / sum(context_weights)
        else:
            context_out=context_sum/len(contexts_out)
        return context_out
    
    
    def skipgram_context(self,words,pos,stopw_rm=True):
        context_wvs=[]
        weights=[]
        for i,word in enumerate(words):
            if i != pos: #surroudn context words
                if word in self.model_skipgram:
                    if stopw_rm==True and remove_stopword(word)==None:
                        continue
                    if self.word_weight!=None:
                        weights.append(self.word_weight[word])
                        context_wvs.append(self.model_skipgram[word])
                    else:
                        #equal weights per word
                        context_wvs.append(self.model_skipgram[word])
                        weights.append(1.0)
                else:
                    pass
        context_embed=sum(np.array(context_wvs)*np.array(weights).reshape(len(weights),1))#/sum(weights)
        return (sum(weights),context_embed)

  
    
    def context2vec_sub(self,words,pos):
        context_rep=self.context2vec_model.context2vec(words, pos)
        top_vec,sim_scores,top_words=produce_top_n_simwords(self.context2vec_w,context_rep, self.n_result, self.context2vec_index2word, debug=False)
        top_vec,index_list=lg_model_out_w2v(top_words,self.model_skipgram.wv.vectors,self.model_skipgram_word2index)
        
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        context_rep=xp.array(sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))))
        return context_rep


# In[5]:


# read in parameters and setup
class ArgTest:
    def __init__(self,model_type=None,gpu=-1,context2vec_param_file=None, skipgram_param_file=None, n_result=None,w2salience_f=None,matrix_f=None):
        self.context2vec_param_file=context2vec_param_file
        self.skipgram_param_file=skipgram_param_file
        self.model_type=model_type
        self.n_result=n_result
        self.gpu=gpu
        self.w2salience_f=w2salience_f
        self.matrix_f=matrix_f
    
    

        
def parse_args():

    if sys.argv==[u'/Users/liuqianchu/OneDrive - University Of Cambridge/research/projects/year1/rare_we/eval/eval_ner.py']:
        print ('load test')
        args=ArgTest(context2vec_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14', skipgram_param_file='../models/wiki_all.model/wiki_all.sent.split.model')
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
        args = parser.parse_args()
        print args

    return args  



 
def gpu_config(gpu):
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    


# In[14]:




class TestCases(unittest.TestCase):


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
                          model_type=SKIPGRAM_ISF,ws_f='../corpora/corpora/WWC_norarew.txt.tokenized.vocab')
        context_final=CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'], test_w='___')
        self.assertSequenceEqual(list(context_final[:10]),[-0.01189851459631592, 0.05283826293922492, -0.1513647235980044, -0.13441699829444592, 0.23343511758950736, -0.016666721591838407, -0.03688059408256895, -0.16563471932318508, 0.1904964859077803, 0.06445073124726468]
)




    def test_context2vec(self):
        CM = ContextModel(model_type=CONTEXT2VEC,context2vec_modelreader=context2vec_modelreader)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')
        self.assertTrue(xp.array(context_final[:10],dtype=DTYPE).all()==xp.array([0.25684038, -0.25186515, 0.8957876, -0.26476282, -0.7319275, 0.50767666, 0.5815378, -0.88842267, -0.19761798, 0.2660652],dtype=DTYPE).all())


        
    def test_context_sub(self):
        CM = ContextModel(model_type=CONTEXT2VEC_SUB,skipgram_model=model_skipgram,context2vec_modelreader=context2vec_modelreader)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')

        self.assertTrue(xp.array(context_final[:10],dtype=DTYPE).all()==xp.array([ 0.03617461, -0.00782071,  0.01954992, -0.03950376,  0.09304708,
       -0.17399982, -0.15108021, -0.0570719 ,  0.20490982,  0.02262263],dtype=DTYPE).all())


    def test_context_sub__skipgram_isf(self):
        CM = ContextModel(model_type=CONTEXT2VEC_SUB__SKIPGRAM_ISF,skipgram_model=model_skipgram,context2vec_modelreader=context2vec_modelreader)
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')
        self.assertTrue(xp.array(context_final[:10],dtype=DTYPE).all()==xp.array([ 0.01663587, 0.02642449, -0.07190355 ,-0.09002009 ,0.16509254 ,-0.09447437,
 -0.08746827, -0.10581389,  0.19385878,  0.0426106 ],dtype=DTYPE).all())

    def test_alacarte(self):
        CM = ContextModel(model_type=A_LA_CARTE,skipgram_model=model_skipgram,matrix_f='../models/ALaCarte/transform/wiki_all_transform.bin')
        context_final = CM.compute_context_reps_ensemble(test_ss=['enjoy fantastic ___', 'hate horrible ___'],
                                                         test_w='___')
        self.assertTrue(xp.array(context_final[:10],dtype=DTYPE).all()==xp.array([-0.26741879756799725, 0.1904742187106585, -0.20207293493171546, -0.035881038638455404, 0.031569901468545686, -0.02004283636814247, 0.04682512920053014, -0.6029695352643941, 0.24818600094342189, 0.18152202352953248],dtype=DTYPE).all())
if __name__=='__main__':
    print(sys.argv)
    args = parse_args()
    #gpu setup

    gpu_config(args.gpu)
    xp = cuda.cupy if args.gpu >= 0 else np


    #read in model
    context2vec_modelreader, model_skipgram = load_model_fromfile(
        skipgram_param_file=args.skipgram_param_file,
        context2vec_param_file=args.context2vec_param_file, gpu=args.gpu)

    # TestCases.test_skipgram(model_skipgram)
    # CM=ContextModel(args.context2vec_param_file, args.skipgram_param_file, args.model_type, args.n_result, args.gpu,args.w2salience_f,args.matrix_f)
    # CM.compute_context_rep('I like ___',"___")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)

