
# coding: utf-8

# In[12]:


import argparse
import sys
import numpy as np
from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader
import gensim
import unittest
#define models
CONTEXT2VEC='context2vec'
CONTEXT2VEC_SUB='context2vec-skipgram'
A_LA_CARTE='alacarte'
SKIPGRAM='skipgram'
SKIPGRAM_ISF='skipgram_isf'
CONTEXT2VEC_SUB__SKIPGRAM_ISF='context2vec-skipgram?skipgram'


# In[2]:


#helper function
def remove_stopword(word):
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    if word not in stopw:
        return word
    else:
        return None
    
   


# In[3]:



#general related functions
def read_skipgram(model_param_file):
        if not model_param_file.endswith('txt'):
            model_skipgram = gensim.models.Word2Vec.load(model_param_file)
        else:
            model_skipgram = KeyedVectors.load_word2vec_format(model_param_file)
       
        return model_skipgram

def read_context2vec(model_param_file,gpu):
    model_reader = ModelReader(model_param_file,gpu)
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
    
def load_transform(Afile,model):
    '''loads the transform from a text file
    Args:
    Afile: string; transform file name
    Returns:
    numpy array
    '''
    if Afile==None:
        return None
    elif Afile.endswith('bin'):
        M = np.fromfile(Afile, dtype=FLOAT)
        d = int(np.sqrt(M.shape[0]))
        print (d)
        assert d == next(iter(model.values())).shape[0], "induction matrix dimension and word embedding dimension must be the same"
        M = M.reshape(d, d)
        M=xp.array(M)
        return M
    elif Afile.endswith('txt'):
        with open(Afile, 'r') as f:
            return xp.array(np.vstack([np.array([FLOAT(x) for x in line.split()]) for line in f]))
 


# In[4]:


# main class

class ContextModel():
    def __init__(self, context2vec_param_file,skipgram_param_file, model_type, gpu, n_result, ws_f,matrix_f):
        self.gpu=gpu
        self.model_type=model_type
        self.n_result=20 if n_result ==None else n_result
        
        self.load_model(context2vec_param_file,skipgram_param_file)
        self.word_weight=load_w2salience(self.model_skipgram,ws_f) if ws_f!=None else None
        self.alacarte_m=load_transform(matrix_f,{word:self.model_skipgram.wv.__getitem__(word) for word in self.model_skipgram.wv.vocab})

   
    def load_model(self, context2vec_param_file,skipgram_param_file):
        if context2vec_param_file!=None:
            self.context2vec_w,self.context2vec_index2word,self.context2vec_word2index,self.context2vec_model=read_context2vec(context2vec_param_file,self.gpu)
        elif skipgram_param_file !=None:
            self.model_skipgram=read_skipgram(skipgram_param_file)
     
    def compute_context_rep(self,test_s,test_w,model):
        words,pos=process_sent(test_s,test_w)        
        if model==CONTEXT2VEC:
            context_rep=self.context2vec_model(words, pos)
        elif model==CONTEXT2VEC_SUB:
            context_rep=self.context2vec_sub(word,pos)
        elif model==SKIPGRAM or model==SKIPGRAM_ISF:
            context_rep=self.skipgram_context(words, pos)
        elif model==A_LA_CARTE:
            context_rep=self.skipgram_context(words, pos)
        else:
            print ('WARNING: incorrect model type{0}'.format(model))
        return context_rep
    
    def compute_context_reps(self,test_ss,test_w,model):  
        print ('model type is :{0}'.format(model_type))
        context_out=[]
        for test_id in range(len(test_ss)):
            test_s=test_ss[test_id]
            test_s=test_s.lower().strip()
            context_out.append(self.compute_context_rep(test_s,test_w,model))
        
        return contexts_out
    
    def compute_context_reps_ensemble(self,test_ss,test_w):
        context_out_ensemble=[]
        for model in self.model_type:
            contexts_out=self.compute_context_reps(test_ss,test_w,model)
            contxt_out=self.compute_context_reps_aggregate(contexts_out)
            context_out_ensemble.append(context_out)
        context_out=sum(context_out_ensemble)/len(context_out_ensemble)
        return context_out
    
    def compute_context_reps_aggregate(self,contexts_out,model):
        if model==SKIPGRAM or SKIPGRAM_ISF or A_LA_CARTE:
            context_out,context_weights=zip(* contexts_out)
            context_out=sum(context_out)/sum(context_weights)
        else:
            context_out=sum(contexts_out)/len(contexts_out)

        if model==A_LA_CARTE:
            context_out=self.alacarte_m.dot(contexts_out)
        return context_out
    
    
    def skipgram_context(self,words,pos):
        context_wvs=[]
        weights=[]
        for i,word in enumerate(words):
            if i != pos: #surroudn context words
                if word in self.model_skipgram and remove_stopword(word)!=None:
                    if self.word_weight!=None:
                        weights.append(self.word_weight[word])
                        context_wvs.append(model[word])                   
                    else:
                        #equal weights per word
                        context_wvs.append(model[word])
                        weights.append(1.0)
                else:
                    pass
        context_embed=sum(np.array(context_wvs)*np.array(weights).reshape(len(weights),1))#/sum(weights)
    #     print ('skipgram context sum:', context_embed[:10])
        return (sum(weights),context_embed)

  
    
    def context2vec_sub(self,word,pos):
        
        context_rep=self.context2vec_model(words, pos)
        top_vec,sim_scores,top_words=produce_top_n_simwords(self.context2vec_w,context_rep, self.n_result, self.context2vec_index2word, debug=True)
        top_vec,index_list=lg_model_out_w2v(top_words,self.model_skipgram.wv.vectors,self.model_skipgram.wv.index2word) 
        
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        context_rep=xp.array(sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))))
        return context_rep


# In[5]:


# read in parameters and setup
class ArgTest:
    def __init__(self,model_type,gpu,context2vec_param_file=None, skipgram_param_file=None, n_result=None,w2salience_f=None,matrix_f=None):
        self.context2vec_param_file=context2vec_param_file
        self.skipgram_param_file=skipgram_param_file
        self.model_type=model_type
        self.n_result=n_result
        self.gpu=gpu
        self.w2salience_f=w2salience_f
        self.matrix_f=matrix_f
    
    

        
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on long tail emerging ner')
    parser.add_argument('--cm',  type=str,
                        help='context2vec_param_file',dest='context2vec_param_file',default=None)
    parser.add_argument('--sm',type=str, default=None, help='skipgram_param_file')
    parser.add_argument('--m', dest='model_type', type=str,
                        help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
    parser.add_argument('--d', dest='data', type=str, help='data file')
    parser.add_argument('--g', dest='gpu',type=int, default=-1,help='gpu, default is -1')
    parser.add_argument('--ws', dest='w2salience_f',type=str, default=None,help='word2salience file, optional')
    parser.add_argument('--n_result',default=20,dest='n_result',type=int,help='top n result for language model substitutes')
    parser.add_argument('--ma', dest='matrix_f',type=str,default=None,help='matrix file for a la carte')
    args = parser.parse_args()
    return args  
    
        
def read_args():
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        args=ArgTest(
#             context2vec_param_file='../models/context2vec/model_dir/MODEL-wiki.params.12',
            skipgram_param_file='../models/wiki_all.model/wiki_all.sent.split.model',
            model_type=SKIPGRAM,
            n_result=20,
            gpu=-1,
#             w2salience_f='../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab',
#             matrix_f='../models/ALaCarte/transform/nonce_samecorpus.bin'

        )
    else:
        args=parse_args()
    return args

 
def gpu_config(gpu):
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    


# In[14]:


get_ipython().magic(u'config IPCompleter.greedy=True')


class TestCases(unittest.TestCase):
        
    def test_skipgram(self):
        super(TestCases, self).init()
        self.
#         CM=ContextModel(skipgram_param_file='../models/wiki_all.model/wiki_all.sent.split.model',gpu=1,model_type=SKIPGRAM)
        pass
     
class Test(list):
    
    def test(self):
        super
    
        

        


# In[10]:


if __name__=='__main__':
#     args=read_args()
    
#     #gpu setup
#     gpu_config(args.gpu)
#     xp = cuda.cupy if args.gpu >= 0 else np
    
#     #read in model
#     CM=ContextModel(args.context2vec_param_file, args.skipgram_param_file, args.model_type, args.n_result, args.gpu,args.w2salience_f,args.matrix_f)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

