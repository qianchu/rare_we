
# coding: utf-8

# In[49]:


def load_nonces(partition):
  with open('./eval_data/data-nonces/n2v.definitional.dataset.'+partition+'.txt', 'r') as f:
    return zip(*((n, [w for w in d.split() if not w == '___']) for n, d in (line.split('\t') for line in f if not line[0] in {'#', '\n'})))

def load_nonce_vocab(vocab_f):
    with open(vocab_f) as f:
        return {line.split('\t')[0]:line.split('\t')[1] for line in f if not line.startswith('sentence total')}

def load_crw(crw_data):
    with open(crw_data) as f:
        return {line.split(' ')[0]:line.split(' ')[1] for line in f if not line.startswith('sentence total')}
                   


# In[44]:


# [w for w in load_nonces('test')[0]]
nonce2freq=load_nonce_vocab('../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab')
nonces=load_nonces('test')[0]


# In[50]:


crws=load_crw('./eval_data/CRW/rarevocab.txt').keys()


# In[55]:


nonce_freq=[int(nonce2freq[nonce]) if nonce in nonce2freq else 0 for nonce in nonces ]
crws_freq=[int(nonce2freq[crw]) if crw in nonce2freq else 0 for crw in crws ]


# In[67]:


# len(crws_freq)
import numpy as np
np.mean(crws_freq)


# In[66]:


np.mean(nonce_freq)

