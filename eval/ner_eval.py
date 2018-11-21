__author__ = 'qianchu_liu'


from Context_represent import *
from nltk.corpus.reader.conll import ConllCorpusReader
from copy import deepcopy
import os
import h5py
import torch
import torch.nn.functional as F


TARGET_W='___'



def as_numpy(array):
    if isinstance(array,list) or isinstance(array,np.ndarray):
        return np.array(array)
    else:
        return xp.asnumpy(array)


def return_tag_position(tagged_sent):
    tags_position=[]
    flag_tag_found=False
    for position,word_tag in enumerate(tagged_sent):
        if word_tag[1]!=u'O':
            if not flag_tag_found:
                start=position
                flag_tag_found=True
        else:
            if flag_tag_found:
                tags_position .append((start,position))
                flag_tag_found=False
    return tags_position

def from_label_to_y(labels):
    label2y={}
    for i,label in enumerate(labels):
        y = [0] * len(labels)
        y[i]=1
        label2y[label]=y
    return label2y
def convert_to_masked_wordlist(tagged_sent,tags_position):
    words_lst=list(zip(*tagged_sent)[0])
    tags_lst=list(zip(*tagged_sent)[1])
    for tag_position in tags_position:
        word_lst_masked=deepcopy(words_lst)
        word_lst_masked[tag_position[0]:tag_position[1]]='_'
        word_lst_masked[tag_position[0]] = TARGET_W
        tag=tags_lst[tag_position[0]][2:]
        yield word_lst_masked,tag


def process_sent(tagged_sent):
    tags_position=return_tag_position(tagged_sent)
    for masked_wordlist,tag in convert_to_masked_wordlist(tagged_sent,tags_position):
        yield masked_wordlist,tag


def process_tagged_sents(tagged_sents,batchsize,labels2y):
    masked_word_lsts=[]
    tags=[]
    for tagged_sent in tagged_sents:
        for masked_word_lst,tag in process_sent(tagged_sent):
            masked_word_lsts.append(masked_word_lst)
            tags.append(labels2y[tag]) if labels2y is not None else tags.append(tag)
            if len(tags)>=batchsize:
                yield masked_word_lsts,tags
                masked_word_lsts = []
                tags = []

    if len(tags)>0:
        yield masked_word_lsts,tags


def load_ner_data_label(root,filename,batchsize,labels2y=None):
    CCR = ConllCorpusReader(root=root, fileids='.conll',
                            columntypes=('words', 'pos', 'ne', 'chunk'))
    for masked_word_lsts, tags in process_tagged_sents(CCR.tagged_sents(filename),batchsize,labels2y):
        yield masked_word_lsts,tags

def check_embed_file_exist(parsed_args,part):

    fn_lst=os.path.join(parsed_args.output_dir,'_'.join([arg.split('/')[-1] for arg in [part,parsed_args.model_type,parsed_args.context2vec_param_file,parsed_args.skipgram_param_file] if arg is not None]))
    embedding_fname = fn_lst+'.h5'
    if os.path.isfile(embedding_fname):
        return embedding_fname,True
    else:
        return embedding_fname,False




def compute_rep_chunk(data_label_generator,CM):
    for word_lsts,tags in data_label_generator:
        context_reps = []
        for sent in word_lsts:
            print (sent)
            context_rep=CM.compute_context_reps_ensemble([' '.join(sent)], TARGET_W)
            if context_rep is None:
                context_reps.append(np.zeros(CM.model_dimension))
            else:
                context_reps.append(as_numpy(context_rep))
        yield np.array(context_reps)


def increment_write_h5py(hf,chunk,data_name='data'):
    if data_name not in hf:
        maxshape = (None,) + chunk.shape[1:]
        data=hf.create_dataset(data_name,data=chunk,chunks=chunk.shape,maxshape=maxshape,compression="gzip",compression_opts=9)

    else:
        data=hf[data_name]
        data.resize((chunk.shape[0]+data.shape[0],)+data.shape[1:])
        data[-chunk.shape[0]:]=chunk

def write_embedding_tofile(args,part) :

    embedding_fname,embedding_file_exist=check_embed_file_exist(args,part)
    print ('load data from', embedding_fname)
    if not embedding_file_exist:
        data_label_generator = load_ner_data_label(root=args.data, filename=PARTITION[part],
                                                   batchsize=args.batchsize)
        context2vec_modelreader, model_skipgram = load_model_fromfile(
            skipgram_param_file=args.skipgram_param_file,
            context2vec_param_file=args.context2vec_param_file, gpu=args.gpu)
        CM = ContextModel(model_type=args.model_type, context2vec_modelreader=context2vec_modelreader,
                          skipgram_model=model_skipgram, n_result=args.n_result, ws_f=args.w2salience_f,
                          matrix_f=args.matrix_f)
        hf = h5py.File(embedding_fname,'w')
        for chunk in compute_rep_chunk(data_label_generator, CM):
            increment_write_h5py(hf, chunk)

        hf.close()
    return embedding_fname


class ArgTestNer(dict):
    def __init__(self, *args):
            super(ArgTestNer, self).__init__(*args)
            self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in self:
            return None
        else:
            return self[attr]


def parse_args_ner(test_files):
    if len(sys.argv) == 1:
        print ('missing arguments..')
        print ('load test')
        args=ArgTestNer(test_files)

    else:
        parser = argparse.ArgumentParser(description='Evaluate on long tail emerging ner')
        parser.add_argument('--cm', type=str,
                            help='context2vec_param_file', dest='context2vec_param_file', default=None)
        parser.add_argument('--sm', type=str, default=None, dest='skipgram_param_file', help='skipgram_param_file')
        parser.add_argument('--m', dest='model_type', type=str,
                            help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
        parser.add_argument('--d', dest='data', type=str, help='data file', default=None)
        parser.add_argument('--g', dest='gpu', type=int, default=-1, help='gpu, default= -1')
        parser.add_argument('--ws', dest='w2salience_f', type=str, default=None, help='word2salience file, optional')
        parser.add_argument('--n_result', default=20, dest='n_result', type=int,
                            help='top n result for language model substitutes')
        parser.add_argument('--ma', dest='matrix_f', type=str, default=None, help='matrix file for a la carte')
        parser.add_argument('--tot', dest='train_or_test', type=str, default='train',help='training flag')
        parser.add_argument('--output_dir', dest='output_dir', type=str,help='output dir for embedding file and learned model')
        parser.add_argument('--batchsize', dest='batchsize', type=int,help='batchsize',default=100)
        parser.add_argument('--lr',dest='lr',type=float,help='learning rate', default=1e-4)
        parser.add_argument('--ep',dest='epochs',type=float,help='epochs', default=10)
        parser.add_argument('--n',dest='save_every_n',type=int,help='save every n epochs', default=5)
        parser.add_argument('--path',dest='model_path',type=str,help='mlp model path')

        args = parser.parse_args()
        print args

    return args

def gpu_device(gpu):
    gpu_config(args.gpu)

    if gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:{0}'.format(gpu))
    return device

def generate_embed_labels_batch(args,embed_data,batchsize,labels2y,train_or_test):
    data_label_generator = load_ner_data_label(root=args.data, filename=PARTITION[train_or_test],
                                               batchsize=batchsize, labels2y=labels2y)
    l=len(embed_data)
    for start in range(0,l,batchsize):
        yield embed_data[start:min(start+batchsize,l)], next(data_label_generator)[1]

def validate_per_epoch(embed_dev,args,criterion,labels2y,model):

    loss_dev=0
    for data_batch,y in generate_embed_labels_batch(args, embed_dev,len(embed_dev), labels2y,'dev'):
        y_pred_dev = model(torch.tensor(data_batch).float())
        y = torch.max(torch.tensor(y), 1)[1]
        loss_dev_per_batch = criterion(y_pred_dev, y)
        loss_dev+=loss_dev_per_batch.item()
    return loss_dev

def train(embedding_train_fname, embedding_dev_fname,device,args,labels2y,H):
    # dtype = torch.float

    train_file=h5py.File(embedding_train_fname,'r')
    embed_train=train_file['data']
    dev_file = h5py.File(embedding_dev_fname,'r')
    embed_dev = dev_file['data']

    D_in, D_out = embed_train[0].shape[0], len(labels2y)
    model=TwoLayerNet(D_in,H,D_out)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    for t in range(args.epochs):
        loss_per_epoch=0
        for data_batch,y in generate_embed_labels_batch(args,embed_train,args.batchsize,labels2y,'train'):

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(torch.tensor(data_batch).float())
            # Compute and print loss
            y=torch.max(torch.tensor(y),1)[1]
            loss=criterion(y_pred,y)
            loss_per_epoch+=loss.item()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break
        print ('training loss per epoch: {0},epoch {1}'.format(loss_per_epoch,t))
        loss_dev=validate_per_epoch(embed_dev=embed_dev,args=args,criterion=criterion,labels2y=labels2y,model=model)
        print ('dev loss per epoch: {0}'.format(loss_dev))

        if t%args.save_every_n==0 and t>=args.save_every_n:
            torch.save(model.state_dict(), embedding_train_fname+'_epoch{0}_loss{1}'.format(t,loss_dev))

    train_file.close()
    dev_file.close()

def test_output(y_pred,args,LABELS):
    CCR = ConllCorpusReader(root=args.data, fileids='.conll',
                            columntypes=('words', 'pos', 'ne', 'chunk'))
    tagged_sents=CCR.tagged_sents(PARTITION['test'])
    pred_conll=[]
    counter=0
    for tagged_sent in tagged_sents:
        tags_position=return_tag_position(tagged_sent)
        token_lst,tags_lst = zip(*tagged_sent)
        tags_pred=deepcopy(list(tags_lst))
        for tag_pos in tags_position:
            for pos in range(tag_pos[0],tag_pos[1]):
                tags_pred[pos]=tags_lst[pos][:2]+LABELS[y_pred[counter]]
            counter+=1
        pred_conll.append(zip(token_lst, tags_lst, tags_pred))
    return pred_conll
            # f.write('\n'.join([' '.join(token) for token in zip(token_lst, tags_lst, tags_pred)]))
            # f.write('\n')

def output_conll_to_file(pred_conll,output_fname):
    with open(output_fname, 'w') as f:
        f.write('\n\n'.join(['\n'.join(['	'.join(token) for token in sent]) for sent in pred_conll ]).encode('utf-8'))



def test(path,test_data_embed_fname,H,labels2y):
    test_file = h5py.File(test_data_embed_fname,'r')
    embed_test=test_file['data']
    D_in, D_out = embed_test[0].shape[0], len(labels2y)

    model=TwoLayerNet(D_in,H,D_out)
    model.load_state_dict(torch.load(path))
    model.eval()

    for data_batch, y in generate_embed_labels_batch(args, embed_test, len(embed_test), labels2y, 'test'):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = torch.max(model(torch.tensor(data_batch).float()),1)[1]

    test_file.close()
    return y_pred

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


    # for h5py.File(embedding_file)['data'].value[:10]

if __name__ == "__main__":
    print ('start')
    import time
    LABELS=[ u'person',u'location',u'group',u'creative-work',u'corporation',u'product']
    PARTITION = {'train': 'wnut17train.conll', 'test': 'emerging.test.annotated', 'dev': 'emerging.dev.conll'}
    LABELS2Y=from_label_to_y(LABELS)
    start_time = time.time()
    Hidden_unit=100

    # 1. load args
    args = parse_args_ner(test_files=
                      {
                          'context2vec_param_file': '../models/context2vec/model_dir/MODEL-wiki.params.14',
                       'skipgram_param_file': '../models/wiki_all.model/wiki_all.sent.split.model',
                       # 'ws_f': '../corpora/corpora/WWC_norarew.txt.tokenized.vocab',
                       # 'matrix_f': '../models/ALaCarte/transform/wiki_all_transform.bin',
                       'data':'./eval_data/emerging_entities/',
                       'train_or_test':'test',
                       'model_type':CONTEXT2VEC_SUB,
                       'output_dir':'./results/ner/',
                          'batchsize':10,
                          'lr':0.01,
                          'epochs':200,
                          'save_every_n':5,
                          'model_path':'./results/ner/train_context2vec-skipgram_MODEL-wiki.params.14_wiki_all.sent.split.model.h5_epoch35_loss1.20299649239'
                       })




    # 2. load data

    # 3. gpu setup
    device=gpu_device(args.gpu)
    xp = cuda.cupy if args.gpu >= 0 else np



    # 4. read embeddings and train
    # params = {'batch_size': 64,
    #           'shuffle': True,
    #           'num_workers': 6}
    # max_epochs = 100

    if args.train_or_test=='train':
        print ('load data>>')
        train_data_embed_fname=write_embedding_tofile(args,'train')
        dev_data_embed_fname=write_embedding_tofile(args,'dev')
        print ('train...')
        # train
        train(embedding_train_fname=train_data_embed_fname,embedding_dev_fname=dev_data_embed_fname,device=device,args=args, labels2y=LABELS2Y,H=Hidden_unit)
        # validation
    # read embeddings and test
    elif args.train_or_test=='test':
        test_data_embed_fname=write_embedding_tofile(args,'test')
        y_pred=test(path=args.model_path,test_data_embed_fname=test_data_embed_fname,H=Hidden_unit,labels2y=LABELS2Y)
        pred_conll=test_output(y_pred=y_pred,args=args,LABELS=LABELS)
        output_conll_to_file(pred_conll,output_fname=test_data_embed_fname+'.output')
        #test
        pass


    print ('processing ner sentences costs {0}'.format(time.time() - start_time))
