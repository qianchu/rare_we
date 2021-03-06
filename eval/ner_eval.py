__author__ = 'qianchu_liu'



from eval.Context_represent import *
from nltk.corpus.reader.conll import ConllCorpusReader
from copy import deepcopy
import h5py
import torch
import os
from collections import defaultdict




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
    # if flag_tag_found:
    #     tags_position.append((start,position+1))
    return tags_position

# def return_tag_position_test(tagged_sent):
#     tags_position=[]
#     flag_tag_found=False
#     for position,word_tag in enumerate(tagged_sent):
#         if word_tag[1]!=u'O':
#             if not flag_tag_found:
#                 start=position
#                 flag_tag_found=True
#         else:
#             if flag_tag_found:
#                 tags_position .append((start,position))
#                 flag_tag_found=False
#     if flag_tag_found:
#         tags_position.append((start,position+1))
#     return tags_position

def from_label_to_y(labels):
    label2y={}
    for i,label in enumerate(labels):
        y = [0] * len(labels)
        y[i]=1
        label2y[label]=y
    return label2y
def convert_to_masked_wordlist(tagged_sent,tags_position):
    words_lst=list(list(zip(*tagged_sent))[0])
    tags_lst=list(list(zip(*tagged_sent))[1])
    for tag_position in tags_position:
        word_lst_masked=deepcopy(words_lst)
        targetw=''.join(word_lst_masked[tag_position[0]:tag_position[1]])
        word_lst_masked[tag_position[0]:tag_position[1]]='_'
        word_lst_masked[tag_position[0]] = targetw
        tag=tags_lst[tag_position[0]][2:]
        yield word_lst_masked,tag_position[0],tag


def process_sent(tagged_sent):
    tags_position=return_tag_position(tagged_sent)
    for masked_wordlist,pos,tag in convert_to_masked_wordlist(tagged_sent,tags_position):
        yield masked_wordlist,pos,tag


def process_tagged_sents(tagged_sents,batchsize,labels2y):
    masked_word_lsts=[]
    tags=[]
    pos_lst=[]
    for tagged_sent in tagged_sents:
        for masked_word_lst,pos,tag in process_sent(tagged_sent):
            masked_word_lsts.append(masked_word_lst)
            pos_lst.append(pos)
            tags.append(labels2y[tag]) if labels2y is not None else tags.append(tag)
            if len(tags)>=batchsize:
                yield masked_word_lsts,pos_lst,tags
                masked_word_lsts = []
                tags = []
                pos_lst=[]

    if len(tags)>0:
        yield masked_word_lsts,pos_lst,tags


def load_ner_data_label(root,filename,batchsize,labels2y=None):
    CCR = ConllCorpusReader(root=root, fileids='.conll',
                            columntypes=('words', 'pos', 'ne', 'chunk'))
    for masked_word_lsts, pos_lst,tags in process_tagged_sents(CCR.tagged_sents(filename),batchsize,labels2y):
        yield masked_word_lsts,pos_lst,tags

def check_embed_file_exist(parsed_args,part,model_type=None):
    if model_type is None:
        model_type=parsed_args.model_type

    if model_type==ELMO_WITH_TARGET:
        context2vec_param_file=None
    else:
        context2vec_param_file=parsed_args.context2vec_param_file
    fn_lst=os.path.join(parsed_args.output_dir,'_'.join([arg.split('/')[-1] for arg in [part,model_type,context2vec_param_file,parsed_args.skipgram_param_file] if arg is not None]))
    embedding_fname = fn_lst+'.h5'
    if os.path.isfile(embedding_fname):
        return embedding_fname,True
    else:
        return embedding_fname,False




def compute_rep_chunk(data_label_generator,CM):
    for word_lsts,pos_lst,tags in data_label_generator:
        context_reps = []
        for i,sent in enumerate(word_lsts):
            print (sent)
            context_rep=CM.compute_context_reps_ensemble([sent], [pos_lst[i]])
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

def produce_ensemble_embedding(embedding_fname_lst,hf,ensemble_method):
    data_embed_lst=[h5py.File(fname, 'r')['data'] for fname in embedding_fname_lst]

    l=len(data_embed_lst[0])

    for start in range(0, l, args.batchsize):
        ensemble_data_embed_lst=[data_embed[start:min(start + args.batchsize, l)] for data_embed in data_embed_lst]
        if ensemble_method=='concatenate':
            ensemble_data_embed=xp.concatenate(ensemble_data_embed_lst,axis=1)
        elif ensemble_method=='average':
            ensemble_data_embed=sum(ensemble_data_embed_lst)/len(ensemble_data_embed_lst)
        else:
            print ('Warning, ensemble method unknown')
            exit(1)
        print(ensemble_data_embed.shape)
        increment_write_h5py(hf, ensemble_data_embed)


def check_ensemble_file_exist(args,part):
    embedding_f_lst=[]

    for model_type in args.model_type.split('__'):
        embedding_fname_compon, embedding_file_exist_compon = check_embed_file_exist(args, part,model_type)
        if embedding_file_exist_compon is False:
           return None
        else:
            embedding_f_lst.append(embedding_fname_compon)
    return embedding_f_lst


def write_embedding_tofile(args,part,ensemble_method='concatenate') :

    embedding_fname,embedding_file_exist=check_embed_file_exist(args,part)
    print ('load data from', embedding_fname)
    if not embedding_file_exist:
        hf = h5py.File(embedding_fname, 'w')
        embeddng_f_lst=check_ensemble_file_exist(args,part)
        if embeddng_f_lst is not None:
            produce_ensemble_embedding(embeddng_f_lst,hf,ensemble_method)



        else:
            data_label_generator = load_ner_data_label(root=args.data, filename=PARTITION[part],
                                                       batchsize=args.batchsize)
            context2vec_modelreader, model_skipgram,model_elmo = load_model_fromfile(
                skipgram_param_file=args.skipgram_param_file,
                context2vec_param_file=args.context2vec_param_file, elmo_param_file=args.elmo_param_file,gpu=args.gpu)
            CM = ContextModel(model_type=args.model_type, context2vec_modelreader=context2vec_modelreader,
                              skipgram_model=model_skipgram, elmo_model=model_elmo,n_result=args.n_result, ws_f=args.w2salience_f,
                              matrix_f=args.matrix_f)
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
        parser.add_argument('--elmo',type=str, default=None, dest='elmo_param_file',nargs='+',help='elmo_param_file')

        parser.add_argument('--m', dest='model_type', type=str,
                            help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram__skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
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
        parser.add_argument('--ep',dest='epochs',type=int,help='epochs', default=10)
        parser.add_argument('--n',dest='save_every_n',type=int,help='save every n epochs', default=20)
        parser.add_argument('--run',dest='n_runs',type=int,help='number of runs of the MLP', default=5)

        parser.add_argument('--path',dest='model_path',type=str,help='mlp model path')

        args = parser.parse_args()
        print (args)

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
        yield embed_data[start:min(start+batchsize,l)], next(data_label_generator)[2]

def calculate_f1(y_pred,y_gold):
    f1_lst=[]
    result_dict=defaultdict(lambda: defaultdict(float))
    for i,tag in enumerate(list(y_gold)):
        if y_pred[i]==tag:
            result_dict[int(tag)]['correct']+=1
        else:
            result_dict[int(tag)]['fn']+=1
            result_dict[int(y_pred[i])]['fp']+=1
    for entity in result_dict:
        try:
            precision=result_dict[entity]['correct']/(result_dict[entity]['correct']+result_dict[entity]['fn'])
        except ZeroDivisionError as e:
            precision=0.
        try:
            recall=result_dict[entity]['correct']/(result_dict[entity]['correct']+result_dict[entity]['fp'])
        except:
            recall=0.
        try:
            f1= 2*((precision*recall)/(precision+recall))
        except:
            f1=0
        f1_lst.append(f1)
        # print ('==entity {0} precision {1} recall {2} f1 {3}'.format(entity,precision,recall,f1))
    f1_macro=sum(f1_lst)/len(f1_lst)
    print ('==f1 mean {0}'.format(f1_macro))
    return f1_macro


def accuracy(y_pred,y_gold):
    accurate_items=[i for i in range(len(y_pred)) if y_pred[i]==y_gold[i]]
    return len(accurate_items)
def validate_per_epoch(embed_dev,args,criterion,labels2y,model,part):
    accur=0
    loss_dev=0
    for data_batch,y in generate_embed_labels_batch(args, embed_dev,len(embed_dev), labels2y,part):
        # data_batch,y=filter_data(data_batch,y)
        y_pred_dev=model(torch.tensor(data_batch).float())
        y_pred_dev_argmax = torch.max(y_pred_dev,1)[1]
        # print ('pred',y_pred_dev[:10])
        # print ('argmax',y_pred_dev_argmax[:10])
        y = torch.max(torch.tensor(y), 1)[1]
        # print ('y',y[:10])
        accur+=accuracy(y_pred_dev_argmax, y)
        loss_dev_per_batch = criterion(y_pred_dev, y)
        loss_dev+=loss_dev_per_batch.item()
    f1_macro=calculate_f1(y_pred_dev_argmax, y)
    return accur,loss_dev,f1_macro

def filter_data(data_batch,y):
    y=[(i,item) for i,item in enumerate(y) if item[0]==1 or item[1]==1]
    # print (type(data_batch),data_batch.shape)
    item_lst,y=zip(*y)
    data_batch=data_batch[list(item_lst)]
    return data_batch,list(y)



def train(embedding_train_fname, embedding_dev_fname,device,args,labels2y,H,runs=5,embedding_test_fname=None):
    # dtype = torch.float
    STOP_EPOCH_NO=100
    train_file=h5py.File(embedding_train_fname,'r')
    embed_train=train_file['data']
    dev_file = h5py.File(embedding_dev_fname,'r')
    embed_dev = dev_file['data']

    if embedding_test_fname is not None:
        test_file = h5py.File(embedding_test_fname, 'r')
        embed_test = test_file['data']

    D_in, D_out = embed_train[0].shape[0], len(labels2y)

    for run in range(runs):
        model=TwoLayerNet(D_in,H,D_out)
        # print ('WEIGHTS',WEIGHTS)
        criterion = torch.nn.CrossEntropyLoss()
        accur_dev_best=0
        loss_dev_best=math.inf
        epoch_best=0
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        accur_train=0
        for t in range(args.epochs):
            loss_per_epoch=0
            for data_batch,y in generate_embed_labels_batch(args,embed_train,args.batchsize,labels2y,os.path.basename(embedding_train_fname).split('_')[0]):
                # Forward pass: Compute predicted y by passing x to the model
                # data_batch,y=filter_data(data_batch,y)
                y_pred = model(torch.tensor(data_batch).float())
                y_pred_train_argmax = torch.max(y_pred, 1)[1]

                # Compute and print loss
                y=torch.max(torch.tensor(y),1)[1]
                accur_train += accuracy(y_pred_train_argmax, y)

                loss=criterion(y_pred,y)
                loss_per_epoch+=loss.item()
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break
            print ('training loss per epoch: {0}, accur {2},epoch {1}'.format(loss_per_epoch,t,accur_train))
            accur_train=0
            accur_dev,loss_dev,f1_macro_dev=validate_per_epoch(embed_dev=embed_dev,args=args,criterion=criterion,labels2y=labels2y,model=model,part=os.path.basename(embedding_dev_fname).split('_')[0])
            print ('dev accuracy per epoch: {0}, loss is {1}'.format(accur_dev,loss_dev))

            if embedding_test_fname is not None:
                accur_test, loss_test,f1_macro_test = validate_per_epoch(embed_dev=embed_test, args=args, criterion=criterion,
                                                         labels2y=labels2y, model=model,
                                                         part=os.path.basename(embedding_test_fname).split('_')[0])

                print ('test accuracy per epoch: {0}, loss is {1}'.format(accur_test,loss_test))

            if t%args.save_every_n==0 and t>=args.save_every_n:
                # if f1_macro_dev>f1_macro_dev_best: #and loss_dev<loss_dev_best:
                if accur_dev > accur_dev_best and loss_dev<loss_dev_best:
                    torch.save(model.state_dict(), embedding_train_fname+'_run{2}_epoch{0}_accur{3}_f1macro{1}'.format(t,f1_macro_test,run,accur_dev))
                    if epoch_best>0:
                        os.remove(embedding_train_fname+'_run{2}_epoch{0}_accur{3}_f1macro{1}'.format(epoch_best,f1_macro_test_best,run,accur_dev_best))
                    epoch_best,accur_dev_best,loss_dev_best,f1_macro_test_best=t,accur_dev,loss_dev,f1_macro_test
                elif t-epoch_best> STOP_EPOCH_NO:
                    break



    train_file.close()
    dev_file.close()

def test_output(y_pred,args,LABELS,data_fname):
    CCR = ConllCorpusReader(root=args.data, fileids='.conll',
                            columntypes=('words', 'pos', 'ne', 'chunk'))
    print (data_fname)
    tagged_sents=CCR.tagged_sents(data_fname)
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
        pred_conll.append(list(zip(token_lst, tags_lst, tags_pred)))
    return pred_conll
            # f.write('\n'.join([' '.join(token) for token in zip(token_lst, tags_lst, tags_pred)]))
            # f.write('\n')

def output_conll_to_file(pred_conll,output_fname):
    with open(output_fname, 'w') as f:
        print (output_fname)
        f.write('\n\n'.join(['\n'.join(['	'.join(token) for token in sent]) for sent in pred_conll ]))



def test(path,test_data_embed_fname,H,labels2y):
    test_file = h5py.File(test_data_embed_fname,'r')
    embed_test=test_file['data']
    D_in, D_out = embed_test[0].shape[0], len(labels2y)

    model=TwoLayerNet(D_in,H,D_out)
    model.load_state_dict(torch.load(path))
    model.eval()

    for data_batch, y in generate_embed_labels_batch(args, embed_test, len(embed_test), labels2y, os.path.basename(test_data_embed_fname).split('_')[0]):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = torch.max(model(torch.tensor(data_batch).float()),1)[1]
        print (set([int(i)for i in y_pred]))
        y = torch.max(torch.tensor(y), 1)[1]
        print (len(y_pred),len(y),len(data_batch))
        print ('accuracy is',accuracy(y_pred,y))

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
    WEIGHTS=[660,548,264,140,221,142]
    WEIGHTS_SUM=float(sum(WEIGHTS))
    WEIGHTS=[(1-float(w/WEIGHTS_SUM)) for w in WEIGHTS]
    PARTITION = {'train': 'wnut17train.conll', 'test': 'emerging.test.annotated.modified', 'dev': 'emerging.dev.conll'}
    LABELS2Y=from_label_to_y(LABELS)
    start_time = time.time()
    Hidden_unit=50

    # 1. load args
    args = parse_args_ner(test_files=
                      {
                          'context2vec_param_file': '../models/context2vec/model_dir/MODEL-1B-300dim.params.10',
                       # 'skipgram_param_file': '../models/wiki_all.model/wiki_all.sent.split.model',
                       # 'w2salience_f': '../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab',
                       # 'matrix_f': '../models/ALaCarte/transform/wiki_all_transform.bin',
                          'elmo_param_file': [
                              "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                              "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"],

                          'data':'./eval_data/emerging_entities/',
                       'train_or_test':'train',
                       'model_type':CONTEXT2VEC_SUB_ELMO__ELMO_WITH_TARGET,
                       'output_dir':'./results/ner/',
                          'batchsize':10,
                          'lr':0.01,
                          'epochs':3000,
                          'n_runs':5,
                          'save_every_n':20000,
                          'gpu':-1,
                          'n_result':20,
                          'model_path':'./results/ner/train_skipgram_isf_wiki_all.sent.split.model.h5_run0_epoch5_accur418_f1macro0.11560737359897838'
                       })





    # 3. gpu setup
    device=gpu_device(args.gpu)
    xp = cuda.cupy if args.gpu >= 0 else np



    # 4. read embeddings and train

    if args.train_or_test=='train':
        print ('load data>>')

        train_data_embed_fname=write_embedding_tofile(args,'train')
        dev_data_embed_fname=write_embedding_tofile(args,'dev')
        test_data_embed_fname=write_embedding_tofile(args,'test')
        print ('train...')
                # train
        train(embedding_train_fname=train_data_embed_fname,embedding_test_fname=test_data_embed_fname, embedding_dev_fname=dev_data_embed_fname,device=device,args=args, labels2y=LABELS2Y,H=Hidden_unit,runs=args.n_runs)
        # validation
    # read embeddings and test
    elif args.train_or_test=='test':

        if args.model_type=='major_vote':
            y_pred=[0]*5000
            test_data_embed_fname=os.path.join(args.output_dir,'test')
            args.model_path='majorvote_majorvote_majorvote_majorvote'
        else:
            test_data_embed_fname=write_embedding_tofile(args,'test')
            y_pred=test(path=args.model_path,test_data_embed_fname=test_data_embed_fname,H=Hidden_unit,labels2y=LABELS2Y)
        pred_conll=test_output(y_pred=y_pred,args=args,LABELS=LABELS,data_fname=PARTITION['test'])
        output_conll_to_file(pred_conll,output_fname=test_data_embed_fname+'.'+args.model_path.split('_')[-4]+'.output')
        #test
        pass


    print ('processing ner sentences costs {0}'.format(time.time() - start_time))
