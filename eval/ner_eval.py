__author__ = 'qianchu_liu'


from Context_represent import *
from nltk.corpus.reader.conll import ConllCorpusReader
from copy import deepcopy
import os
import h5py
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


def convert_to_masked_wordlist(tagged_sent,tags_position):
    words_lst=list(zip(*tagged_sent)[0])
    for tag_position in tags_position:
        word_lst_masked=deepcopy(words_lst)
        word_lst_masked[tag_position[0]:tag_position[1]]='_'
        word_lst_masked[tag_position[0]] = TARGET_W
        yield word_lst_masked


def process_sent(tagged_sent):
    tags_position=return_tag_position(tagged_sent)
    for masked_wordlist in convert_to_masked_wordlist(tagged_sent,tags_position):
        yield masked_wordlist


def process_tagged_sents(tagged_sents):
    masked_word_lsts=[]
    for tagged_sent in tagged_sents:
        for masked_word_lst in process_sent(tagged_sent):
            masked_word_lsts.append(masked_word_lst)
    return masked_word_lsts

def load_ner(root,filename):
    CCR = ConllCorpusReader(root=root, fileids='.conll',
                            columntypes=('words', 'pos', 'ne', 'chunk'))
    masked_word_lsts = process_tagged_sents(CCR.tagged_sents(filename))
    return masked_word_lsts

def check_embed_file_exist(parsed_args):
    fn_lst = os.path.join(parsed_args.output_dir,'_'.join([str(fn[1]).split('/')[-1] for fn in sorted(parsed_args.__dict__.items()) if fn[1] is not None]))
    embedding_fname = fn_lst+'.h5'
    if os.path.isfile(embedding_fname):
        return embedding_fname,True
    else:
        return embedding_fname,False




def compute_rep_chunk(data,CM, chunksize):
    context_reps=[]

    counter=0
    for sent in data:
        print (sent)
        context_rep=CM.compute_context_reps_ensemble([' '.join(sent)], TARGET_W)
        if context_rep is not None:
            context_reps.append(context_rep)
            counter+=1
            if counter>=chunksize:
                yield as_numpy(context_reps)
                context_reps=[]
                counter=0
    if context_reps:
        yield as_numpy(context_reps)



def increment_write_h5py(hf,chunk,data_name='data'):
    if data_name not in hf:
        maxshape = (None,) + chunk.shape[1:]
        data=hf.create_dataset(data_name,data=chunk,chunks=chunk.shape,maxshape=maxshape,compression="gzip",compression_opts=9)

    else:
        data=hf[data_name]
        data.resize((chunk.shape[0]+data.shape[0],)+data.shape[1:])
        data[-chunk.shape[0]:]=chunk

def write_embedding_tofile(CM,embedding_fname,data,chunksize) :



    hf = h5py.File(embedding_fname,'w')
    for chunk in compute_rep_chunk(data, CM, chunksize):
        increment_write_h5py(hf, chunk)


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
        args = parser.parse_args()
        print args

    return args

if __name__ == "__main__":



    # 1. load args
    args = parse_args_ner(test_files=
                      {
                          # 'context2vec_param_file': '../models/context2vec/model_dir/MODEL-wiki.params.14',
                       'skipgram_param_file': '../models/wiki_all.model/wiki_all.sent.split.model',
                       # 'ws_f': '../corpora/corpora/WWC_norarew.txt.tokenized.vocab',
                       # 'matrix_f': '../models/ALaCarte/transform/wiki_all_transform.bin',
                       'data':'./eval_data/emerging_entities/wnut17train.conll',
                       'train_or_test':'train',
                       'model_type':SKIPGRAM,
                       'output_dir':'./results/ner/'
                       })
    # 2. load data
    data_masked_ner_lsts=load_ner(root=os.path.dirname(args.data),filename=os.path.basename(args.data))

    # 3. gpu setup

    gpu_config(args.gpu)
    xp = cuda.cupy if args.gpu >= 0 else np

    # 4. load embedding file or model
    embedding_fname,embedding_file_exist=check_embed_file_exist(args)
    if not embedding_file_exist:
        print('processing ner sentences')
        import time
        start_time = time.time()
        context2vec_modelreader, model_skipgram = load_model_fromfile(
            skipgram_param_file=args.skipgram_param_file,
            context2vec_param_file=args.context2vec_param_file, gpu=args.gpu)
        CM = ContextModel(model_type=args.model_type, context2vec_modelreader=context2vec_modelreader,
                          skipgram_model=model_skipgram, n_result=args.n_result, ws_f=args.w2salience_f,
                          matrix_f=args.matrix_f)
        write_embedding_tofile(CM=CM,embedding_fname=embedding_fname,data=data_masked_ner_lsts,chunksize=200)
        print ('processing ner sentences costs {0}'.format(time.time() - start_time))


    # read embeddings and train
    if args.train_or_test=='train':
        pass
    # read embeddings and test
    elif args.train_or_test=='test':
        pass

