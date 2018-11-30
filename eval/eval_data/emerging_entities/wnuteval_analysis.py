__author__ = 'qianchu_liu'


from eval.eval_data.emerging_entities.wnuteval import *
import sys
import os
from collections import defaultdict

def produce_entities_per_sys(fname):
    entities_gold=[]
    with open(fname, 'r') as f:
        lines = [line for line in f]
        entities_pred = doc_to_entities(lines)['sys_1']
        if entities_gold==[]:
            entities_gold = doc_to_entities(lines)['gold']

    return [entity for entity in entities_pred if entity.tag!='O'],[entity for entity in entities_gold if entity.tag!='O']

def produce_entities_across_sys():
    sys_2_pred_entities = {}
    gold_entities=[]
    for fname in sys.argv[1:]:
        pred_entities, gold_entities= produce_entities_per_sys(fname)
        sys_2_pred_entities[fname]= pred_entities
    return sys_2_pred_entities,gold_entities

def compare_sys_analysis(sys_2_pred_entities,gold_entities):
    sys2score=defaultdict(list)
    for i in range(len(gold_entities)):
        for fname in sys_2_pred_entities:
            if gold_entities[i]==sys_2_pred_entities[fname][i]:
                sys2score[fname].append('1')
            else:
                sys2score[fname].append('0')
    return sys2score

def print_sys2score(sys2score,dirname):
    sysnames=sys2score.keys()
    with open(os.path.join(dirname,'sys_analysis'),'w') as f:
         f.write('\t'.join([os.path.basename(sysname) for sysname in sysnames ]))
         sys_score_pairs=list(zip(*[sys2score[sys] for sys in sysnames]))
         f.write('\n'.join(['\t'.join(pair) for pair in sys_score_pairs]))

def main():
    #get tokens and entities across systems
    dirname=os.path.dirname(sys.argv[1])
    sys_2_pred_entities,gold_entities=produce_entities_across_sys()
    sys2score=compare_sys_analysis(sys_2_pred_entities, gold_entities)
    print_sys2score(sys2score,dirname)







if __name__ == '__main__':
    main()