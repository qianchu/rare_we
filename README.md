# rare_we

1. requirements
   Option 1: Docker
   Dockerfile found in /Docker/Dockerfile
   Bash ./start
   source activate rare_we_clean
   
   Option 2: Conda
   Requirements can be found in /Docker/environment_rare_we_clean.yml
		

   You can load context2vec models from https://www.github.com/melamud/context2vec/

2. chimeras/nonce/crw evaluation

    Usage:eval_script.py [-h] [--f MODEL_PARAM_FILE] [--m MODEL_TYPE]
                      [--w WEIGHTS [WEIGHTS ...]] [--d DATA] [--g GPU]
                      [--ws W2SALIENCE_F] [--n_result N_RESULT] [--t TRIALS]
                      [--c CONTEXT_FLAG] [--ma MATRIX_F]

  

    WEIGHTS (integers in WEIGHT_DICT):
        
	WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ,5:TOP_CLUSTER_DENSITY, 6:SUBSTITUTE_PROB}

        TOP_MUTUAL_SIM='top_mutual_sim'
            measure the top n substitutes' mutual similarity (weighted by the top 1 substitutes' compatibility to context)
        
        TOP_CLUSTER_DENSITY='top_cluster_density'
            measure the top n substitutes' distance towards the cluster centroid (weighted by the top 1 substitutes' compatibility to context)

        LDA='lda'
            word salience measured by lda topic entropy 

        INVERSE_S_FREQ='inverse_s_freq'
            word salience measured by inverse sentence frequency

        INVERSE_W_FREQ='inverse_w_q'
            word salience measuerd by inverse word frequency

        SUBSTITUTE_PROB='substitute_prob'
            substitutes weighted by their compatibility to context



    MODEL_TYPE:
    
        skipgram (context skipgram input vector without stop words)
        
        context2vec (word embedding in context2vec space)
        
        context2vec-skipgram(context2vec substitutes in skipgram space)
        
        context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)


3. Context Evaluation in Rare and Emerging Entity Classification 

python eval_ner.py -h:
		   [--sm SKIPGRAM_PARAM_FILE]
                   [--elmo ELMO_PARAM_FILE [ELMO_PARAM_FILE ...]]
                   [--m MODEL_TYPE] [--d DATA] [--g GPU] [--ws W2SALIENCE_F]
                   [--n_result N_RESULT] [--ma MATRIX_F] [--tot TRAIN_OR_TEST]
                   [--output_dir OUTPUT_DIR] [--batchsize BATCHSIZE] [--lr LR]
                   [--ep EPOCHS] [--n SAVE_EVERY_N] [--run N_RUNS]
                   [--path MODEL_PATH]





