# rare_we

1. requirements
   Option 1: Docker
   Dockerfile found in /Docker/Dockerfile
   Bash ./start
   source activate rare_we_clean
   
   Option 2: Conda
   Requirements can be found in /Docker/environment_rare_we_clean.yml
		

   You can load context2vec models from https://www.github.com/melamud/context2vec/
   You can load skipgram models from https://github.com/minimalparts/nonce2vec

2. chimeras/nonce/crw evaluation

    Usage:eval_script.py [-h] [--f MODEL_PARAM_FILE] [--m MODEL_TYPE]
                      [--w WEIGHTS [WEIGHTS ...]] [--d DATA] [--g GPU]
                      [--ws W2SALIENCE_F] [--n_result N_RESULT]
                     

  

    WEIGHTS (integers in WEIGHT_DICT):
        
	WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ,5:TOP_CLUSTER_DENSITY, 6:SUBSTITUTE_PROB}

        For our experiments, choose 0 or 3 for skipgram; choose 0 for the substitutes-based method



    MODEL_TYPE:
    
        skipgram (context skipgram input vector without stop words)
        
        context2vec (word embedding in context2vec space)
        
        context2vec-skipgram(context2vec substitutes in skipgram space) (the substitute method)
        
        context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)







