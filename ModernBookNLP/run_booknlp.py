from booknlp.english.english_booknlp import EnglishBookNLP
import os
from tqdm.auto import tqdm
import sys


if __name__ == "__main__":
    NOVELS = set([i.replace('.txt', '') for i in os.listdir('data/litbank_speaker_input/') if '.txt' in i])

    model_params={
                "pipeline":"entity,quote,coref", 
                "model":"big"
            }
    
    if sys.argv[1] == 'booknlp' : 
        tgt_path = 'data/booknlp_litbank_out/'
    elif sys.argv[1] == 'joint' : 
        tgt_path = 'data/joint_litbank_out/'
        model_params={
                "pipeline":"entity,quote,coref", 
                "model":"big",
                "modern_qa":True
            }
    elif sys.argv[1] == 'direct' :
        model_params={
                "pipeline":"entity,quote,coref", 
                "model":"big",
                "modern_qa":True,
                'direct_qa': True
            }
        tgt_path = 'data/direct_litbank_out/'
    else : 
        raise ValueError('Unknown mode, choose from booknlp, joint, direct')
    
    bnlp = EnglishBookNLP(model_params)

    for n in tqdm(NOVELS) : 
        print(n)
        
        # Input file to process
        input_file=f'data/litbank_speaker_input/{n}.txt'
        # Output directory to store resulting files in
        output_directory=f"{tgt_path}/{n}/"
        os.makedirs(output_directory, exist_ok=True)
        
        bnlp.process(input_file, output_directory, n)

        print('\n\n\n')