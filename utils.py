# define utility tools (functions here) like rogue (metrics) , loss functions etc.


# Rouge 
from rouge import Rouge

rouge = Rouge(metrics = ['rouge-1','rouge-2', 'rouge-l'])

def evaluateSummary(hypothesis, reference) :
    
    n = len(hypothesis)
    i = 0
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    
    for i in range(n) :
        
        scores = rouge.get_scores(hypothesis[i], reference[i])
        rouge_1 = rouge_1 + scores[0]['rouge-1']['f']
        rouge_2 = rouge_2 + scores[0]['rouge-2']['f']
        rouge_l = rouge_l + scores[0]['rouge-l']['f']
        
    new_score = {
        'Rouge-1' : rouge_1/n,
        'Rouge-2' : rouge_2/n,
        'Rouge-L' : rouge_l/n,
    }
    
    return new_score


def generate_output(model,input_ids):
    
    if not config.BEAM:
        output=model.generate(input_ids=input_ids,attention_mask=att_mask,min_length=10,
                               early_stopping=True,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id
                               ,eos_token_id=tok.eos_token_id)
    else:
        output=model.generate(input_ids=input_ids,attention_mask=att_mask,min_length=10,
                               early_stopping=True,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id
                               ,eos_token_id=tok.eos_token_id,num_beams=config.NBEAM)
    
    return config.TOKENIZER.decode(output[0], skip_special_tokens=True)