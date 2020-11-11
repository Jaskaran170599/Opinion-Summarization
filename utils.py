# define utility tools (functions here) like rogue (metrics) , loss functions etc.


# Rouge 
from rouge import Rouge

rouge = Rouge(metrics = ['rouge-1','rouge-2', 'rouge-l'])

def evaluateSummary(hypothesis, reference) :
    scores = rouge.get_scores(hypothesis, reference)
    new_score = {
        'Rouge-1' : scores[0]['rouge-1']['f'],
        'Rouge-2' : scores[0]['rouge-2']['f'],
        'Rouge-L' : scores[0]['rouge-l']['f'],
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