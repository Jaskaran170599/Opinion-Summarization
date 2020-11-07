# define utility tools (functions here) like rogue (metrics) , loss functions etc.


# Rouge 
from rouge import Rouge

rouge = Rouge(metrics = ['rouge-1','rouge-2', 'rouge-l'])


# Function to compare created summary with reference summary
def evaluateSummary(hypothesis, reference) :
    scores = rouge.get_scores(hypothesis, reference)
    new_score = {
        'Rouge-1' : scores[0]['rouge-1']['f'],
        'Rouge-2' : scores[0]['rouge-2']['f'],
        'Rouge-L' : scores[0]['rouge-l']['f'],
    }
    return new_score

# scores =  evaluateSummary(hypothesis, reference)