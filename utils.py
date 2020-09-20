# define utility tools (functions here) like rogue (metrics) , loss functions etc.


# Rouge 
from rouge import Rouge

rouge = Rouge(metrics = ['rouge-1','rouge-2', 'rouge-l'])


# Function to compare created summary with reference summary
def evaluateSummary(hypothesis, reference) :
    scores = rouge.get_scores(hypothesis, reference)
    return scores

# scores =  evaluateSummary(hypothesis, reference)