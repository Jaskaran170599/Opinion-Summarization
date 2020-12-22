# Opinion-Summarization
### The aim of this project is to create Abstract based opinion summarization using the power of transformers.

The study focuses on training different Transformers used in this project that is:
#### Stacking

Stacking the transformers encoder and decoder with pre-trained models.

#### Using pre-trained models

Using pre-trained transformers model which are trained as a whole on data ex. T5

It also compares the performance of different Transformer models based on the above training techniques. The models are Bert2Bert ,T5and Roberta2Roberta .

The pipeline is based on [Opinion Digest] (https://arxiv.org/abs/2005.01901) Framework

Model | Rouge-1 | Rouge-2 | Rouge-L | 
--- | --- | --- | --- |
Bert2Bert | 25.03239 | 4.05246 | 16.35603 |
Roberta2Roberta | 25.96 | 4.08 | 16.66 |
T5 | 27.97 | 4.6 | 17.1 |
