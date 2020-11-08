import os
import pandas as pd
import nltk
import json
from tqdm import tqdm

from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split

from configs import config


class prepare:
    
    def __init__(self,path):
        
        self.path=path
        self.min_ext=config.MIN_EXT
        self.max_ext=config.MAX_EXT
        self.min_sent=config.MIN_SENT
        self.max_sent=config.MAX_SENT
        self.max_len=config.MAX_LEN
        self.target_path=config.TARGET_PATH
        self.ratio=config.SPLIT
        self.detokenizer=TreebankWordDetokenizer()
        self.aspects2embed=config.ASPECTS2EMBED
    
    def _split(self,data):
        
        train,test=train_test_split(data,random_state=5,test_size=self.ratio[2])
        train,val=train_test_split(train,random_state=5,test_size=self.ratio[1])
        return train,val,test
    
    def _read_file1(self):
        
        data=[]
        
        with open(self.path, "r", encoding="utf-8") as file:
            for _, line in enumerate(tqdm(file, desc="reviews")):
                review = json.loads(str(line))
                sents = review["sentences"]
                exts = review["extractions"]
                
                if len(sents) < self.min_sent or len(sents) > self.max_sent:
                    continue
                
                if len(exts) < self.min_ext or len(exts) > self.max_ext:
                    continue
                
                sents = [self.detokenizer.detokenize(toks) for toks in sents]
                
                sid=[]
                phrases=[]
                attributes=[]
                sentiments=[]
                
                for ext in review["extractions"]:
                    if not ('opinion' in ext.keys() and 'aspect' in ext.keys()):
                        continue
                    opinion = ext["opinion"]
                    aspect = ext["aspect"]
                    attrib = ext['attribute']
                    sentim = ext['sentiment']
                    phrases.append(opinion+' '+aspect)
                    sid.append(ext["sid"])
                    sentiments.append(sentim)
                    attributes.append(attrib)
                if(len(sid)==0):
                    continue
                data.append((' [SEP] '.join(phrases),' '.join([sents[i] for i in sid]),','.join(attributes),
                             ','.join(sentiments),[sents[i] for i in sid]))
                
        return pd.DataFrame(data,columns=['phrases','text','attributes','sentiments','sents'])

    def _add_aspect_embedding(self,data):

        data_emb=[]

        for _, row in tqdm(data.iterrows(), desc="adding embeddings"):
            aspects=row['attributes'].split(',')
            sentim=row['sentiments'].split(',')
            sents=row['sents']
            phrases=row['phrases']
            aspects=[self.aspects2embed[asp] for asp in aspects]

            rev={}

            for n,asp in enumerate(aspects):
                if asp not in rev.keys():
                    rev[asp]=(sents[n],sentim[n])
                else:
                    rev[asp]=(rev[asp][0]+' '+sents[n],rev[asp][1]+','+sentim[n])

            for asp in rev.keys():
                data_emb.append((asp+' [SEP] '+phrases,rev[asp][0],asp,rev[asp][1]))

            data_emb.append(('all'+' [SEP] '+phrases,row['text'],row['attributes'],row['sentiments']))
        
        return pd.DataFrame(data_emb,columns=['phrases','text','attributes','sentiments'])    
    
    def run(self,method=0):
        
        data=self._read_file1()
        
        train,val,test=self._split(data)
        
        train.to_csv(os.path.join(self.target_path,'train%d.csv'%(0)))
        val.to_csv(os.path.join(self.target_path,'val%d.csv'%(0)))
        test.to_csv(os.path.join(self.target_path,'test.csv'%(0)))
        
        
        if method==1:
            train=self._add_aspect_embedding(train)
            val=self._add_aspect_embedding(val)
        
            train.to_csv(os.path.join(self.target_path,'train%d.csv'%(method)))
            val.to_csv(os.path.join(self.target_path,'val%d.csv'%(method)))
            