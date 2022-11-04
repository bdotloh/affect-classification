import numpy as np
import os
import pandas as pd
import pickle
import re
from torch.utils.data import Dataset as D


class EmpatheticDialoguesDataset(D):
    def __init__(self, data_path = 'data/empathetic-dialogues', split='train', only_prompts=False, prune_length=10):
        super().__init__()
        self.only_prompts = only_prompts

        self.dataset = self.process_dataset(data_path,split)

        if self.only_prompts:
            self.conv_ids = self.dataset.loc[:,'conv_id'].tolist()
        else:
            self.conv_ids = self.dataset.index.get_level_values(1).tolist()

        with open('data/emotion-ids.p','rb') as f:
            self.id2emotion = pickle.load(f)
            self.emotion2id =  {emotion:id for id, emotion in self.id2emotion.items()}
        
        with open('data/emotion-ids.p','rb') as f:
            self.id2emotion = pickle.load(f)
            self.emotion2id =  {emotion:id for id, emotion in self.id2emotion.items()}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, id):
        conv_id = self.conv_ids[id]
        if self.only_prompts:
            example = self.dataset[self.dataset['conv_id']==conv_id]
        else:
            example = self.dataset.xs(conv_id, level='conv_id')
        return example
        
    def process_dataset(self, data_path, split):
        col_list = ['conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx' ,'utterance']
        dataset = pd.read_csv(f"{os.path.join(data_path,split)}.csv", usecols=col_list)
        dataset['utterance'] = dataset.utterance.apply(lambda utterance: re.sub(r"_comma_", ",", utterance))
        dataset['prompt'] = dataset.prompt.apply(lambda prompt: re.sub(r"_comma_", ",", prompt))
        dataset=pd.pivot(
            dataset,    
            values= ["utterance","speaker_idx"], #values in column
            index=["context", "conv_id", "prompt"], 
            columns="utterance_idx"
            )

        if self.only_prompts:
            dataset = pd.DataFrame(dataset.index.values.tolist(), columns= ['context', 'conv_id','prompt'])
            dataset = dataset[dataset.prompt.str.len() > 10]

        return dataset
    
    def tokenize_text(self, example):
        return 

if __name__ == '__main__':
    dataset = EmpatheticDialoguesDataset()
    print(dataset[0])
