import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from dataloader import *
import os 
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import top_k_accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from scipy.special import softmax
import wandb


class Transformer:
    def __init__(self, checkpoint, num_labels, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, output_hidden_states=True)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def tokenise_text(self, example):
        return self.tokenizer(example['prompt'], padding=True, truncation=True, return_tensors='pt')
    
    def convert_label_to_one_hot(self, example):
        one_hot = np.zeros(len(self.model.config.id2label))
        one_hot[self.model.config.label2id[example['context']]] = 1
        example['labels'] = self.model.config.label2id[example['context']]
        return example



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # loss_fn
        loss_fn = CrossEntropyLoss()
        # get targets
        targets = inputs.pop('labels') # get labels and argmax, pop labels so model don't evaluate loss again
    
        outputs = model(**inputs)
        logits = outputs.get("logits") # get preds from outputs
        loss = loss_fn(logits,targets)

        return (loss, outputs) if return_outputs else loss
    

def compute_metrics(eval_pred_hidden):
    predictions, labels = eval_pred_hidden
    top_prediction = np.argmax(predictions[0],axis=-1)
  
    return {
        'acc': balanced_accuracy_score(labels,top_prediction),
        'top 5 acc': top_k_accuracy_score(labels,predictions[0], k=5),
        'f1': f1_score(labels,top_prediction,average='weighted'),
        'roc': roc_auc_score(labels,softmax(predictions[0], axis=-1), multi_class='ovr', average='macro')
        }

checkpoint_runid = {
    'bert-base-uncased': '3mmzx9ty',
    'distilbert-base-uncased':'1nncpbza', 
    'distilbert-base-uncased-finetuned-sst-2-english' :'2g5jjqh2',
    'bhadresh-savani/distilbert-base-uncased-emotion':'13ae8sug',
    'roberta-base':'bpucrcgp',
    'deepset/roberta-base-squad2':'2axmjc6a',
    'arpanghoshal/EmoRoBERTa':'2ny4fnn2',
    'sentence-transformers/all-MiniLM-L6-v2': 'cquhbtif',
    'sentence-transformers/all-mpnet-base-v2': '30pmwvby'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, default='roberta-base')
    parser.add_argument("--output-dir",  type=str, default='outputs')
    parser.add_argument("--data", type=str, default='empathetic-dialogues')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float,default=5e-5)
    parser.add_argument("--run-name", type=str,help='a discriptor for wanb run')
    parser.add_argument("--loss", default='binary-cross-entropy', type=str, help='loss function')
    parser.add_argument("--device", default='cpu', type=str, help='device to use')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--no-train", dest='train', action='store_false')
    parser.set_defaults(train=False)
    #parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    #parser.add_argument("--use-wanb-model", action=argparse.BooleanOptionalAction)

    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser.parse_args()

    device = torch.device(args.device)

    # check if there is a run for this checkpoint. If there is, resume. Else, start new run
    id = None
    name = f"{args.checkpoint}-{args.loss}"

    config = None
    if args.train:
        config = {
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "loss": args.loss
            }

    if args.checkpoint in checkpoint_runid.keys():
        print('run exists..resume previous run')
        id = checkpoint_runid[args.checkpoint]
        name = None
    run = wandb.init(
        project="emotion-vectors",
        name = name,
        id=id,
        group="affect classification",
        resume=True,
        config=config
        )

    #run = wandb.init(project='emotion-vectors',id='pbucrcgp', resume=True) # roberta experiment id
    # create dir for saving outputs
    output_dir = f"{args.output_dir}/{args.checkpoint}-finetuned-{args.data}"
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    # load model 
    if args.data == 'empathetic-dialogues':
        num_labels = 32
    transformer = Transformer(args.checkpoint, num_labels=num_labels,device=device)
    
    # get train and val set, store in dtct, pass dict to DatasetDict
    if args.data == 'empathetic-dialogues':
        dataset_dict = {split: EmpatheticDialoguesDataset(split=split,only_prompts=True) for split in ['train', 'valid','test']}  
        transformer.model.config.label2id = dataset_dict['test'].emotion2id    #store mappings in model
        transformer.model.config.id2label = dataset_dict['test'].id2emotion    #store mappings in model
        model_inputs = DatasetDict({split: Dataset.from_pandas(dataset.dataset)for split, dataset in dataset_dict.items()})
            
    else:
        NotImplementedError     #other datasets?
        
    model_inputs = model_inputs.map(transformer.tokenise_text,batched=True)
    model_inputs = model_inputs.map(transformer.convert_label_to_one_hot)
    model_inputs
    # set up trainer 
    training_args = TrainingArguments(
        output_dir= output_dir,
        evaluation_strategy = "epoch",
        save_strategy='epoch',
        load_best_model_at_end=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.n_epochs,
        weight_decay=0.01,
        #logging_steps=len(model_inputs["train"]) // args.batch_size,
        optim = 'adamw_torch',
        report_to = 'wandb',
        run_name= args.run_name,
        metric_for_best_model='f1'
        )
    if args.loss == 'cross-entropy':
        trainer = CustomTrainer(
            transformer.model,
            training_args,
            train_dataset=model_inputs["train"],
            eval_dataset=model_inputs["valid"],
            compute_metrics=compute_metrics,
            data_collator=transformer.data_collator
        )
    else:
        trainer = Trainer(
            transformer.model,
            training_args,
            train_dataset=model_inputs["train"],
            eval_dataset=model_inputs["valid"],
            compute_metrics=compute_metrics,
            data_collator=transformer.data_collator
        )
    #
    if args.train:
        print('training model...')
        trainer.train()
        print('training done')

    
    print('generating predictions...')
    predicton_output = trainer.predict(model_inputs['test'])
    predict_vectors, hidden_states = predicton_output.predictions
    np.save(f"{output_dir}/{args.data}-test-output.npy", predict_vectors)
    f = open(os.path.join(output_dir, 'model_hidden_states'),'wb')
    pickle.dump(hidden_states,f)
    # get vector representation from bert hidden states
    # all_hidden_layers = [torch.tensor(np.array(hidden_states)) for i in (-1, -2, -3, -4)]
    # cat_hidden_states = torch.cat(tuple(last_four_layers),-1)
    # embeddings = torch.mean(cat_hidden_states, dim=0).detach().numpy()
    #print(embeddings.shape)
    # get top 5 predictions, add to dataframe
    top5preds = torch.topk(torch.tensor(predict_vectors),k=5).indices.tolist()
    top5preds = [[transformer.model.config.id2label[val] for val in pred] for pred in top5preds]
    top5preds = pd.DataFrame(
        top5preds,
        columns=['top_1','top_2','top_3','top_4','top_5']
        )
    dataset_with_predicts = pd.concat([dataset_dict['test'].dataset,top5preds],axis=1)
    table = wandb.Table(dataframe=dataset_with_predicts)
    
    # connp.save(f"ed-test-prompts-arrays/test-prompt_embedding-{args.checkpoint}", embeddings)
    wandb.log({'test_with_top5_predictions':table})
    wandb.log(predicton_output.metrics)

    



