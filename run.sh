declare -a checkpoints=(
    #'bert-base-uncased'
    #'finiteautomata/bertweet-base-sentiment-analysis'
    'distilbert-base-uncased' 
    'distilbert-base-uncased-finetuned-sst-2-english' 
    'bhadresh-savani/distilbert-base-uncased-emotion'
    'roberta-base'
    'deepset/roberta-base-squad2'
    'arpanghoshal/EmoRoBERTa',
    'sentence-transformers/all-MiniLM-L6-v2'
    'sentence-transformers/all-mpnet-base-v2'
    )

for checkpoint in "${checkpoints[@]}"
do
    python main.py --checkpoint "$checkpoint"
done

