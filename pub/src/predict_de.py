from simpletransformers.ner import NERModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

labels=[ 'B-TREAT', 'I-TREAT','B-MED', 'I-MED','B-DIAG', 'I-DIAG', 'O']

# Load NERModel
model = NERModel('bert','./model_de', labels=labels, use_cuda=True)

to_predict = []

rows = open("./conllIOBTags/randomSentSet1.CONLL", errors='ignore').readlines()

sentence_tokens = []
subsentence_tokens = []
counter = 0

for line in rows:
    if line == '\n':
        sentence_tokens.append(subsentence_tokens)
        subsentence_tokens = []
        counter = counter + 1
    else:
        subsentence_tokens.append(line.split('\t')[0])

# predict list of token list
preds, raw_outputs = model.predict(sentence_tokens, split_on_space=False)

# save prediction to file
result = open('./evaluate/prediction_de.conll', 'w')

for i, sentence in enumerate(sentence_tokens):
    for j, token in enumerate(sentence):
        try:
            value = list(preds[i][j].values())[0]
            key = list(preds[i][j].keys())[0]
            result.write(key.strip() + ' ' + value.strip() + '\n')
        except:
            result.write(token + ' ' + 'O' + '\n')
    result.write('\n')
    
print('Saved test prediction to /evaluate/prediction_de.conll')
