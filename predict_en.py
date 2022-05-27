from simpletransformers.ner import NERModel
import nltk
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

labels=[ 'B-TREAT', 'I-TREAT','B-MED', 'I-MED','B-DIAG', 'I-DIAG', 'O']

# open trained model from path
model = NERModel('bert','./model', labels=labels, use_cuda=True)

# open sentences to predict ner classes treat, med, diag and o
rows = open('./evaluate/randomSentSet1_en.txt', errors='ignore').readlines()

# create list of token list
to_predict = []
for sentence in rows:
    to_predict.append(nltk.word_tokenize(sentence))
    
# predict list of token list
preds, raw_outputs = model.predict(to_predict, split_on_space=False)

# save prediction to file
result = open('./evaluate/prediction_en.conll', 'w')

for sentence in preds:
    for token in sentence:
        value = list(token.values())[0]
        key = list(token.keys())[0]
        result.write(key + ' ' + value.strip() + '\n')
    result.write('\n')
    
print('Saved prediction to /evaluate/prediction_en.conll')