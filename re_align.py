import nltk
from simalign import SentenceAligner
from tqdm import tqdm
from collections import defaultdict

myaligner = SentenceAligner(model="bert-base-multilingual-cased", token_type="bpe", matching_methods="m")

#source
prediction = open('./evaluate/prediction_en.conll', errors='ignore').readlines()

#target 
truth = open('./conllIOBTags/randomSentSet1.CONLL', errors='ignore').readlines()

outfile = open('./evaluate/prediction_de_aligned.conll', 'w')

pred_sentence_tokens = []
pred_sentence_tokens_ann = []

counter = 0

#load prediction tokens and token ann
pred_subsentence_tokens = []
pred_subsentence_tokens_ann = []

for line in prediction:
    if line == '\n':
        pred_sentence_tokens.append(pred_subsentence_tokens)
        pred_sentence_tokens_ann.append(pred_subsentence_tokens_ann)
        pred_subsentence_tokens = []
        pred_subsentence_tokens_ann = []
        counter = counter + 1
    else:
        pred_subsentence_tokens.append(line.split(' ')[0])
        pred_subsentence_tokens_ann.append(line.split(' ')[1].strip())
pred_sentence_tokens.append(pred_subsentence_tokens)
pred_sentence_tokens_ann.append(pred_subsentence_tokens_ann) 

truth_sentence_tokens = []

counter = 0

#load truth tokens for alignment
truth_subsentence_tokens = []

for line in truth:
    if line == '\n':
        truth_sentence_tokens.append(truth_subsentence_tokens)
        truth_subsentence_tokens = []
        counter = counter + 1
    else:
        truth_subsentence_tokens.append(line.split('\t')[0])
truth_sentence_tokens.append(truth_subsentence_tokens)

prev_token = 'O'
#go through english prediction sentences and align annotations with german tokens
for i, sentence in enumerate(tqdm(pred_sentence_tokens)):
    try:
        src_sentence = pred_sentence_tokens[i]
        trg_sentence = truth_sentence_tokens[i]
        alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)
       
        target_annotations = []
        target_entity = defaultdict(str)
        for word_index, words in enumerate(pred_sentence_tokens[i]):
            if 'DIAG' in pred_sentence_tokens_ann[i][word_index] or 'TREAT' in pred_sentence_tokens_ann[i][word_index] or 'MED' in pred_sentence_tokens_ann[i][word_index]:
                #print("trying to align " + pred_sentence_tokens[i][word_index])
                for matching_method in alignments:
                    for tuple in alignments[matching_method]:
                        #check for locations of annotated word in target language
                        if tuple[0] == word_index:
                            target_annotations.append(tuple[1])
                            target_entity[tuple[1]] = pred_sentence_tokens_ann[i][word_index]
                    break
            else:
                continue
        
        for index, token in enumerate(trg_sentence):
            if index in target_annotations:
                if prev_token is "B":
                    if 'DIAG' in target_entity[index]:
                        outfile.write(token + " I-DIAG" + '\n')
                    if 'TREAT' in target_entity[index]:
                        outfile.write(token + " I-TREAT" + '\n')
                    if 'MED' in target_entity[index]:
                        outfile.write(token + " I-MED" + '\n')
                else:
                    if 'DIAG' in target_entity[index]:
                        outfile.write(token + " B-DIAG" + '\n')
                    if 'TREAT' in target_entity[index]:
                        outfile.write(token + " B-TREAT" + '\n')
                    if 'MED' in target_entity[index]:
                        outfile.write(token + " B-MED" + '\n')
                prev_token = "B"
            else:
                outfile.write(token + " O" + '\n')
                prev_token = "O"
        outfile.write('\n')
        prev_token = 'O'
    except:
        print("End of file reached")
        
print("Saved test prediction to /evaluate/prediction_de_aligned.conll")