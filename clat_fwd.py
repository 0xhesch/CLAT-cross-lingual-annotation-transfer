import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from simalign import SentenceAligner
from tqdm import tqdm
from nltk.tokenize import word_tokenize

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

aligner = SentenceAligner(model="bert-base-multilingual-cased", token_type="bpe", matching_methods="i")

in_file = sys.argv[1]
out_file = sys.argv[2]
lang_code = sys.argv[3]

def translate_sentence(src_sentence):
	input_ids = tokenizer(src_sentence, return_tensors="pt", is_split_into_words=True)
	outputs = model.generate(
    **input_ids,
    forced_bos_token_id=tokenizer.get_lang_id(lang_code)
	)
	decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
	return word_tokenize(decoded[0])



def extract_entities(conll_file):
	ents =[]
	words = []
	
	sentences = [[]]
	sentences_ents = [[]]
	
	for line in conll_file:
		if '\t' in line:
			ents.append(line.split()[1])
			words.append(line.split()[0])
		else:
			ents.append('\n')
			words.append('\n')
	for i in words:
		if i == '\n':
			sentences.append([])
		else:
			sentences[-1].append(i)
	for j in ents:
		if j == '\n':
			sentences_ents.append([])
		else:
			sentences_ents[-1].append(j)
	return sentences, sentences_ents

def repair_bio_format(entities):
	for i, ents in enumerate(entities):
		prev_token = 'O'
		for j, ent in enumerate(ents):
			if ent != 'O' and prev_token == 'O':
				ents[j] = ents[j].replace('I-','B-')
			elif ent != 'O' and prev_token !='O':
				ents[j] = ents[j].replace('B-','I-')
			prev_token = ents[j]
	return entities
	
def save_target_conll(filename, sentences, entities):
	if len(sentences) != len(entities):
		print("Token-Entity mismatch!")
	else:
		outfile = open(filename, 'w')
		for i, sentence in enumerate(sentences):
			for j, token in enumerate(sentence):
				outfile.write(token + '\t' + entities[i][j] + '\n')
			outfile.write('\n')

conll_file = open(in_file, "r", errors='ignore').readlines()
sentences, ents = extract_entities(conll_file)

trg_sentences = []
trg_sentences_ents = []

print("Translating and aligning " + str(len(sentences)) + " sentences:")

for i, tokens in enumerate(tqdm(sentences)):
	src_sentence = tokens
	trg_sentence = translate_sentence(src_sentence)
	trg_sentences.append(trg_sentence)
	
	alignments = aligner.get_word_aligns(src_sentence, trg_sentence)
	for matching_method in alignments:
		#init target entities
		trg_ents = ["O" for i in range(len(trg_sentence))]
		#place aligned entities
		for j, word in enumerate(src_sentence):
			for tuple in alignments[matching_method]:
				src = tuple[0]
				trg = tuple[1]
				trg_ents[trg] = ents[i][src]
		trg_sentences_ents.append(trg_ents)
		bio_trg_sentences_ents = repair_bio_format(trg_sentences_ents)
		save_target_conll(out_file, trg_sentences, bio_trg_sentences_ents)

print("Saved output to " + out_file)