from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from tqdm import tqdm
mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

#read test file from /textFiles/
de_text = open('./textFiles/randomSentSet1.txt', errors='ignore').readlines()

#save test file translation to /evaluate/
en_text = open('./evaluate/randomSentSet1_en.txt', 'w')

for sentence in tqdm(de_text):
    input = sentence
    input_ids = tokenizer.encode(input, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.strip() is "":
        en_text.write(input.strip() + '\n')
    else:
        en_text.write(decoded.strip() + '\n')
