from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
from sentence_transformers import CrossEncoder

similar_model = CrossEncoder("similarity-minilm/checkpoint-20216", num_labels=2)

sen_1 = input("Enter Sentence 1: ")
sen_2 = input("Enter Sentence 2: ")


similar_output = similar_model.predict([(sen_1, sen_2)])
is_similar = similar_output[0][0] < similar_output[0][1]

if is_similar:
    print("Similar Sentences")
else:
    print("Non Similar Sentences")

import torch
device = torch.device('cuda')
safety_model = AutoModelForSequenceClassification.from_pretrained("sentence-safety-deberta/checkpoint-46000", num_labels=2).to(device)
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
sen_1_tokenized = tokenizer(sen_1, return_tensors='pt').to(device)
sen_2_tokenized = tokenizer(sen_2, return_tensors='pt').to(device)

sen_1_output, sen_2_output = safety_model(**sen_1_tokenized).logits, safety_model(**sen_2_tokenized).logits
is_sen_1_safe = int(sen_1_output[0][0] > sen_1_output[0][1])
is_sen_2_safe = int(sen_2_output[0][0] > sen_2_output[0][1])

print(sen_1_output, sen_2_output)

if is_sen_1_safe:
    print("Sentence 1 safe")
else:
    print("Sentence 1 unsafe")

if is_sen_2_safe:
    print("Sentence 2 safe")
else:
    print("Sentence 2 unsafe")

