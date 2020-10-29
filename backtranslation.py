
import torch
import json

!pip install fastBPE regex requests sacremoses subword_nmt

# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

def paraphrase_back_translation_de2en(answer):
  paraphrase1 = de2en.translate(en2de.translate(answer))
  return paraphrase1

# Opening input JSON file 
f = open('/content/drive/My Drive/Colab Notebooks/VQUANDA-master/dataset/vquanda-dataset-extended-complete.json') 

# returns JSON object as a dictionary 
data = json.load(f) 
paraphrased_data = {}
final_data = []

def write_json(final_data):
    final_data.sort(key=lambda x: int(x["uid"]))
    with open('/content/drive/My Drive/Colab Notebooks/VQUANDA-master/dataset/paraphrased.json', 'w') as json_file:
        json.dump(final_data, json_file)

for ans in data:
  #print(ans)
  ANSWER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_de2en(ans['verbalized_answer'])
  GENDER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_de2en(ans['gender_verbalization_template'])
  NER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_de2en(ans['ner_verbalization_template'])
  ASK_PARAPHRASED_TEMPLATE = paraphrase_back_translation_de2en(ans['ask_verbalization_template'])
  paraphrased_data['uid'] = ans['_id']
  paraphrased_data['question'] = ans['question']
  paraphrased_data['verbalized_answer'] = ans['verbalized_answer']
  paraphrased_data['verbalized_paraphrased_answer'] = ANSWER_PARAPHRASED_TEMPLATE
  paraphrased_data['gender_verbalization_template'] = ans['gender_verbalization_template']
  paraphrased_data['gender_paraphrased_answer'] = GENDER_PARAPHRASED_TEMPLATE
  paraphrased_data['ner_verbalization_template'] = ans['ner_verbalization_template']
  paraphrased_data['ner_paraphrased_answer'] = NER_PARAPHRASED_TEMPLATE
  paraphrased_data['ask_verbalization_template'] = ans['ask_verbalization_template']
  paraphrased_data['ask_paraphrased_answer'] = ASK_PARAPHRASED_TEMPLATE
  paraphrased_data['query'] = ans['sparql_query']
  final_data.append(paraphrased_data)
  #print (ANSWER_PARAPHRASED_TEMPLATE)
  paraphrased_data = {}
write_json(final_data)

# Round-trip translations between English and Russian:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
def paraphrase_back_translation_ru2en(answer):
  paraphrase2 = ru2en.translate(en2ru.translate(answer))
  return paraphrase2

# Opening input JSON file 
f = open('/content/drive/My Drive/Colab Notebooks/VQUANDA-master/dataset/paraphrased-1000.json') 

# returns JSON object as a dictionary 
data = json.load(f) 
paraphrased_data = {}
final_data = []

def write_json(final_data):
    final_data.sort(key=lambda x: int(x["uid"]))
    with open('/content/drive/My Drive/Colab Notebooks/Data Output/paraphrased-1000.json', 'w') as json_file:
        json.dump(final_data, json_file)
for ans in data:
  #print(ans)
  verbalized_ANSWER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_ru2en(ans['verbalized_paraphrased_answer'])
  gender_ANSWER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_ru2en(ans['gender_paraphrased_answer'])
  ner_ANSWER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_ru2en(ans['ner_paraphrased_answer'])
  ask_ANSWER_PARAPHRASED_TEMPLATE = paraphrase_back_translation_ru2en(ans['ask_paraphrased_answer'])
  paraphrased_data['uid'] = ans['uid']
  paraphrased_data['question'] = ans['question']

  paraphrased_data['verbalized_en_ru_answer'] = verbalized_ANSWER_PARAPHRASED_TEMPLATE
  paraphrased_data['gender_en_ru_answer'] = gender_ANSWER_PARAPHRASED_TEMPLATE
  paraphrased_data['ner_en_ru_answer'] = ner_ANSWER_PARAPHRASED_TEMPLATE
  paraphrased_data['ask_en_ru_answer'] = ask_ANSWER_PARAPHRASED_TEMPLATE
  paraphrased_data['query'] = ans['query']
  final_data.append(paraphrased_data)
  paraphrased_data = {}
write_json(final_data)