# Create basic verbalization templates of LC-QuAD dataset
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import json
import torch
import re
import csv
import random
import spacy 
import torch
import nltk
from textblob import TextBlob
from spacy import displacy
from collections import Counter
import en_core_web_sm
import spacy
nlp = spacy.load("en")
nlp_ner_count = en_core_web_sm.load()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp_gender = en_core_web_sm.load()

# define paths
#ROOT_PATH = Path(os.path.dirname(__file__))
JSON_TEMPLATES_FILE = '/content/drive/My Drive/Colab Notebooks/LC-QuAD/resources/templates.json'
JSON_TRAIN_DATA_FILE = '/content/drive/My Drive/Colab Notebooks/LC-QuAD/train-data.json'
JSON_TEST_DATA_FILE = '/content/drive/My Drive/Colab Notebooks/LC-QuAD/test-data.json'
#QUESTION_TEMPLATE_FILE = ROOT_PATH / 'question_template.csv'
NEW_JSON_PATH = '/content/drive/My Drive/Colab Notebooks/Barshana_Outputs/vquanda_v4.json'

# define helper strings
TYPE = 'type'
VANILLA = 'vanilla'
COUNT = 'count'
ASK = 'ask'
ID = 'id'
DATA_ID = '_id'
SPARQL_TEMPLATE_ID = 'sparql_template_id'
SPARQL_QUERY_ANSWER = 'sparql_query_answer'
SPARQL_QUERY = 'sparql_query'
ANSWER_VERBALIZATION_TEMPLATE = 'answer_verbalization_template'
ANSWER_ASK_TEMPLATE_2 = 'ask_verbalization_template'    # For ASK questions
ANSWER_COUNT_TEMPLATE_2 = 'ner_verbalization_template' # For COUNT questions
ANSWER_GENDER_TEMPLATE_2 = 'gender_verbalization_template' # For VANILLA questions
ANSWER_PARAPHRASED_TEMPLATE = 'answer_paraphrased_template'
INTERMIDIATE_QUESTION_TEMPLATE = 'intermediary_question'
CORRECTED_QUESTION_TEMPLATE = 'corrected_question'
SPARQL_ENDPOINT_PATH = 'http://kddste.sda.tech/dbpedia201604'
ANSWER = 'answer'
BE_VERB_PLURAL = 'are'
BE_VERB_SINGULAR = 'is'
MAX_ALLOWED_QUERY_RESULTS = 15

# define helper lists
VANILLA_TYPE_ID = []
COUNT_TYPE_ID = []
ASK_TYPE_ID = []
LCQUAD_DATA = []
FINAL_VERBALIZED_DATA = []
VANILLA_QUESTIONS_PATTERNS = []
COUNT_QUESTION_PATTERNS = []
ASK_QUESTION_PATTERNS = []

# initialize sparql endpoint
SPARQL_ENDPOINT = SPARQLWrapper(SPARQL_ENDPOINT_PATH)
RETURN_FORMAT = JSON
SPARQL_ENDPOINT_PATH_FOR_GENDER = 'https://dbpedia.org/sparql'
# initialize sparql endpoint
SPARQL_ENDPOINT_FOR_GENDER = SPARQLWrapper(SPARQL_ENDPOINT_PATH_FOR_GENDER)

named_entity = {'ORG': 'the Organisation',
                'PRODUCT': 'the product',
                'LOC': 'location',
                'PERSON':'the person',
                'GPE': 'the country',
                'WORK_OF_ART': 'others',
                'NORP': 'others',
                'FAC': 'others',
                'EVENT': 'others',
                'WORK_OF_ART': 'others',
                'LAW': 'others',
                'LANGUAGE': 'others',
                'DATE': 'others',
                'TIME': 'others',
                'PERCENT': 'others',
                'MONEY': 'others',
                'QUANTITY': 'others',
                'ORDINAL': 'others',
                'CARDINAL': 'others'}

# helper functions
def read_template_file():
    with open(JSON_TEMPLATES_FILE) as json_file:
        for template in json.load(json_file):
            if template[TYPE] == VANILLA: VANILLA_TYPE_ID.append(template[ID])
            if template[TYPE] == COUNT: COUNT_TYPE_ID.append(template[ID])
            if template[TYPE] == ASK: ASK_TYPE_ID.append(template[ID])
        # Somehow template ids are missing from file, we add them manually
        VANILLA_TYPE_ID.extend([11, 605, 906, 601])
        VANILLA_TYPE_ID.sort()
        COUNT_TYPE_ID.sort()
        ASK_TYPE_ID.sort()

def read_lcquad_data():
    for data_file in [JSON_TRAIN_DATA_FILE, JSON_TEST_DATA_FILE]:
        with open(data_file) as json_file:
            for data in json.load(json_file):
                LCQUAD_DATA.append(data)

# For vanilla templates we provide 2 initial verbalized answers
# 1. Give first the answer and then fraction of the question that has all triple information
# 2. First provide triple information and at the end show the answer
def generate_verbalization_template():
    for data in LCQUAD_DATA:
        data[SPARQL_QUERY_ANSWER] = get_sparql_query_answer(data[SPARQL_QUERY])
        if data[SPARQL_TEMPLATE_ID] in VANILLA_TYPE_ID: data = process_vanilla_template(data)
        if data[SPARQL_TEMPLATE_ID] in COUNT_TYPE_ID: data = process_count_template(data)
        if data[SPARQL_TEMPLATE_ID] in ASK_TYPE_ID: data = process_ask_template(data)
        #print(data)
        FINAL_VERBALIZED_DATA.append(data)

# Sparql for gender
def get_sparql_query_gender(query):
    try:
        SPARQL_ENDPOINT_FOR_GENDER.setQuery(query)
        SPARQL_ENDPOINT_FOR_GENDER.setReturnFormat(RETURN_FORMAT)
        gender = SPARQL_ENDPOINT_FOR_GENDER.query().convert()
        #print('SPARQL QUERY RESPONSE: ', gender)
        return gender
    except Exception as e:
        #print(f"Unable to Send query to {SPARQL_ENDPOINT_FOR_GENDER.endpoint}")
        #print(str(e))
        return None

# recognise gender
def gender_recognition(query):
  query_list = query.split(' ')
  #print(query_list)
  for obj in query_list:
    if 'resource' in obj:
      resource = obj
      break
  query = 'SELECT ?gender WHERE { ' + str(resource) +' <http://xmlns.com/foaf/0.1/gender> ?gender }'
  respo = get_sparql_query_gender(query)['results']['bindings']
  if len(respo) != 0:
    final_gender = respo[0]['gender']['value']
    #print(final_gender)
    if final_gender:
      return final_gender
    else:
      #print('No gender found')
      return None
  else:
    #print('No gender found')
    return None

def entity_recognition(question, answer, gender):
  if gender == 'male':
    gender = 'him'
  else:
    gender = 'her'
  doc = nlp(question)
  final_answer = ''
  #print([(X.text, X.label_) for X in doc.ents])
  entity = [(X.text, X.label_) for X in doc.ents]
  for ent in entity:
    #print(ent)
    if ent[1] == 'PERSON':
      final_answer = answer.replace(ent[0],gender)
      break
  return final_answer

# NER for COUNT primarily
def ner(answer, named_entity):
  doc = nlp_ner_count(answer)
  #print([(X.text, X.label_) for X in doc.ents])
  entity = [(X.text, X.label_) for X in doc.ents]
  if len(entity) == 2:
    #print(entity[1][1])
    #print(answer.replace(entity[1][0], named_entity[entity[1][1]]))
    final_answer = answer.replace(entity[1][0], named_entity[entity[1][1]])
    return final_answer
  else:
    #print(answer, 'has more than one entity')
    return None

def process_vanilla_template(data):
    #print(data)
    answers = []
    if data[SPARQL_QUERY_ANSWER] != None:
        all_results = data[SPARQL_QUERY_ANSWER]['results']['bindings']
        data[SPARQL_QUERY_ANSWER] = [result['uri']['value'] for result in all_results]
    for result in data[SPARQL_QUERY_ANSWER]:
        result = result.rsplit('/', 1)[-1] if 'http' in result else result
        answers.append(result.replace('_', ' '))
    be_verb = BE_VERB_PLURAL if len(answers) > 1 else BE_VERB_SINGULAR
    if random.uniform(0, 1) > 0.5 and len(answers) == 1: # 1. First answer
        triples_information = data[INTERMIDIATE_QUESTION_TEMPLATE].split('<', 1)[1].replace('?', '')
        data[ANSWER_VERBALIZATION_TEMPLATE] = f"[{', '.join(answers)}] {be_verb} the <{triples_information}."
        # Gender-based answer generation
        gender = gender_recognition(data[SPARQL_QUERY])
        if gender:
          data[ANSWER_GENDER_TEMPLATE_2] = entity_recognition(data[CORRECTED_QUESTION_TEMPLATE], data[ANSWER_VERBALIZATION_TEMPLATE],  gender)
        else:
          data[ANSWER_GENDER_TEMPLATE_2] = ''
        
        #Entity based aswer generation
        ENTITY_RECOGNISED_ANSWER_TEMPLATE = ner(data[ANSWER_VERBALIZATION_TEMPLATE],named_entity)
        if ENTITY_RECOGNISED_ANSWER_TEMPLATE:
          data[ANSWER_COUNT_TEMPLATE_2] =  ENTITY_RECOGNISED_ANSWER_TEMPLATE
        else:
          data[ANSWER_COUNT_TEMPLATE_2] = ''
        # No ask template
        data[ANSWER_ASK_TEMPLATE_2] = ''
    else: # 2. First triples information
        if len(answers) > MAX_ALLOWED_QUERY_RESULTS: # Queries with too many answers cannot be supported for now
            data[SPARQL_QUERY_ANSWER] = f'Total answers: {len(answers)}'
            answers.clear()
            answers.append(ANSWER)
        triples_information = data[INTERMIDIATE_QUESTION_TEMPLATE].split('<', 1)[1].replace('?', '')
        data[ANSWER_VERBALIZATION_TEMPLATE] = f"The <{triples_information} {be_verb} [{', '.join(answers)}]."
        # Gender-based answer generation
        gender = gender_recognition(data[SPARQL_QUERY])
        if gender:
          data[ANSWER_GENDER_TEMPLATE_2] = entity_recognition(data[CORRECTED_QUESTION_TEMPLATE], data[ANSWER_VERBALIZATION_TEMPLATE],  gender)
        else:
          data[ANSWER_GENDER_TEMPLATE_2] = ''
        
        #Entity based aswer generation
        ENTITY_RECOGNISED_ANSWER_TEMPLATE = ner(data[ANSWER_VERBALIZATION_TEMPLATE],named_entity)
        if ENTITY_RECOGNISED_ANSWER_TEMPLATE:
          data[ANSWER_COUNT_TEMPLATE_2] =  ENTITY_RECOGNISED_ANSWER_TEMPLATE
        else:
          data[ANSWER_COUNT_TEMPLATE_2] = ''
        # No ask template
        data[ANSWER_ASK_TEMPLATE_2] = ''
        #data[ANSWER_PARAPHRASED_TEMPLATE] = paraphrase_back_translation(data[ANSWER_VERBALIZATION_TEMPLATE])
    return data

def process_count_template(data):
    if data[SPARQL_QUERY_ANSWER] != None:
        data[SPARQL_QUERY_ANSWER] = data[SPARQL_QUERY_ANSWER]['results']['bindings'][0]['callret-0']['value']
    answer = data[SPARQL_QUERY_ANSWER]
    be_verb = BE_VERB_PLURAL
    triples_information = data[INTERMIDIATE_QUESTION_TEMPLATE].split('<', 1)[1].replace('?', '.')
    data[ANSWER_VERBALIZATION_TEMPLATE] = f"There {be_verb} [{answer}] <{triples_information}"
    # Gender-based answer generation
    gender = gender_recognition(data[SPARQL_QUERY])
    if gender:
      data[ANSWER_GENDER_TEMPLATE_2] = entity_recognition(data[CORRECTED_QUESTION_TEMPLATE], data[ANSWER_VERBALIZATION_TEMPLATE],  gender)
    else:
      data[ANSWER_GENDER_TEMPLATE_2] = ''
    
    #Entity based aswer generation
    ENTITY_RECOGNISED_ANSWER_TEMPLATE = ner(data[ANSWER_VERBALIZATION_TEMPLATE],named_entity)
    if ENTITY_RECOGNISED_ANSWER_TEMPLATE:
      data[ANSWER_COUNT_TEMPLATE_2] =  ENTITY_RECOGNISED_ANSWER_TEMPLATE
    else:
      data[ANSWER_COUNT_TEMPLATE_2] = ''
    # No ask template
    data[ANSWER_ASK_TEMPLATE_2] = ''
    #data[ANSWER_PARAPHRASED_TEMPLATE] = paraphrase_back_translation(data[ANSWER_VERBALIZATION_TEMPLATE])
    return data


def process_ask_template(data):
    if data[SPARQL_QUERY_ANSWER] != None:
        data[SPARQL_QUERY_ANSWER] = data[SPARQL_QUERY_ANSWER]['boolean']
    answer = 'Yes' if data[SPARQL_QUERY_ANSWER] else 'No'
    be_verb = BE_VERB_PLURAL if BE_VERB_PLURAL in data[INTERMIDIATE_QUESTION_TEMPLATE] else BE_VERB_SINGULAR
    triples_information_first_part = data[INTERMIDIATE_QUESTION_TEMPLATE].split('>', 1)[0].split('<', 1)[1]
    triples_information_last_part = data[INTERMIDIATE_QUESTION_TEMPLATE].split('>', 1)[1].replace('?', '.')
    data[ANSWER_VERBALIZATION_TEMPLATE] = f"[{answer}], <{triples_information_first_part}> {be_verb} {triples_information_last_part}"
    # Gender-based answer generation
    gender = gender_recognition(data[SPARQL_QUERY])
    if gender:
      data[ANSWER_GENDER_TEMPLATE_2] = entity_recognition(data[CORRECTED_QUESTION_TEMPLATE], data[ANSWER_VERBALIZATION_TEMPLATE],  gender)
    else:
      data[ANSWER_GENDER_TEMPLATE_2] = ''
    
    #Entity based aswer generation
    ENTITY_RECOGNISED_ANSWER_TEMPLATE = ner(data[ANSWER_VERBALIZATION_TEMPLATE],named_entity)
    if ENTITY_RECOGNISED_ANSWER_TEMPLATE:
      data[ANSWER_COUNT_TEMPLATE_2] =  ENTITY_RECOGNISED_ANSWER_TEMPLATE
    else:
      data[ANSWER_COUNT_TEMPLATE_2] = ''
    
    #ASK template 2
    doc2 = nlp(data[CORRECTED_QUESTION_TEMPLATE])
    pos_tags = [(i, i.tag_) for i in doc2]
    #print(pos_tags)
    for i in doc2 :
      if 'VB' in i.tag_ :
        be_verb2 = str(i).lower()
        break
    triples_information_first_part2 = data[INTERMIDIATE_QUESTION_TEMPLATE].split('>', 1)[0].split('<', 1)[1]
    triples_information_last_part2 = data[INTERMIDIATE_QUESTION_TEMPLATE].split('>', 1)[1].replace('?', '')
    data[ANSWER_ASK_TEMPLATE_2] = f"[{answer}], {triples_information_last_part2} {be_verb2} {triples_information_first_part2}."   
    return data

def get_sparql_query_answer(query):
    try:
        SPARQL_ENDPOINT.setQuery(query)
        SPARQL_ENDPOINT.setReturnFormat(RETURN_FORMAT)
        answers = SPARQL_ENDPOINT.query().convert()
        return answers
    except Exception as e:
        print(f"Unable to Send query to {SPARQL_ENDPOINT.endpoint}")
        print(str(e))
        return None

def write_new_data_on_file():
    FINAL_VERBALIZED_DATA.sort(key=lambda x: int(x["_id"]))
    with open(NEW_JSON_PATH, 'w') as json_file:
        json.dump(FINAL_VERBALIZED_DATA, json_file)

print('Template file reading ...')
read_template_file()
print('LCQUAD data reading ...')
read_lcquad_data()
print('Generating verbalization ...')
generate_verbalization_template()
print('Writing to file ...')
write_new_data_on_file()

