import os
import json
import yaml
import random
from copy import deepcopy
from tqdm import tqdm
from prompt import INSTRUCTION_ED_TEMPLATE, INSTRUCTION_NER_TEMPLATE, PROMPT, OUTPUT_TEMPLATE, ARG_TEMPLATE, NONE_TEMPLATE, IUIE_NER_INSTRUCT_TEMPLATE
from transformers import LlamaTokenizer

from utils.prompter import Prompter

dataset_path = {
    'docee': './data/DocEE/normal_setting/fixed/',
    'fewnerd': './data/fewnerd/K200/',
    'fewnerd_sup': './data/fewnerd/supervised/',
    'ace': './data/ace05/'
}
mapper_path = {
    'docee': './data/DocEE-en/type_mapping.json',
    'fewnerd': './data/fewnerd/mapper.json',
    'fewnerd_sup': './data/fewnerd/mapper.json',
    'ace': './data/ace05/mapper.yaml',
}
set_type = ['train', 'test', 'dev']
tokenizer = LlamaTokenizer.from_pretrained('yahma/llama-7b-hf')
prompter = Prompter("alpaca")
cutoff_len = 500

def load_json(path):
    with open(path) as f:
        docs = json.load(f)
        print(len(docs))
    return docs

def sub_pattern(text, p_tuple_list):
    for p_tuple in p_tuple_list:
        text = text.replace(p_tuple[0], p_tuple[1])
    return text

def process(dataset, docs, label_mapper, task='ner'):
    prompted = []
    exceed_len_samples = 0
    max_len = 0
    for doc in tqdm(docs):
        if dataset == 'docee':
            prmpt = _process_docee(deepcopy(PROMPT), doc, label_mapper)
        else:
            prmpt = _process_fewnerd(deepcopy(PROMPT), doc, label_mapper, task)
        full_prompt = prompter.generate_prompt(
            prmpt["instruction"],
            prmpt["input"],
            prmpt["output"],
        )
        result = tokenizer(full_prompt)
        if len(result.input_ids) > max_len:
            max_len = len(result.input_ids)

        if len(result.input_ids) > cutoff_len:
            exceed_len_samples += 1
            print(f"exceed length limit {cutoff_len}: {len(result.input_ids)}")
            print("total dropped ", exceed_len_samples)
            continue
        prompted.append(prmpt)
    print('Max length: ', len(result.input_ids))
    return prompted

def _process_docee(prmpt, doc, label_mapper):
    # dict_keys(['doc_id', 'title', 'text', 'event_type', 'event_mentions'])
    prmpt['instruction'] = random.choice(INSTRUCTION_ED_TEMPLATE)
    prmpt['input'] = doc['text']

    args = ', '.join([sub_pattern(ARG_TEMPLATE[0], [('{arg}', m['text']), ('{arg}', m['text']), ('{role}', m['type'])]) for m in doc['event_mentions']])
    event_type = label_mapper[doc['event_type']]
    prmpt['output'] = sub_pattern(OUTPUT_TEMPLATE[1], [('{evt}', event_type), ('{args}', args)])
    return prmpt

def _process_fewnerd(prmpt, doc, label_mapper, task='ner'):
    options = " Options: " + ', '.join(label_mapper.values())
    if task == 'ed':
        instruct_template = INSTRUCTION_ED_TEMPLATE
        key = 'event'
    elif task == 'ner':
        instruct_template = IUIE_NER_INSTRUCT_TEMPLATE
        key = 'entity'
    else:
        raise 

    prmpt['instruction'] = random.choice(instruct_template) + options
    prmpt['input'] = doc['sent']

    if doc['event_list']:
        args = ', '.join([sub_pattern(ARG_TEMPLATE[0], [('{arg}', m['trigger']['text']), ('{role}', label_mapper[m['type']])]) for m in doc['event_list']])
        prmpt['output'] = args
    else:
        prmpt['output'] = sub_pattern(NONE_TEMPLATE, [('{target}', key)])
    
    return prmpt

def save(save_filepath, data):
    with open(save_filepath, 'w') as f:
        f.write(json.dumps(data))

def norm_fewnerd(docs):
    norm_docs = []
    for label, samples in docs.items():
        norm_docs.extend(samples)
    return norm_docs

def run(dataset, task='ner'):
    for st in set_type:
        ds_path = dataset_path.get(dataset)
        data_path = os.path.join(ds_path, f'{st}.json')
        if not os.path.exists(data_path):
            continue
        docs = load_json(data_path)
        if dataset == 'fewnerd':
            docs = norm_fewnerd(docs)

        mp_path = mapper_path.get(dataset)
        
        if mp_path.endswith('.yaml'):
            mapper = yaml.load(open(mp_path), Loader=yaml.FullLoader).get('mapper')
        else:
            mapper = load_json(mp_path)
        prompted_data = process(dataset, docs, mapper, task)
        print(len(prompted_data))
        output_dir = os.path.join(ds_path, 'fixed_prompted/')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        save(os.path.join(output_dir, f'{st}.json'), prompted_data)

run('fewnerd', 'ner')