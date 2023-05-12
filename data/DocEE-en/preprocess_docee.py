import jsonlines
import json
import os
import spacy
from tqdm import tqdm
import re

nlp = spacy.load("en_core_web_sm")

docee_path = f'./normal_setting'
output_path = f'./normal_setting/fixed'

def read_jsonlines(input_file):
    lines = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def norm_role(text):
    role = text.replace('_', ' ').replace('.', ' ')
    return role

def norm_event_type(text):
    event = text.replace('_', ' ').replace('-', ' ')
    
    return event

def norm_paragraph(text):
    normalized = text.replace('--', ' ')

    return normalized

def norm_tokenize(text):
    doc = nlp(text)
    tokens = []
    char_start_idx = []
    
    for token in doc:
        if not token.is_space or '\\' in repr(token.text):
            tokens.append(token.text)
            char_start_idx.append(token.idx)

    return tokens, char_start_idx

# Fix incorrect mention span index
"""
Original Text:  iscal policy legislation | Fixed:  {'start': 1258, 'end': 1283, 'type': 'Policy Proposals', 'text': 'fiscal policy legislation', 'tokens': ['fiscal', 'policy', 'legislation']}
Original Text:  nvestment in education catch-up, retraining schemes, infrastructure spending, demand management frameworks | Fixed:  {'start': 9692, 'end': 9799, 'type': 'Policy Proposals', 'text': 'investment in education catch-up, retraining schemes, infrastructure spending, demand management frameworks', 'tokens': ['investment', 'in', 'education', 'catch', '-', 'up', ',', 'retraining', 'schemes', ',', 'infrastructure', 'spending', ',', 'demand', 'management', 'frameworks']}
Original Text:  efaults | Fixed:  {'start': 4036, 'end': 4044, 'type': 'Cause', 'text': 'defaults', 'tokens': ['defaults']}
Original Text:  atya Sai Baba | Fixed:  {'start': 328, 'end': 342, 'type': 'People', 'text': 'Satya Sai Baba', 'tokens': ['Satya', 'Sai', 'Baba']}
Original Text:  he Queen | Fixed:  {'start': 384, 'end': 393, 'type': 'People', 'text': 'the Queen', 'tokens': ['the', 'Queen']}
"""
def process(lines, processed_path, processed=[], set_type='train'):
    fixed_num = 0
    
    processed_docs = []
    print(f'Loaded {set_type} raw data: {len(lines)}, processed data: {len(processed)}')

    def validate(s, e, text, tok_idx, last_start_tok_idx):
        start = tok_idx + last_start_tok_idx
        mentions, _ = norm_tokenize(text[s:e])
        end = start + len(mentions)

        return start, end

    for doc_idx, line in tqdm(enumerate(lines)):
        if processed and doc_idx <= len(processed)-1:
            doc = processed[doc_idx]
            processed_docs.append(doc)
        else:
            text = norm_paragraph(line[1])
            tokens, char_start_idx = norm_tokenize(text)

            doc = {
                'doc_id': doc_idx,
                'title': line[0],
                'text': line[1],
                'event_type': line[2],
                'event_mentions': line[3]
            }
            
            doc['event_mentions'].sort(key=lambda x: x['start'])

            offset = 0
            last_start_tok_idx = 0
            
            for m_idx, m in enumerate(doc['event_mentions']):
                offset = line[1].count('--', 0, m['start'])
                s = m['start'] - offset
                e = m['end'] - offset
                
                for tok_idx, char_idx in enumerate(char_start_idx[last_start_tok_idx:]):
                    # Validate and fix incorrect labelled data (1 index forward error)
                    if s < char_idx:
                        if s-1 == char_start_idx[last_start_tok_idx:][tok_idx-1]:
                            start, end = validate(s-1, e, text, tok_idx-1, last_start_tok_idx)

                            if start > len(tokens) or end > len(tokens):
                                break

                            last_start_tok_idx += tok_idx
                            s -= 1
                            updated_text = text[s:e]

                            if updated_text and not re.match('^[a-zA-Z0-9_]+$',updated_text[0]):
                                break

                            print('Original Text: ', m['text'], '| Fixed: ', {
                                'start': s,
                                'end': e, 
                                'type': m['type'] ,
                                'text': updated_text,
                                'tokens': tokens[start:end],
                            })

                            m.update({
                                'start': s,
                                'end': e,
                                'type': m['type'],
                                'text': updated_text
                            })
                            fixed_num += 1
                            print('Fixed Arguments: ', fixed_num)
                            break
                        last_start_tok_idx += tok_idx-1
                        break
                
            processed_docs.append(doc)
            
            if (doc_idx % 100 == 0) or (doc_idx % (len(lines)-1) == 0):
                with open(processed_path, 'w') as f:
                    f.write(json.dumps(processed_docs))
                    print(f'Saved {len(processed_docs)} processed {set_type} DocEE at {processed_path}')

def run_fixed(set_type):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    file_path = os.path.join(docee_path, f'{set_type}.json')
    fixed_path = os.path.join(output_path, f'{set_type}.json')
    fixed = os.path.exists(fixed_path)
    fixed_lines = []

    if fixed:
        fixed_lines = read_jsonlines(fixed_path)
        fixed_lines = fixed_lines[0] if fixed_lines else []
    
    lines = read_jsonlines(file_path)[0]
    return process(lines, fixed_path, fixed_lines, set_type)

def start():
    for set_type in ['train', 'test', 'dev']:
        run_fixed(set_type)

start()