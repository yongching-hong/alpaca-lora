import re

def parse_response(text):
    text = text.replace('Entities: ', '')

    prefix_suffix_bracket = "^\(|\)$"
    text = re.sub(prefix_suffix_bracket, '', text)
    response_sets = text.split('), (')
    entities = [tuple(re.split('(?<=\S), ', set.lower())) for set in response_sets]

    return entities
