

PROMPT = {
    'instruction': '',
    'input': '',
    'output': ''
}

INSTRUCTION_ED_TEMPLATE = [
    'Given a sentence or paragraph, identify the event type from the given options and its corresponding trigger word.',
    'Identify the event type associated with the given options and the trigger word from input.',
    'Please identify the trigger word and event type belong to given options from the following input.',
]

INSTRUCTION_NER_TEMPLATE = [
    'Given a sentence or paragraph, identify the entities from the given options. Output format is (word1, type1), (word2, type2).',
    'List the entities associated with the given options from input. Output format is (word1, type1), (word2, type2).',
    'Please identify the entities with the given options from the following input. Output format is (word1, type1), (word2, type2).',
    'Find all the entities belong to the given options from given text. Output format is (word1, type1), (word2, type2).'
]

OUTPUT_TEMPLATE = [
    'The event type is {evt} and the corresponding arguments are {args}',
    'Event type: {evt}, Arguments: {args}',
    'Entities: {args}'
]

ARG_TEMPLATE = [
    '({arg}, {role})'
]

NONE_TEMPLATE = 'There is no {target} found from the input.'


IUIE_NER_INSTRUCT_TEMPLATE = [
    'Please list all entity words in the text that fit the category. Output format is "word1:type1; word2:type2".',
    'Please find all the entity words associated with the category in the given text. Output format is "word1:type1; word2:type2".',
    'Please tell me all the entity words in the text that belong to a given category. Output format is "word1:type1; word2:type2".'
]

IUIE_RE_INSTRUCT_TEMPLATE = [
    'Given a phrase that describes the relationship between two words, extract the words and the lexical relationship between them. The output format should be "relation1: word1, word2; relation2: word3, word4".',
    'Find the phrases in the following sentences that have a given relationship. The output format is "relation1: word1, word2; relation2: word3, word4".',
    'Given a sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of "relation1: word1, word2; relation2: word3, word4".'
]