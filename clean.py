import re
from abbreviations import fix_abbreviations
# import spacy

def has_abbreviations(token):
    found = re.findall(r"([\w]*[.])", token)
    if found:
        return found
    else:
        return []

def has_annotation(token):
    found = re.findall(r"(\[\d+\])", token)
    if found:
        return found
    else:
        return []

def has_hashtags(token):
    found = re.findall(r"(#)\w*", token)
    if found:
        return found
    else:
        return []

def has_mentions(token):
    found = re.findall(r"(@\w*)", token)
    if found:
        return found
    else:
        return []

# To enable, must import spacy and load model
def remove_stopwords(doc):
    all_stopwords = [] #spacy.Defaults.stop_words
    tokens_without_stops = [word for word in doc if not word in all_stopwords]
    return tokens_without_stops

def clean_text(doc):
    filtered_sentence = ''
    split_doc = doc.split()
    for i, token in enumerate(split_doc):
        bad_text = has_annotation(token) + has_hashtags(token) + has_mentions(token)
        if len(bad_text) > 0:
            for text in bad_text:
                split_doc[i] = token.replace(text,'')
        split_doc[i] = split_doc[i].strip()

        abbreviations = has_abbreviations(split_doc[i])
        if len(abbreviations) > 0:
            split_doc[i] = fix_abbreviations(split_doc[i])

    #split_doc = remove_stopwords(split_doc)

    for token in split_doc:
        filtered_sentence += token + ' '

    return filtered_sentence.strip()

