import spacy

def compound_getter(token):
    #span
    if type(token) is spacy.tokens.Span:
        return token
    
    doc = token.doc
    if token.dep_ != "compound":
        return doc[token.i:token.i+1]
    start = token.i
    end = token.i + 1
    temp = token

    if len(list(temp.lefts)):
        for left in list(temp.lefts):
            start -= 1

    while temp.dep_ == "compound":
        end += 1
        temp = temp.head
    compound = doc[start:end]
    return compound

def add_custom_modules(spacy):
    spacy.tokens.Token.set_extension("compound_", getter=compound_getter)
    return spacy
