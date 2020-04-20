#!/usr/bin/env python
# coding: utf8

import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
import pydash
import clean
from custom_spacy import add_custom_modules

#TESTING MATERIALS
# =========================================================================================
# test = 'On March 24, Tarrant County issued a stay-at-home order through April 3.' 
# test = 'Natural Language Processing can be simple.'
# test = 'Dallas County imposed a stay-at-home order effective March 23 through April 3. Rowlett extended this order to cover the Rockwall County portion of the city. On March 23, Richardson extended the order to cover the Collin County portion of the city, effective until April 30. On March 24, Dallas extended the order to the entire city, including portions in Denton and Kaufman counties.'
# test = 'Ft. Worth extended the order to the entire city, including portions in Denton, Parker, and Wise counties.' 
# test = 'El Paso and El Paso County issued a shelter-in-place order effective March 24 until further notice. Denton County issued a stay-at-home order effective March 25. Travis and Williamson counties issued stay-at-home orders effective March 24 through April 13. Austin extended the Travis County order to the entire city, including portions in Hays County. Teague issued a shelter-in-place order effective March 25 until April 3. Milam County issued a shelter-in-place order effective March 25 until April 7. Fort Bend County issued a stay-at-home order effective March 25 through April 3. Castro County issued a shelter-in-place order effective March 24 through April 3. New Braunfels issued a stay-at-home order effective March 25. Newton County issued a 24-hour curfew for juveniles effective March 24. Gun Barrel City issued a shelter-in-place order effective March 25 until further notice. Starr County issued a shelter-in-place order effective March 25. Kaufman County issued a shelter-in-place order effective March 25 through April 8.'
# test = 'Galveston County issued a stay-at-home order effective March 24 until April 3. Lampasas issued a shelter-in-place order effective March 23 through April 5. Ft. Worth extended the order to the entire city, including portions in Denton, Parker, and Wise counties.'
# test = 'On March 23, Brazos County, in conjunction with Bryan and College Station, issued the order to begin shelter-in-place effective March 24 through April 7.'
# test = 'Gov of NJ is actually doing just as good a job as Cuomo. Acted sooner, #Hoboken was first city to shelter in place. #healthcare #COVID19'
# test = '@wigwam Milam County[339] issued a shelter-in-place order effective[229] March 25 until April 7. Fort Bend County #issued a stay-at-home order effective March 25 through April 3. New Braunfels issued a stay-at-home order effective March 25. Newton County issued a 24-hour curfew for juveniles effective March 24.'
# test = 'Patterns are not invented; they are harvested from repeated use in practice. If you have built integration solutions, it is likely that you have used some of these patterns, maybe in slight variations and maybe calling them by a different name.'
# =========================================================================================

class SpacyInsights:
    def __init__(self):
        self.spacy = add_custom_modules(spacy)
        self.spacy.prefer_gpu()
        self.nlp = spacy.load('en_core_web_sm')

    def extract_relationships(self,docs):
        if type(docs) == str:
            return self.sentencizer(docs)
        else:
            results = []
            for doc in docs:
                results.append(self.sentencizer(doc))
            return results

    def get_nlp(self, text, display = False):
        clean_text = clean.clean_text(text)
        doc = self.nlp(clean_text)
        if display:
            displacy.serve(doc, style="dep")
        return doc

    def on_order_match(self, matcher, doc, id, matches):
        len_change = 0
        match_id, start, end = matches[id]
        for i in range(0,id):
            match_id_, start_, end_ = matches[i]
            len_change += max(0,end_ - start_ - 1)
        span = doc[start-len_change:end-len_change]
        with doc.retokenize() as retokenizer:
            retokenizer.merge(span)

    def filter_ents(self, spans, doc):
        results = []
        for span in spans:
            last = span[0]
            start = span.start
            for token in span:
                # Case 1: Token is not a Proper Noun, but the previous token is. 
                # The token is also not a right dependency of the last Proper Noun.
                # Therefore, The prior proper noun phrase has ended, and it must be added to the set
                if token.pos_ != 'PROPN' and last.pos_ == 'PROPN' and token not in list(last.rights):
                    results.append(doc[start:token.i])
                    # start = token.i
                    last = token
                # Case 2: Token is a Proper Noun, but the previous token is not.
                # Therefore, a proper noun phrase has begun
                elif token.pos_ == 'PROPN' and last.pos_ != 'PROPN':
                    start = token.i
                    last = token
                # Case 3: Token is either a Proper Noun, or it is a right dependency of the prior proper noun
                # Token is also the final token before the end of the phrase
                # Therefore, a proper noun phrase has ended and it must be added to the set
                elif token.i == span.end - 1 and (token.pos_ == 'PROPN' or token in list(last.rights)):
                    results.append(doc[start:token.i+1])
        return results
            
    def filter_spans(self, spans):
        # Filter a sequence of spans so they don't contain overlaps
        # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
        get_sort_key = lambda span: (span.end - span.start, -span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                    result.append(span)
            seen_tokens.update(range(span.start, span.end))
        result = sorted(result, key=lambda span: span.start)
        return result

    def combine_phrases(self, doc):
        ordermatcher = PhraseMatcher(self.nlp.vocab)
        ordermatcher.add("SOCIALDISTANCE", self.on_order_match, self.nlp("social distancing"), self.nlp("social-distancing"), self.nlp("Social Distancing"))
        ordermatcher.add("SAFERATHOME", self.on_order_match, self.nlp("safer at home"), self.nlp("safer-at-home"), self.nlp("Safer at Home"))
        ordermatcher.add("SHELTERINPLACE", self.on_order_match, self.nlp("shelter in place"), self.nlp("shelter-in-place"), self.nlp("Shelter in Place"))
        ordermatcher.add("STAYATHOME", self.on_order_match, self.nlp("stay at home"), self.nlp("stay-at-home"), self.nlp("Stay at Home"))
        ordermatcher(doc)

    def find_right_dependency(self, verb, dependencies):
        result = [x for x in verb.rights if x.dep_ in dependencies]
        if len(result) == 0:
            for right in list(verb.rights):
                result = self.find_right_dependency(right, dependencies)
                if len(result) > 0:
                    return result
        return result
    
    def find_left_dependency(self, verb, dependencies):
        result = [x for x in verb.lefts if x.dep_ in dependencies]
        if len(result) == 0:
            for left in list(verb.lefts):
                result = self.find_left_dependency(left, dependencies)
                if len(result) > 0:
                    return result
        return result

    def find_dependency(self, verb, dependencies, start_right = True):
        result = []
        if start_right:
            result = self.find_right_dependency(verb, dependencies)
            if len(result) == 0:
                result = self.find_left_dependency(verb, dependencies)
        else:
            result = self.find_left_dependency(verb, dependencies)
            if len(result) == 0:
                result = self.find_right_dependency(verb, dependencies)
        return result

    def find_parent_verb(self,w):
        verb = w
        while verb.pos_ != "VERB":
            verb = verb.head
        return verb
    
    def find_parent_noun(self,w):
        noun = w
        while noun.pos_ != "NOUN":
            noun = noun.head
        return noun

    def extract(self,doc):
        # Merge entities and noun chunks into one token
        self.combine_phrases(doc)
        spans = self.filter_ents(list(doc.ents), doc)
        spans += list(doc.noun_chunks)
        spans = self.filter_spans(spans)

        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

        relations = []
        for w in filter(lambda w: w.pos_ == "PROPN" or w.pos_ == "NOUN", doc):
            #w is a direct object of a verb OR an attr
            if w.dep_ in ("attr", "dobj"):
                verb = w.head
                #The verb is a clausal modifier of a noun
                if verb.dep_ == "acl":
                    subject = self.find_parent_noun(verb)
                    relations.append((subject.lemma_, w.head.lemma_, w.lemma_))
                else:
                    [subject] = self.find_left_dependency(w.head, ("nsubj")) 
                    relations.append((subject.lemma_, w.head.lemma_, w.lemma_))
            if w.dep_ == "pobj" and w.head.dep_ == "prep":
                verb = w.head.head
                if verb.pos_ != "VERB": 
                    verb = self.find_parent_verb(w)
                [result] = self.find_right_dependency(verb, ("pobj","dobj"))
                #We have a prepositional modifier on the verb
                relations.append((result.lemma_, verb.lemma_, w.head.lemma_, doc[w.i:w.i+len(list(w.subtree))].lemma_))
            if w.dep_ == "nsubj":
                verb = w.head
                result = None
                if verb.pos_ == "AUX": # AUX Means We Have Modifiers
                    verb = pydash.find(list(verb.lefts), lambda x: x.dep_ in ("advcl","aux") and x.pos_ == "VERB")
                    [result] = self.find_right_dependency(verb.head, ("relcl","pobj","dobj","nsubj","acomp"))
                    relations.append((w.lemma_, verb.lemma_, w.head.lemma_ + ' ' + result.lemma_))
                else:
                    [result] = self.find_right_dependency(verb, ("pobj","dobj"))
                    relations.append((w.lemma_, verb.lemma_, result.lemma_))
            if w.dep_ in ("npadvmod", "npmod"):
                verb = self.find_parent_verb(w)
                [result] = self.find_right_dependency(verb, ("pobj","dobj"))
                #There is a noun phrase modifier on the verb
                relations.append((result.lemma_, verb.lemma_, w.head.lemma_, w.lemma_))
        return pydash.arrays.uniq(relations)

    def sentencizer(self,text):
        doc = self.get_nlp(text)
        results = []
        for sentence in doc.sents:
            s_doc = sentence.as_doc()
            results.append(self.extract(s_doc))
        # print("\n++++ RESULTS ++++")
        # for i, res in enumerate(results):
        #     print(i,':',res)
        return results

# TESTING
# x = SpacyInsights()
# x.extract_relationships(test)

