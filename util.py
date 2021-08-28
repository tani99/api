import spacy

nlp = spacy.load("en_core_web_sm")
# Merge noun phrases and entities for easier analysis
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

def get_keywords(text):
    keywords = []
    imp_ents = ["ORG", "DATE", "MONEY", "PERSON", "TIME",  "GPE", "EVENT", "WORK_OF_ART", "PERCENT", "FAC", "NORP", "LAW"]
    doc = nlp(text)
    for token in doc:
        if token.ent_type_ in imp_ents:
            keywords.append(token.text.lower())
    return keywords