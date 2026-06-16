import spacy

nlp = spacy.load("en_core_web_lg")

text = """
Alex Johnson studied Computer Science at Caldwell University.
He worked at Amazon from June 2024 to August 2024 in New Jersey.
"""

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
