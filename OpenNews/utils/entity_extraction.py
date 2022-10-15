"""

Need to run first:
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words');"

"""

import nltk
import re

def get_entity_list(text):
    entities = []
    for sentence in nltk.sent_tokenize(text):
        chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        entities += [" ".join(t[0] for t in chunk) for chunk in chunks if hasattr(chunk, "label")]
    return entities


def get_entity_token_list(text):
    entities_list = get_entity_list(text)
    entity_token_list = []
    for entity in entities_list:
        entity_token_list += entity.split(" ")
    entity_token_list = list(map(lambda s: re.sub(r"\W+", "", s).lower(), entity_token_list))
    return entity_token_list

if __name__ == "__main__":
    print(get_entity_list("Elon Musk did this.."))