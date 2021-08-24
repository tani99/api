import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load('en_core_web_sm')


def summarise_paragraph(paragraph, include_first_half):
    summarised = ""
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        summarised += summarise(sentence, include_first_half)
        summarised += '\n'

    return summarised

# Takes a sentence with POS tags and extracts notes in the format: VERB: Description
def summarise(sentence, include_first_half):
    # nlp.add_pipe("merge_entities")
    doc = nlp(sentence)
    merged_doc = merge_phrases(doc)
    pos_tagged = tag_sentence(merged_doc)
    pos_tagged = [(nltk.map_tag('en-ptb', 'universal', tag), word) for word, tag in pos_tagged]
    merged = merge_adjacent_tags(pos_tagged)
    verb, content = extract(pos_tagged)

    if include_first_half:
        print('\u2022 ', verb, ': ', content)
        return '\u2022  ' + verb + ': ' + content
    else:
        print('\u2022  ', content)
        return '\u2022  ' + content


def merge_adjacent_tags(tagged):
    for i in reversed(range(len(tagged))):
        if i == 0:
            break
        (pos, word) = tagged[i]
        (pos_next, word_prev) = tagged[i - 1]
        if pos == pos_next:
            tagged[i - 1] = (pos, word_prev + ' ' + word)
            tagged.remove((pos, word))
    return tagged


def extract(tagged):
    verb = ''
    content = ''
    verb_found = False
    for (pos, word) in tagged:
        if pos == 'VERB' and not verb_found and word != "/":
            verb_found = True

        if not verb_found:
            verb += (' ' + word)
        if verb_found:
            content += (' ' + word)

    return verb, content


def tag_sentence(doc):
    tokenized = [token.text for token in doc]
    # print(tokenized)
    pos_tagged = nltk.pos_tag(tokenized)
    return pos_tagged


def merge_phrases(doc):
    with doc.retokenize() as retokenizer:
        for np in list(doc.noun_chunks):
            attrs = {
                "tag": np.root.tag_,
                "lemma": np.root.lemma_,
                "ent_type": np.root.ent_type_,
            }
            retokenizer.merge(np, attrs=attrs)
    return doc


def is_title(line):
    words = word_tokenize(line)
    return len(words) < 5


def process_text(text, include_first_half=True):
    lines = text.split('\n')
    final = ""
    for line in lines:
        if is_title(line):
            final += line
            final += '\n'
            print(line)
        else:
            final+= summarise_paragraph(line, include_first_half)
    return final


# sentence = "Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha"
# summarise(sentence)

paragraph = """Term 
The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and had to resign. 

Composition
The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545. 

Election
Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individuaL The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. 
What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back. """

para2 = """The Speaker
The Presiding Officer is the Speaker. He/she is elected by the members of the Lok Sabha from amongst themselves on the first day Parliament meets. Usually, it is the ruling party that nominates a person for the post and after consultation with other members. The Speaker is a very respected and experienced parliamentarian. He/she is assisted by the Deputy Speaker. The Speaker is the person responsible for the smooth functioning of the Lok Sabha. 
The Speaker presides over the meetings of the Lok Sabha. He/she is responsible for the discipline in the House and must maintain its decorum and dignity. He/she can expel a member for the day or even an entire session. The Speaker has the responsibility to see that each member gets a fair chance to speak and be heard."""

# process_text(paragraph)
process_text(paragraph)
process_text(paragraph, False)
process_text(para2)
process_text(para2, False)
