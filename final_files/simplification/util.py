from final_files.util import sentence_tokenizer


def pre_process_text(text):
    new_lines_removed = text.replace('\n', '')
    # Make sure sentences are separated by a space
    sentences = sentence_tokenizer(new_lines_removed)

    return " ".join(sentences)


def groups_of_n(n, sentences):
    # Simply concatentate sentences without grouping
    if n == 0:
        return Exception("Need n > 0")

    if n == 1:
        return " ".join(sentences)

    grouped_text = ""
    i = 0
    while i < len(sentences):
        if i == 0:
            grouped_text += sentences[i].strip()
            grouped_text += " "
        elif i % 3 == 0:
            grouped_text += "\n"
            grouped_text += sentences[i].strip()
        else:
            grouped_text += sentences[i].strip()
            grouped_text += " "

        # print("sentence ", sentences[i])
            # if n == 1:
            #     grouped_text += sentences[i]
            #     grouped_text += "\n"
            # elif i == 0 or i == 1:
            #     grouped_text += sentences[i]
            #     grouped_text += " "
            # elif i % n == 0:
            #     grouped_text += sentences[i]
            #     grouped_text += "\n"
            # else:
            #     grouped_text += sentences[i]
            #     grouped_text += " "

        i += 1
    print("GROUPED: ", grouped_text)

    return grouped_text.strip()
