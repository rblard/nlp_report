import nltk
nltk.download('punkt')
nltk.download('wordnet')

import os

# NLTK API requires hypotheses be an array of string arrays even if there is only one

def word_tokenize_cust(str, isHypothesis):
    if not isHypothesis:
        return [nltk.tokenize.word_tokenize(str)]
    else:
        return nltk.tokenize.word_tokenize(str)

def tokenize_hypothesis(str):
    return word_tokenize_cust(str,True)

def tokenize_reference(str):
    return word_tokenize_cust(str,False)

# Tokenize text into sentences and words

def process_text(str,**kwargs):
    if kwargs["isHypothesis"]:
        tokenize_function = tokenize_hypothesis
    else:
        tokenize_function = tokenize_reference
    return list(map(tokenize_function,nltk.tokenize.sent_tokenize(str,kwargs["language"])))

# Read from file

def secure_read(path,process_routine=(lambda x : x), **kwargs):
    with open(path, mode="r", encoding="utf-8-sig") as reader:
        base = reader.read()
        return process_routine(base,**kwargs)

# Write to file

def secure_write(path,seq,mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode=mode, encoding="utf-8-sig") as writer:
        writer.write(seq)

# Delete file contents

def clear_file(path):
    secure_write(path,"",mode="w")

# Read from file and return the proper tokenized array / array of arrays

def process_file(path,isHypothesis=False,language="english"):
    return secure_read(path,process_routine=process_text,isHypothesis=isHypothesis,language=language)
