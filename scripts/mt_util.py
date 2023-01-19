import deepl
import googletrans

deepl_auth_key = "personal key removed from github repository"
deepl_translator = deepl.Translator(deepl_auth_key)
gt_translator = googletrans.Translator()

# Return a text as an array of chunks 5000 or less characters long

def divide_in_chunks(text):
    max_size = 5000
    return [text[i:i+max_size] for i in range(0,len(text),max_size)]

# Translate individual chunks of a text too large for one request, and piece them back together.

def translate_by_chunks(text,translate_func,**kwargs):
    result = []
    text_chunks = divide_in_chunks(text)

    for chunk in text_chunks:
        result.append(translate_func(chunk,**kwargs).text)

    return ''.join(result)

# Generate DeepL translation

def deepl_translate_to_english(text):
    result = deepl_translator.translate_text(text,target_lang="EN-US") # DeepL forces us to specify EN-US or EN-GB
    return result.text

# Generate DeepL chunk translation

def deepl_translate_by_chunks(text):
    kwargs = {"target_lang":"EN-US"}
    return translate_by_chunks(text,deepl_translator.translate_text,**kwargs)

# Generate Google Translation (chunk) translation

def gt_translate_to_english(text):
    kwargs = {"src":"fr","dest":"en"}
    # We have no choice but to translate in chunks, as the API will not take more than 5000 characters without an error.
    # However, this changes nothing in our comparison.
    return translate_by_chunks(text,gt_translator.translate,**kwargs)
