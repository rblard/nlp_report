from process_util import process_text
from deepl_util import deepl_translate_to_english
from nltk.translate.bleu_score import corpus_bleu

# Simple test script used at the beginning of the project. Useless now.

if __name__ == "__main__":
    test_string_fr = "C\'était un profond cratère, recouvert de mousses sombres et d'herbe fraîche."

    test_string_en = "It was a deep crater, covered in dark moss and fresh grass."

    test_string_mt = deepl_translate_to_english(test_string_fr)

    processed_en = process_text(test_string_en,False)
    processed_mt = process_text(test_string_mt,True)

    print(processed_en)
    print(processed_mt)
    print(corpus_bleu(processed_en,processed_mt))
