from constants import fr_path, deepl_path, deepl_path_alt, gt_path
from process_util import secure_read, secure_write
from mt_util import deepl_translate_to_english, deepl_translate_by_chunks, gt_translate_to_english

# WARNING :
# This script will call the DeepL API twice.
# For free plans, there is a monthly character limit of 500,000 on this API.

# No DeepL API key is provided in this repository, therefore this script will not work as is.
# Premade translations are provided instead in the assets folder.

if __name__ == "__main__":
    text_fr = secure_read(fr_path)
    text_deepl = deepl_translate_to_english(text_fr)
    secure_write(deepl_path,text_deepl)
    text_deepl = deepl_translate_by_chunks(text_fr)
    secure_write(deepl_path_alt,text_deepl)
    text_gt = gt_translate_to_english(text_fr)
    secure_write(gt_path,text_gt)
