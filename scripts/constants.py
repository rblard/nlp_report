# Paths of the original text files

fr_path = "assets/Delphine_et_l_oiseau.txt"
en_path = "assets/Delphine_and_the_bird.txt"
deepl_path = "assets/Delphine_and_the_bird_DEEPL.txt"
deepl_path_alt = "assets/Delphine_and_the_bird_DEEPL_CHUNK.txt"
gt_path = "assets/Delphine_and_the_bird_GT.txt"

# The processed_{}_path vars should normally not be used and text should be reprocessed on every run.

processed_deepl_path = "assets/processed_deepl.txt"
processed_gt_path = "assets/processed_gt.txt"
processed_en_path = "assets/processed_en.txt"

# For graph insertion

formatted_version_names = {"DEEPL" : "DeepL", "GT": "Google Translate"}

# Dict keys used in the main function of the evaluate_translation.py script, and by create_path_dic

first_level_keys = ["DEEPL","GT"]
second_level_keys = ["BLEU","NIST","METEOR"]

# Generate path hierarchy for the generated files

def create_path_dic(basename,extension=".png"):

    root_path = "results/"

    first_level_dic = {}
    second_level_dic = {}

    for fkey in first_level_keys:
        for skey in second_level_keys:

            version_dir = fkey.lower()+"/"
            metric_dir = skey.lower()+"/"

            filename = fkey.lower()+"_"+skey.lower()+"_"+basename+extension

            entry = root_path+version_dir+metric_dir+filename

            second_level_dic.update({skey: entry})

        first_level_dic.update({fkey: second_level_dic})
        second_level_dic = {}

    return first_level_dic

stats_path = create_path_dic("stats",".json")
individual_score_path = create_path_dic("individual_score")
score_by_length_path = create_path_dic("score_by_length")
score_repartition_path = create_path_dic("score_repartition")
