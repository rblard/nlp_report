import json
import numpy as np
import matplotlib.pyplot as plt

from process_util import process_file, secure_write, secure_read, clear_file
from constants import *
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.nist_score import corpus_nist, sentence_nist
from nltk.translate.meteor_score import meteor_score

processed_en = []
processed_deepl = []
processed_gt = []
common_len = 0

human_length_plotted = False

# methods 5 and 7 give incorrect (>1) scores to some sentences and must be left out
# method 2 gives the highest sentence average
#  corpus BLEU score barely changes
chosen_smoothing_function = SmoothingFunction().method2

def prepare_processed_arrs():
    global processed_en, processed_deepl, processed_gt, common_len
    processed_en = process_file(en_path,isHypothesis=False)
    processed_deepl = process_file(deepl_path,isHypothesis=True)
    processed_gt = process_file(gt_path, isHypothesis=True)
    common_len = len(processed_en)

def determine_eval_func(metric,mode):
    eval_func = None
    kwargs = {}

    match metric:
        case "BLEU":
            match mode:
                case "SENTENCE":
                    eval_func = sentence_bleu
                case "CORPUS":
                    eval_func = corpus_bleu
            kwargs['smoothing_function'] = chosen_smoothing_function
        case "NIST":
            match mode:
                case "SENTENCE":
                    eval_func = sentence_nist
                case "CORPUS":
                    eval_func = corpus_nist
            kwargs['n'] = 4
        case "METEOR":
            match mode:
                case "SENTENCE":
                    eval_func = meteor_score

    return eval_func, kwargs

def evaluate_corpus(version,metric):
    prepare_processed_arrs()

    eval_func, kwargs  = determine_eval_func(metric,"CORPUS")

    score = -1

    processed_mt = []

    if version=="DEEPL":
        processed_mt = processed_deepl
    elif version=="GT":
        processed_mt = processed_gt

    if eval_func != None: # For METEOR, there is no corpus-wise evaluation
        score = eval_func(processed_en,processed_mt,**kwargs)

    return processed_mt, score

def create_debug_files():

    clear_file(processed_en_path)
    clear_file(processed_deepl_path)
    clear_file(processed_gt_path)

    prepare_processed_arrs()

    sent_str = ""

    for sent in processed_deepl:
        for word in sent:
            sent_str+=word
            sent_str+=" "
        sent_str+="\n"
        secure_write(processed_deepl_path,sent_str,mode="a")
        sent_str=""

    sent_str=""

    for sent in processed_gt:
        for word in sent:
            sent_str+=word
            sent_str+=" "
        sent_str+="\n"
        secure_write(processed_gt_path,sent_str,mode="a")
        sent_str=""

    sent_str=""

    for sent_cont in processed_en:
        for sent in sent_cont:
            for word in sent:
                sent_str+=word
                sent_str+=" "
            sent_str+="\n"
            secure_write(processed_en_path,sent_str,mode="a")
            sent_str=""

def write_results_as_json(version,metric):

    # Cleanup

    clear_file(stats_path[version][metric])
    dict = {}

    # Calculate global score for the given machine translation with the given metric

    processed_mt, score = evaluate_corpus(version,metric)

    dict.update({"corpus_size": common_len})
    if score > 0:
        dict.update({"total_corpus_score": score})

    # Retrieve each sentence's length for both human and machine translation

    arr_of_en_sentence_lengths = []

    for i in range(common_len):
        arr_of_en_sentence_lengths.append(len(processed_en[i][0]))

    dict.update({"human_sentence_lengths": arr_of_en_sentence_lengths})

    arr_of_mt_sentence_lengths = []

    for i in range(common_len):
        arr_of_mt_sentence_lengths.append(len(processed_mt[i]))

    dict.update({"mt_sentence_lengths": arr_of_mt_sentence_lengths})

    # Apply the given metric on each individual sentence of the given machine translation

    arr_of_scores = []

    eval_func, kwargs = determine_eval_func(metric,"SENTENCE")

    for i in range(common_len):
        try: # Some NIST sentences result in a zero denominator
            sent_score = eval_func(processed_en[i],processed_mt[i],**kwargs)
            arr_of_scores.append(sent_score)
        except ZeroDivisionError:
            arr_of_scores.append(0)

    dict.update({"individual_sentence_scores": arr_of_scores})

    secure_write(stats_path[version][metric],json.dumps(dict),mode="a")

def draw_graphs_from_json(version,metric):

    # Gather data

    stats_data = json.loads(secure_read(stats_path[version][metric]))

    corpus_size = stats_data['corpus_size']

    # Fix graph size

    fig = plt.figure(figsize=(12,7))

    # 1. Individual sentence scores as data points

    by_sentence_x = range(1,corpus_size+1)
    by_sentence_y = stats_data['individual_sentence_scores']

    if metric != "METEOR":
        corpus_score = stats_data['total_corpus_score']
        by_sentence_corpus = [corpus_score for i in range(corpus_size)]

    sentence_average = np.mean(stats_data['individual_sentence_scores'])
    by_sentence_average = [sentence_average for i in range(corpus_size)]

    plt.clf()
    plt.grid(True)

    plt.xlabel("Sentence index")
    plt.ylabel(f"{metric} score")
    plt.title(f"Individual {metric} sentence scores for the {formatted_version_names[version]} translation")

    if metric == "BLEU" or metric == "METEOR" : # These 2 go between 0 and 1 so we can do this
        # For NIST, there is no cap, so the scale must change with the graph too
        plt.gca().set_ylim(bottom=-0.02, top=1.02)

    plt.scatter(by_sentence_x,by_sentence_y, label=f"Individual {formatted_version_names[version]} {metric} sentence scores")

    if metric != "METEOR":
        plt.plot(by_sentence_x,by_sentence_corpus,label=f"{formatted_version_names[version]} corpus-size {metric} score : {corpus_score:.2f}")

    plt.plot(by_sentence_x,by_sentence_average,color="black",label=f"Average of {formatted_version_names[version]} {metric} sentence scores : {sentence_average:.2f}")

    ################### DIMENSIONING ###################

    legend_x = -0.05
    legend_y = 1.15
    ncol = 3

    if metric == "METEOR":
        legend_x = 0.075

    if version == "GT":
        if metric == "METEOR":
            legend_x -= 0.07
        else:
            legend_x += 0.07
            legend_y += 0.015
        ncol = 2

    ############### DIMENSIONING OVER ##################

    plt.legend(loc='upper left', bbox_to_anchor=(legend_x, legend_y),
          ncol=ncol, fancybox=True, shadow=True)

    plt.savefig(individual_score_path[version][metric])

    # 2. Correlation of sentence scores and sentence lengths

    by_length_x = np.sort(stats_data['mt_sentence_lengths'])
    by_length_y = by_sentence_y

    plt.clf()
    plt.grid(True)

    plt.xlabel(f"Sentence length in {version} translation (words)")
    plt.ylabel(f"{metric} score")
    plt.title(f"{metric} sentence scores by sentence length")

    plt.scatter(by_length_x,by_length_y)

    plt.savefig(score_by_length_path[version][metric])

    # 3. Repartition of sentence scores

    def calculate_proportion_under_ref(ref_val,arr):

        prop = 0

        for elem in arr:
            if elem > ref_val:
                break # assumes sorted array
            prop += 1

        return prop / len(arr)

    repartition_x = np.sort(by_sentence_y) # fulfill assumption
    repartition_y = [calculate_proportion_under_ref(repartition_x[i],repartition_x) for i in range(corpus_size)]

    plt.clf()
    plt.grid(True)

    if metric == "NIST" : # ensure that we have precision to 0.5 on the x axis, otherwise it would be so for GT but not DeepL
        plt.xticks(np.arange(min(repartition_x), max(repartition_x)+1, 0.5))
    else: # since BLEU and METEOR go between 0 and 1, we use that, and 0.1 precision is the one we want
        plt.xticks(np.arange(0, 2, 0.1))

    plt.xlabel(f"{metric} score")

    plt.ylabel("Proportion of scores under value")
    plt.title(f"Repartition of {metric} score for {formatted_version_names[version]} translation")

    plt.scatter(repartition_x,repartition_y)

    average_label = f"Average of {formatted_version_names[version]} {metric} sentence scores : {sentence_average:.2f}"
    if metric == "NIST": # NIST scores increase with length so corpus length will be way off center from sentence scores and deform the graph
    # Therefore it's better not to plot it and write it here
        average_label += f" (corpus-size score : {corpus_score:.2f})"

    plt.axvline(sentence_average,color="black", label=average_label)

    if metric == "BLEU": # METEOR doesn't have a corpus-size score to plot.
        plt.axvline(corpus_score,label=f"{formatted_version_names[version]} corpus-size {metric} score : {corpus_score:.2f}")

    plt.legend(fancybox=True, shadow=True)

    plt.savefig(score_repartition_path[version][metric])

    # 4. Length repartition (proof of concept)

    global human_length_plotted

    if not human_length_plotted:

        human_length_plotted = True

        human_length_repartition_x = np.sort(stats_data['human_sentence_lengths']);
        human_length_repartition_y = [calculate_proportion_under_ref(human_length_repartition_x[i],human_length_repartition_x) for i in range(corpus_size)]

        plt.clf()
        plt.grid(True)

        plt.xlabel("Human sentence length (words)")
        plt.ylabel("Proportion of human sentence lengths under value")

        plt.title("Repartition of sentence lengths in reference human translation")

        plt.scatter(human_length_repartition_x, human_length_repartition_y)

        plt.savefig(human_length_repartition_path)

    mt_length_repartition_x = np.sort(stats_data['mt_sentence_lengths'])
    mt_length_repartition_y = [calculate_proportion_under_ref(mt_length_repartition_x[i],mt_length_repartition_x) for i in range(corpus_size)]

    plt.clf()
    plt.grid(True)

    plt.xlabel(f"{version} sentence length (words)")
    plt.ylabel(f"Proportion of {version} sentence lengths under value")

    plt.title(f"Repartition of sentence lengths in {version} translation")

    plt.scatter(mt_length_repartition_x,mt_length_repartition_y)

    plt.savefig(mt_length_repartition_path[version])

if __name__ == "__main__":
    # create_debug_files()

    for fkey in first_level_keys :
        for skey in second_level_keys:
            write_results_as_json(fkey,skey)
            draw_graphs_from_json(fkey,skey)
