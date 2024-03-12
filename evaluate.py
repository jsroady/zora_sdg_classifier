# !/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

import json
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data_conditions = [('Z', 'Z', 'zora-zora'), ('O', 'O', 'osdg-osdg'), ('O', 'Z', 'osdg-zora'),
                   ('ZO', 'ZO', 'concat-concat'), ('ZO', 'Z', 'concat-zora')]
seeds = ['0', '13', '27', '30', '77']

def generate_classification_report(y_true, y_pred, labels, as_dict):
    """
    Evaluate predictions with sklearn's classification report (by-label, macro, and weighted averages),
    return either as dict or preformatted text.
    """
    target_names = [f"sdg_{i}" if (i > 9) or (i == -1) else f"sdg_0{i}" for i in labels]

    if as_dict:
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                               zero_division=0.0, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                               digits=4, zero_division=0.0, output_dict=False)

    return report


def write_report(model_name, train_set, test_set, report):
    """ Write out sklearn classification report as a .json file. """
    if test_set == 'authors' or test_set == 'none':
        outfile = f"data/models/{model_name}/{train_set}-{test_set}/test_{test_set}_eval.json"
    else:
        outfile = f"data/models/{model_name}/{train_set}/test_{test_set}_eval.json"

    with open(outfile, 'w') as f:
        json.dump(report, f)


def read_predictions(file):
    """ Read predictions dfs and return labels and predictions as arrays for confusion matrices. """
    df = pd.read_csv(file, sep='\t', encoding='utf-8')
    df = df.astype({'label': 'int', 'prediction': 'int'})

    y_true = np.array(df['label'])
    y_pred = np.array(df['prediction'])

    return y_true, y_pred


def sklearn_cm(data_condition, model_name, seed: Optional[str]=None):
    """ Create confusion matrices from predictions files. """
    model_path = model_name.lower()
    base_inpath = f"data/predictions/{model_path}"
    base_outpath = f"data/confusion_matrices/{model_path}"
    labels = range(1,17) if data_condition[2] == 'osdg-osdg' else range(1, 18)

    if seed:
        y_true, y_pred = read_predictions(f"{base_inpath}/{data_condition[2]}_preds-s{seed}.tsv")
    else:
        y_true, y_pred = read_predictions(f"{base_inpath}/{data_condition[2]}_preds.tsv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.T  # transpose to have y_true on x and y_pred on y axes

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.9, 6.1))
    disp.plot(ax=ax)
    disp.im_.colorbar.remove()

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')

    if seed:
        plt.title(f"{data_condition[0]}-{model_name}-{data_condition[1]}-{seed}", pad=12)
        plt.savefig(f"{base_outpath}/{data_condition[2]}_cm-{seed}.png")
    else:
        plt.title(f"{data_condition[0]}-{model_name}-{data_condition[1]}", pad=12)
        plt.savefig(f"{base_outpath}/{data_condition[2]}_cm.png")


def main():
    """
    Create confusion matrices from predictions files.
    """

    ''' Roby and TextCat'''
    for c in data_conditions:
        sklearn_cm(c, 'Roby')
        sklearn_cm(c, 'TextCat')

    ''' RoBERTa and SciBERT'''
    for c in data_conditions:
        if c == ('O', 'Z', 'osdg-zora') or c == ('O', 'O', 'osdg-osdg'):
            sklearn_cm(c, 'SciBERT')
            for s in seeds:
                sklearn_cm(c, 'RoBERTa', s)
        elif c == ('ZO', 'Z', 'concat-zora') or c == ('ZO', 'ZO', 'concat-concat'):
            sklearn_cm(c, 'RoBERTa')
            for s in seeds:
                sklearn_cm(c, 'SciBERT', s)
        else:
            sklearn_cm(c, 'RoBERTa')
            sklearn_cm(c, 'SciBERT')


if __name__ == "__main__":
    main()
