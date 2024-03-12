# !/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

import os
import re
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from helpers import read_df, write_df

stop = stopwords.words('english')
if not os.path.exists("data/train_test"):
    os.makedirs("data/train_test")
if not os.path.exists("data/dists"):
    os.makedirs("data/dists")

def clean_text(c):
    """ Standard NLP cleaning - lowercase, remove links, punctuation, digits, tabs, and linebreaks. """
    c = c.lower()
    c = re.sub(r'http[^\s]+', '', c)  # remove links
    c = re.sub(r'\s+', ' ', c)  # remove tabs, linebreaks
    c = re.sub(r'\d+', '', c)  # remove numbers
    c = re.sub(r'[^\w\s]+', '', c)  # remove punctuation/symbols

    return c


def clean_author(c):
    """ Standard NLP cleaning for names - lowercase, remove punctuation, single characters, and spaces. """
    c = c.lower()
    c = re.sub(r'[^\w\s]+', '', c)  # remove punctuation/symbols
    c = re.sub(r'\b\w\b', '', c)  # remove single characters
    c = re.sub(r'\s+', ' ', c)  # remove multiple spaces
    c = re.sub(' ', '', c)

    return c


def remove_stopwords(c):
    """ Standard NLP cleaning - remove stopwords. """
    c_tokenized = word_tokenize(c)
    c_cleaned = [w for w in c_tokenized if w not in stop]

    return c_cleaned


def plot_class_dists(subtitle, train_df, test_df, labels):
    """ Plot class distributions with train and test counts side-by-side. """
    train_counts = train_df.sdg.value_counts().sort_index().tolist()
    test_counts = test_df.sdg.value_counts().sort_index().tolist()

    if subtitle[0] == 'Z':
        fig, ax = plt.subplots(figsize=(7.4, 5))
        y_lim = math.ceil(max(train_counts) / 10) * 10
        y_ticks = np.arange(0, y_lim + 1, 10)
        fontsize = 8
        tt_counts = {'Train (70%, 280 items)': train_counts, 'Test (30%, 121 items)': test_counts}
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        y_lim = math.ceil(max(train_counts) / 500) * 500
        y_ticks = np.arange(0, y_lim + 1, 500)
        fontsize = 7
        if subtitle[0] == 'O':
            tt_counts = {"Train (70%, 18'248 items)": train_counts, "Test (30%, 7'821 items)": test_counts}
        else:
            tt_counts = {"Train (70%, 18'528 items)": train_counts, "Test (30%, 7'942 items)": test_counts}

    width = 0.4
    multiplier = 0
    x = np.arange(len(labels))
    x_ticks_positions = x + width / 2

    for subset, counts in tt_counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, counts, width, label=subset)
        ax.bar_label(rects, padding=3, fontsize=fontsize)
        multiplier += 1

    ax.set_title(f"Train/test class proportions: {subtitle}")
    ax.legend(loc='upper left', ncols=2)
    ax.set_xlabel('SDG')

    label_vals = {}
    for label, count in enumerate(train_counts, start=1):
        label_vals[label] = round((count / sum(train_counts) * 100), 1)

    x_labels = [f'{k}\n({v}%)' for k,v in label_vals.items()]

    ax.set_xticks(x_ticks_positions, x_labels, fontsize=fontsize)
    ax.set_ylabel('Num. instances')
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, y_lim)

    plt.tight_layout()

    return plt


def main():
    """
    Train/test split, raw/clean versions, create concatenated datasets, plot class distributions.
    """

    zora_preproc = read_df("data/full_datasets/zora_preproc.tsv",
                           coltypes={'sdg': 'string',
                                     'author': 'string',
                                     'faculty': 'string',
                                     'year': 'string',
                                     'abstract': 'string'},
                           index_col=0)
    osdg_preproc = read_df("data/full_datasets/osdg_preproc.tsv",
                           coltypes={'doi': 'string',
                                     'text_id': 'string',
                                     'abstract': 'string',
                                     'sdg': 'int',
                                     'labels_negative': 'int',
                                     'labels_positive': 'int',
                                     'agreement': 'float',
                                     'prob_neg': 'float'},
                           index_col=0)
    osdg_preproc.drop(columns=['doi', 'text_id', 'labels_negative', 'labels_positive', 'agreement', 'prob_neg'],
                      inplace=True)


    """ Train/test split full datasets """
    zora_train_raw, zora_test_raw = train_test_split(zora_preproc, test_size=0.3,
                                                     stratify=zora_preproc['sdg'], random_state=13)
    zora_train_raw.sort_index(inplace=True)
    zora_test_raw.sort_index(inplace=True)

    osdg_train_raw, osdg_test_raw = train_test_split(osdg_preproc, test_size=0.3,
                                                     stratify=osdg_preproc['sdg'], random_state=13)
    osdg_train_raw.sort_index(inplace=True)
    osdg_test_raw.sort_index(inplace=True)


    """ Create cleaned versions for Roby """
    zora_train_clean = zora_train_raw.copy()
    zora_test_clean = zora_test_raw.copy()
    osdg_train_clean = osdg_train_raw.copy()
    osdg_test_clean = osdg_test_raw.copy()

    zora_train_clean['abstract'] = zora_train_clean['abstract'].apply(clean_text)
    zora_train_clean['abstract'] = zora_train_clean['abstract'].apply(remove_stopwords)
    zora_test_clean['abstract'] = zora_test_clean['abstract'].apply(clean_text)
    zora_test_clean['abstract'] = zora_test_clean['abstract'].apply(remove_stopwords)
    zora_train_clean['author'] = zora_train_clean['author'].apply(clean_author)
    zora_test_clean['author'] = zora_test_clean['author'].apply(clean_author)

    osdg_train_clean['abstract'] = osdg_train_clean['abstract'].apply(clean_text)
    osdg_train_clean['abstract'] = osdg_train_clean['abstract'].apply(remove_stopwords)
    osdg_test_clean['abstract'] = osdg_test_clean['abstract'].apply(clean_text)
    osdg_test_clean['abstract'] = osdg_test_clean['abstract'].apply(remove_stopwords)


    """ Write out raw and clean files """
    write_df(zora_train_raw, "data/train_test/zora_train_raw.tsv", index=True)
    write_df(zora_test_raw, "data/train_test/zora_test_raw.tsv", index=True)
    write_df(zora_train_clean, "data/train_test/zora_train_clean.tsv", index=True)
    write_df(zora_test_clean, "data/train_test/zora_test_clean.tsv", index=True)

    write_df(osdg_train_raw, "data/train_test/osdg_train_raw.tsv", index=True)
    write_df(osdg_test_raw, "data/train_test/osdg_test_raw.tsv", index=True)
    write_df(osdg_train_clean, "data/train_test/osdg_train_clean.tsv", index=True)
    write_df(osdg_test_clean, "data/train_test/osdg_test_clean.tsv", index=True)


    """ Create and write out concatenated raw and clean files with original subset information """
    zora_train_raw.drop(columns=['author'], inplace=True)
    zora_train_clean.drop(columns=['author'], inplace=True)
    zora_test_raw.drop(columns=['author'], inplace=True)
    zora_test_clean.drop(columns=['author'], inplace=True)

    zora_train_raw['subset'] = 'zora'
    zora_train_clean['subset'] = 'zora'
    zora_test_raw['subset'] = 'zora'
    zora_test_clean['subset'] = 'zora'

    osdg_train_raw['subset'] = 'osdg'
    osdg_train_clean['subset'] = 'osdg'
    osdg_test_raw['subset'] = 'osdg'
    osdg_test_clean['subset'] = 'osdg'

    concat_train_raw = pd.concat([zora_train_raw, osdg_train_raw], ignore_index=False)
    concat_test_raw = pd.concat([zora_test_raw, osdg_test_raw], ignore_index=False)
    concat_train_clean = pd.concat([zora_train_clean, osdg_train_clean], ignore_index=False)
    concat_test_clean = pd.concat([zora_test_clean, osdg_test_clean], ignore_index=False)

    concat_train_raw['subset_index'] = concat_train_raw.index
    concat_train_raw.reset_index(drop=True, inplace=True)
    concat_test_raw['subset_index'] = concat_test_raw.index
    concat_test_raw.reset_index(drop=True, inplace=True)
    concat_train_clean['subset_index'] = concat_train_clean.index
    concat_train_clean.reset_index(drop=True, inplace=True)
    concat_test_clean['subset_index'] = concat_test_clean.index
    concat_test_clean.reset_index(drop=True, inplace=True)

    write_df(concat_train_raw, "data/train_test/concat_train_raw.tsv", index=False, na_rep='NA')
    write_df(concat_test_raw, "data/train_test/concat_test_raw.tsv", index=False, na_rep='NA')
    write_df(concat_train_clean, "data/train_test/concat_train_clean.tsv", index=False, na_rep='NA')
    write_df(concat_test_clean, "data/train_test/concat_test_clean.tsv", index=False, na_rep='NA')


    """ Plot class dists for each train_set """
    zora_dist_plot = plot_class_dists("ZORA (401 items)", zora_train_raw, zora_test_raw,
                                      labels=range(1, 18))
    zora_dist_plot.savefig("data/dists/zora_class_dist.png")

    osdg_dist_plot = plot_class_dists("OSDG (26'069 items)", osdg_train_raw, osdg_test_raw,
                                      labels=range(1, 17))
    osdg_dist_plot.savefig("data/dists/osdg_class_dist.png")

    concat_dist_plot = plot_class_dists("ZORA + OSDG (26'470 items)", concat_train_raw, concat_test_raw,
                                        labels=range(1, 18))
    concat_dist_plot.savefig("data/dists/concat_class_dist.png")


if __name__ == "__main__":
    main()
