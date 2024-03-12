# !/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
from helpers import read_df

doc_path = "data/spacy_docs"
if not os.path.exists(doc_path):
    os.makedirs(doc_path)


def make_docs(tup_abstract_sdg, target_file, cats):
    nlp = spacy.load("en_core_web_trf")
    docs = DocBin()

    for doc, label in nlp.pipe(tup_abstract_sdg, as_tuples=True):
        for cat in cats:
            doc.cats[cat] = 1 if cat == label else 0
        docs.add(doc)

    docs.to_disk(target_file)

    return docs


def create_dev_set(df, dataset):
    if dataset == 'zora':
        # SDG 4 only appears in train set once - need to leave it out of dev set
        hapax_legomena = df[df.sdg == '4']
        reduced_df = df.drop(hapax_legomena.index)

        train_df, dev_df = train_test_split(reduced_df, test_size=0.1,
                                            stratify=reduced_df['sdg'], random_state=13)
        # Add SDG 4 to train_df
        train_df = pd.concat([train_df, hapax_legomena], ignore_index=False)

        # SDGs 1, 6, and 11 too infrequent (4, 2, 2) to appear in dev_df with stratify - need to split manually
        sdg_1 = train_df[train_df.sdg == '1']
        sdg_6 = train_df[train_df.sdg == '6']
        sdg_11 = train_df[train_df.sdg == '11']

        # Remove one of each from train_df and add to dev_df
        train_df.drop([*sdg_1.head(1).index, *sdg_6.head(1).index, *sdg_11.head(1).index], inplace=True)
        dev_df = pd.concat([dev_df, sdg_1.head(1), sdg_6.head(1), sdg_11.head(1)])
    else:
        train_df, dev_df = train_test_split(df, test_size=0.1,
                                            stratify=df['sdg'], random_state=13)

    return train_df, dev_df


def main():
    """
    Create dev sets for spaCy models, make spaCy Docs for model input.
    """

    """ ZORA """
    print("\n --- ZORA ---")
    dataset = 'zora'
    cats = [str(i) for i in range(1, 18)]

    # labels need to be strings for spaCy
    train_df = read_df(f"data/train_test/{dataset}_train_raw.tsv",
                       coltypes={'sdg': 'string',
                                 'faculty': 'string',
                                 'year': 'string',
                                 'abstract': 'string',
                                 'author': 'string'},
                       index_col=0)
    test_df = read_df(f"data/train_test/{dataset}_test_raw.tsv",
                      coltypes={'sdg': 'string',
                                'faculty': 'string',
                                'year': 'string',
                                'abstract': 'string',
                                'author': 'string'},
                      index_col=0)

    # Drop extra features
    train_df.drop(columns=['faculty', 'year', 'author'], inplace=True)
    test_df.drop(columns=['faculty', 'year', 'author'], inplace=True)

    # Take 10% of train for dev
    train_df, dev_df = create_dev_set(train_df, dataset=dataset)

    # Get inputs (abstracts) and targets (labels)
    X_train = train_df['abstract'].values
    y_train = train_df['sdg'].values
    X_dev = dev_df['abstract'].values
    y_dev = dev_df['sdg'].values
    X_test = test_df['abstract'].values
    y_test = test_df['sdg'].values

    # Make spaCy Doc files
    train_path, dev_path, test_path = (f"{doc_path}/{dataset}_train.spacy",
                                       f"{doc_path}/{dataset}_dev.spacy",
                                       f"{doc_path}/{dataset}_test.spacy")
    make_docs(list(zip(X_train, y_train)), train_path, cats=cats)
    print("Train docs done!")
    make_docs(list(zip(X_dev, y_dev)), dev_path, cats=cats)
    print("Dev docs done!")
    make_docs(list(zip(X_test, y_test)), test_path, cats=cats)
    print("Test docs done!")


    """ OSDG """
    print("\n --- OSDG ---")
    dataset = 'osdg'
    cats = [str(i) for i in range(1, 17)]

    # labels need to be strings for spaCy
    train_df = read_df(f"data/train_test/{dataset}_train_raw.tsv",
                       coltypes={'sdg': 'string',
                                 'abstract': 'string'},
                       index_col=0)
    test_df = read_df(f"data/train_test/{dataset}_test_raw.tsv",
                      coltypes={'sdg': 'string',
                                'abstract': 'string'},
                      index_col=0)

    # Take 10% of train for dev
    train_df, dev_df = create_dev_set(train_df, dataset=dataset)

    # Get inputs (abstracts) and targets (labels)
    X_train = train_df['abstract'].values
    y_train = train_df['sdg'].values
    X_dev = dev_df['abstract'].values
    y_dev = dev_df['sdg'].values
    X_test = test_df['abstract'].values
    y_test = test_df['sdg'].values

    # Make spaCy Doc files
    train_path, dev_path, test_path = (f"{doc_path}/{dataset}_train.spacy",
                                       f"{doc_path}/{dataset}_dev.spacy",
                                       f"{doc_path}/{dataset}_test.spacy")
    make_docs(list(zip(X_train, y_train)), train_path, cats=cats)
    print("Train docs done!")
    make_docs(list(zip(X_dev, y_dev)), dev_path, cats=cats)
    print("Dev docs done!")
    make_docs(list(zip(X_test, y_test)), test_path, cats=cats)
    print("Test docs done!")


    """ CONCAT """
    print("\n --- CONCAT ---")
    dataset = 'concat'
    cats = [str(i) for i in range(1, 18)]

    # labels need to be strings for spaCy
    train_df = read_df(f"data/train_test/{dataset}_train_raw.tsv",
                       coltypes={'sdg': 'string',
                                 'faculty': 'string',
                                 'year': 'string',
                                 'abstract': 'string'})
    test_df = read_df(f"data/train_test/{dataset}_test_raw.tsv",
                      coltypes={'sdg': 'string',
                                'faculty': 'string',
                                'year': 'string',
                                'abstract': 'string'})

    # Drop extra features
    train_df.drop(columns=['faculty', 'year', 'subset', 'subset_index'], inplace=True)
    test_df.drop(columns=['faculty', 'year', 'subset', 'subset_index'], inplace=True)

    # Take 10% of train for dev
    train_df, dev_df = create_dev_set(train_df, dataset=dataset)

    # Get inputs (abstracts) and targets (labels)
    X_train = train_df['abstract'].values
    y_train = train_df['sdg'].values
    X_dev = dev_df['abstract'].values
    y_dev = dev_df['sdg'].values
    X_test = test_df['abstract'].values
    y_test = test_df['sdg'].values

    # Make spaCy Doc files
    train_path, dev_path, test_path = (f"{doc_path}/{dataset}_train.spacy",
                                       f"{doc_path}/{dataset}_dev.spacy",
                                       f"{doc_path}/{dataset}_test.spacy")
    make_docs(list(zip(X_train, y_train)), train_path, cats=cats)
    print("Train docs done!")
    make_docs(list(zip(X_dev, y_dev)), dev_path, cats=cats)
    print("Dev docs done!")
    make_docs(list(zip(X_test, y_test)), test_path, cats=cats)
    print("Test docs done!")


if __name__ == "__main__":
    main()
