# !/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

from langdetect import detect
from helpers import read_df, write_df

def strip_quotes(original_file, stripped_file):
    """ Strip opening and closing quotes from each line of the original OSDG file. """
    lines = []

    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lstrip('"')
            line = line.rstrip('"\n')
            lines.append(line)

    with open(stripped_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def filter_lang(df):
    """ Drop non-English texts. """
    def language_detect(x):
        lang = detect(x)
        return lang

    df['language'] = df['abstract'].apply(language_detect)
    df = df.astype({'language': 'string'})
    df = df.drop(df[df.language != 'en'].index)
    df = df.drop(columns=['language'])

    return df


def filter_agreement(df):
    """ Drop texts with low IAA from OSDG data. """
    df_filtered = df[df['agreement'] >= 0.5]
    return df_filtered


def filter_negative_samples(df):
    """ Drop texts with a high probability of being negative samples of their label. """
    df_prob_neg = df.copy()
    df_prob_neg['prob_neg'] = 0.0
    df_prob_neg['prob_neg'] = (df_prob_neg['labels_negative'] /
                               (df_prob_neg['labels_negative'] + df_prob_neg['labels_positive'])
                               )
    df_filtered = df_prob_neg[df_prob_neg['prob_neg'] <= 0.5]

    return df_filtered


def regex_clean(df):
    """ Clean tabs and linebreaks from texts, 'SDG' and decimals from labels, and change coltype to int of ZORA data. """
    df['abstract'] = df['abstract'].str.replace(r'\s+', ' ', regex=True)  # remove tabs, linebreaks
    df['sdg'] = df['sdg'].str.replace(r'SDG', '')  # remove 'SDG'
    df['sdg'] = df['sdg'].str.replace(r'.00', '')  # remove decimals
    df = df.astype({'sdg': 'int'})  # change coltype to int

    return df


def remove_missing_abstracts(df):
    """ Drop lines with missing abstracts from ZORA data. """
    df[['abstract']] = df[['abstract']].fillna(value='unknown')
    df = df.drop(df[df.abstract == 'unknown'].index)

    return df


def extract_concat_title(df):
    """ Concatenate titles and abstracts in ZORA data. """
    df['title'] = df['title'].str.replace('?', '.')
    df['title'] = df['title'].str.replace('!', '.')

    df['title'] = df.apply(lambda x: x['title'][0:x['title'].find('.')], axis=1)
    df = df.astype({'title': 'string'})

    df['abstract'] = df['title'] + '. ' + df['abstract']
    df = df.astype({'abstract': 'string'})
    df = df.drop(columns=['title'])

    return df


def main():
    """
    Basic preprocessing of original datasets.
    """

    """ ZORA """
    zora_original = read_df("data/full_datasets/original/zora_original.tsv",
                            coltypes={'sdg': 'string',
                                      'author': 'string',
                                      'faculty': 'string',
                                      'year': 'string',
                                      'title': 'string',
                                      'citation2': 'string',
                                      'abstract': 'string'},
                            skiprows=[0],
                            new_names=['sdg', 'author', 'faculty', 'year', 'title', 'citation2', 'abstract']
                            )
    zora_original.drop(columns=['citation2'], inplace=True)

    zora_regex_cleaned = regex_clean(zora_original)
    zora_missing_abstracts_removed = remove_missing_abstracts(zora_regex_cleaned)
    zora_lang_filtered = filter_lang(zora_missing_abstracts_removed)
    zora_titles_concatenated = extract_concat_title(zora_lang_filtered)

    write_df(zora_titles_concatenated, "data/full_datasets/zora_preproc.tsv", index=True)


    """ OSDG """
    strip_quotes("data/full_datasets/original/osdg_original.tsv",
                 "data/full_datasets/original/osdg_original_stripped.tsv")
    osdg_stripped = read_df("data/full_datasets/original/osdg_original_stripped.tsv",
                            coltypes={'doi': 'string',
                                      'text_id': 'string',
                                      'text': 'string',
                                      'sdg': 'int',
                                      'labels_negative': 'int',
                                      'labels_positive': 'int',
                                      'agreement': 'float'}
                            )
    osdg_stripped.rename(columns={'text': 'abstract'}, inplace=True)

    osdg_en = filter_lang(osdg_stripped)
    osdg_iaa_50 = filter_agreement(osdg_en)
    osdg_iaa_50_pos = filter_negative_samples(osdg_iaa_50)

    write_df(osdg_iaa_50_pos, "data/full_datasets/osdg_preproc.tsv", index=True)


if __name__ == "__main__":
    main()
