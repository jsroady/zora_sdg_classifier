#!/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

from pyspark.sql import SparkSession
from helpers import (read_df, create_spark_dfs, combine_tt_dfs, word_vectorize, string_index, one_hot_encode,
                     split_tt_dfs, transform_features, roby_train, roby_predict, write_df)
from evaluate import generate_classification_report, write_report

spark = SparkSession.builder.master('local[*]').appName('RobyZORA').getOrCreate()
labels = range(1, 18)
dataset = 'zora'
model_path = f"data/models/roby/{dataset}"


def main():
    train_df = read_df(f"data/train_test/{dataset}_train_clean.tsv",
                       coltypes={'sdg': 'int',
                                 'faculty': 'string',
                                 'year': 'string',
                                 'abstract': 'string',
                                 'author': 'string'},
                       index_col=0)
    test_df = read_df(f"data/train_test/{dataset}_test_clean.tsv",
                      coltypes={'sdg': 'int',
                                'faculty': 'string',
                                'year': 'string',
                                'abstract': 'string',
                                'author': 'string'},
                       index_col=0)

    """ Prep data """
    spark_train, spark_test = create_spark_dfs(train_df, test_df, spark)
    combined_df = combine_tt_dfs(spark_train, spark_test)

    # w2v embedding
    word_vectorized_df, w2v_vectors_df = word_vectorize(combined_df, 'abstract', 'abstract_vectorized')

    # String indexing and one-hot encoding additional features
    indexed_df = string_index(word_vectorized_df, 'faculty', 'faculty_indexed')
    indexed_df = string_index(indexed_df, 'year', 'year_indexed')
    indexed_df = string_index(indexed_df, 'author', 'author_indexed')

    encoded_df = one_hot_encode(indexed_df, 'faculty_indexed', 'faculty_encoded')
    encoded_df = one_hot_encode(encoded_df, 'year_indexed', 'year_encoded')
    encoded_df = one_hot_encode(encoded_df, 'author_indexed', 'author_encoded')

    # Re-split train and test sets
    train_df_encoded, test_df_encoded = split_tt_dfs(encoded_df)


    """ Train/test: all features ('author' ablation) """
    condition = 'authors'
    model_path_condition = f"{model_path}-{condition}"

    # Transform features
    train_df_transformed, test_df_tranformed, vecAssembler = transform_features(train_df_encoded, test_df_encoded)

    # Train
    roby_model = roby_train(train_df_transformed, vecAssembler, model_path_condition)

    # Test
    predictions = roby_predict(roby_model, test_df_tranformed, test_df)
    write_df(predictions, f"data/predictions/roby/{dataset}-{condition}_preds.tsv", index=False)

    # Evaluate
    y_true = predictions['label']
    y_pred = predictions['prediction']

    report_dict = generate_classification_report(y_true, y_pred, labels, as_dict=True)
    write_report(model_name='roby', train_set=dataset, test_set=condition, report=report_dict)

    report_text = generate_classification_report(y_true, y_pred, labels, as_dict=False)
    print(f"\n --- {dataset.upper()} feature condition: {condition} --- \n")
    print(report_text)


    """ Train/test: faculty, year (default) """
    # Transform features
    train_df_encoded = train_df_encoded.drop('author_encoded')
    test_df_encoded = test_df_encoded.drop('author_encoded')
    train_df_transformed, test_df_tranformed, vecAssembler = transform_features(train_df_encoded, test_df_encoded)

    # Train
    roby_model = roby_train(train_df_transformed, vecAssembler, model_path)
    write_df(w2v_vectors_df, f"{model_path}/w2v_vocab.tsv", index=False, header=False)

    # Test
    predictions = roby_predict(roby_model, test_df_tranformed, test_df)
    write_df(predictions, f"data/predictions/roby/{dataset}-{dataset}_preds.tsv", index=True)

    # Evaluate
    y_true = predictions['label']
    y_pred = predictions['prediction']

    report_dict = generate_classification_report(y_true, y_pred, labels, as_dict=True)
    write_report(model_name='roby', train_set=dataset, test_set=dataset, report=report_dict)

    report_text = generate_classification_report(y_true, y_pred, labels, as_dict=False)
    print(f"\n --- {dataset.upper()} feature condition: faculty, year (default) --- \n")
    print(report_text)


    """ Train/test: no additional features """
    condition = 'none'
    model_path_condition = f"{model_path}-{condition}"

    # Transform features
    train_df_encoded = train_df_encoded.drop('faculty_encoded', 'year_encoded')
    test_df_encoded = test_df_encoded.drop('faculty_encoded', 'year_encoded')
    train_df_transformed, test_df_tranformed, vecAssembler = transform_features(train_df_encoded, test_df_encoded)

    # Train
    roby_model = roby_train(train_df_transformed, vecAssembler, model_path_condition)

    # Test
    predictions = roby_predict(roby_model, test_df_tranformed, test_df)
    write_df(predictions, f"data/predictions/roby/{dataset}-{condition}_preds.tsv", index=False)

    # Evaluate
    y_true = predictions['label']
    y_pred = predictions['prediction']

    report_dict = generate_classification_report(y_true, y_pred, labels, as_dict=True)
    write_report(model_name='roby', train_set=dataset, test_set=condition, report=report_dict)

    report_text = generate_classification_report(y_true, y_pred, labels, as_dict=False)
    print(f"\n --- {dataset.upper()} feature condition: {condition} --- \n")
    print(report_text)


if __name__ == "__main__":
    main()
