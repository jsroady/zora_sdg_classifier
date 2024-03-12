#!/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, regexp_replace
from helpers import (read_df, create_spark_dfs, combine_tt_dfs, word_vectorize, split_tt_dfs, transform_features,
                     roby_train, roby_predict, write_df)
from evaluate import generate_classification_report, write_report

spark = SparkSession.builder.master('local[*]').appName('RobyOSDG').getOrCreate()
train_set = 'osdg'
model_path = f"data/models/roby/{train_set}"


def main():
    train_df = read_df("data/train_test/osdg_train_clean.tsv",
                       coltypes={'sdg': 'int',
                                 'abstract': 'string'},
                       index_col=0)
    test_df = read_df("data/train_test/osdg_test_clean.tsv",
                      coltypes={'sdg': 'int',
                                'abstract': 'string'},
                       index_col=0)

    """ Prep data """
    spark_train, spark_test = create_spark_dfs(train_df, test_df, spark)
    combined_df = combine_tt_dfs(spark_train, spark_test)

    # w2v embedding
    encoded_df, w2v_vectors_df = word_vectorize(combined_df, 'abstract', 'abstract_vectorized')
    train_df_encoded, test_df_encoded = split_tt_dfs(encoded_df)

    # Transform features
    train_df_transformed, test_df_tranformed, vecAssembler = transform_features(train_df_encoded, test_df_encoded)


    """ Train """
    roby_model = roby_train(train_df_transformed, vecAssembler, model_path)
    write_df(w2v_vectors_df, f"{model_path}/w2v_vocab.tsv", index=False, header=False)


    """ Test - OSDG """
    labels = range(1, 17)
    test_set = 'osdg'

    predictions = roby_predict(roby_model, test_df_tranformed, test_df)
    write_df(predictions, f"data/predictions/roby/{train_set}-{test_set}_preds.tsv", index=False)


    """ Evaluate - OSDG """
    y_true = predictions['label']
    y_pred = predictions['prediction']

    report_dict = generate_classification_report(y_true, y_pred, labels, as_dict=True)
    write_report(model_name='roby', train_set=train_set, test_set=test_set, report=report_dict)

    report_text = generate_classification_report(y_true, y_pred, labels, as_dict=False)
    print(f"\n --- {test_set.upper()} results --- \n")
    print(report_text)


    """ Test - ZORA """
    labels = range(1, 18)
    test_set = 'zora'

    test_df = read_df(f"data/train_test/zora_test_clean.tsv",
                      coltypes={'sdg': 'int',
                                'faculty': 'string',
                                'year': 'string',
                                'abstract': 'string',
                                'author': 'string'},
                       index_col=0)

    # Drop additional features
    test_df.drop(columns=['faculty', 'year', 'author'], inplace=True)
    spark_test = spark.createDataFrame(test_df)

    # Fix coltype
    spark_test = spark_test.withColumn('abstract', regexp_replace('abstract', r'^\[|\]$', ''))
    spark_test = spark_test.withColumn('abstract', expr("split(abstract, ', ')"))

    # w2v embedding
    test_df_encoded, w2v_vectors_df = word_vectorize(spark_test, 'abstract', 'abstract_vectorized')

    # Transform features
    test_df_tranformed = test_df_encoded.withColumnRenamed('sdg', 'label')

    # Get predictions
    predictions = roby_predict(roby_model, test_df_tranformed, test_df)
    write_df(predictions, f"data/predictions/roby/{train_set}-{test_set}_preds.tsv", index=False)


    """ Evaluate - ZORA """
    y_true = predictions['label']
    y_pred = predictions['prediction']

    report_dict = generate_classification_report(y_true, y_pred, labels, as_dict=True)
    write_report(model_name='roby', train_set=train_set, test_set=test_set, report=report_dict)

    report_text = generate_classification_report(y_true, y_pred, labels, as_dict=False)
    print(f"\n --- {test_set.upper()} results --- \n")
    print(report_text)


if __name__ == "__main__":
    main()
