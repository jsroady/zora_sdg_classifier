# !/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Author: Jessica Roady

import os
import pandas as pd
import numpy as np
from typing import Optional
from pyspark.sql.functions import expr, lit, regexp_replace
from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression


def read_df(filepath: str,
            coltypes: dict,
            keep_default_na: Optional[bool]=True,
            skiprows: Optional[list[int]]=None,
            new_names: Optional[list[str]]=None,
            index_col: Optional[int]=None) -> pd.DataFrame:

    df = pd.read_csv(filepath,
                     keep_default_na=keep_default_na,
                     skiprows=skiprows,
                     names=new_names,
                     index_col=index_col,
                     sep='\t',
                     encoding='utf-8')

    df = df.astype(coltypes)

    return df


def write_df(df, filename, index: bool,
             na_rep: Optional[str]=None,
             header: Optional[bool]=True) -> None:
    df.to_csv(filename, sep='\t', index=index, na_rep=na_rep, header=header, encoding='utf-8')


def create_spark_dfs(train_df, test_df, spark):
    spark_train, spark_test = (spark.createDataFrame(train_df), spark.createDataFrame(test_df))
    return spark_train, spark_test


def combine_tt_dfs(spark_train, spark_test):
    """ Combine train and test Spark DFs to train the w2v model. """
    train_df = spark_train.withColumn('is_test', lit(False))
    test_df = spark_test.withColumn('is_test', lit(True))
    combined_df = train_df.unionByName(test_df)

    # Fix coltype
    combined_df = combined_df.withColumn('abstract', regexp_replace('abstract', r'^\[|\]$', ''))
    combined_df = combined_df.withColumn('abstract', expr("split(abstract, ', ')"))

    return combined_df


def word_vectorize(combined_df: DataFrame, input_col, vectorized_col) -> tuple[DataFrame, pd.DataFrame]:
    """
    Train a w2v model on the full train_set, embed the input text, and return the word-vectorized DF and the w2v
    vocab DF with their vectors.
    """
    word2vec = Word2Vec(vectorSize=10, minCount=3, inputCol=input_col, outputCol=vectorized_col, seed=0)
    w2v_model = word2vec.fit(combined_df)
    word_vectorized_df = w2v_model.transform(combined_df)
    word_vectorized_df = word_vectorized_df.drop(input_col)

    w2v_vectors_DF = w2v_model.getVectors()
    w2v_vectors_df = w2v_vectors_DF.toPandas()

    return word_vectorized_df, w2v_vectors_df


def string_index(df, input_col, indexed_col):
    """ String-index additional categorical features to be one-hot-encoded. """
    indexer = StringIndexer(inputCol=input_col, outputCol=indexed_col)
    indexing_model = indexer.fit(df)
    indexed_df = indexing_model.transform(df)
    indexed_df = indexed_df.drop(input_col)
    return indexed_df


def one_hot_encode(df, input_col, encoded_col):
    """ One-hot-encode additional categorical features. """
    onehot_encoder = OneHotEncoder(inputCol=input_col, outputCol=encoded_col)
    encoding_model = onehot_encoder.fit(df)
    encoded_df = encoding_model.transform(df)
    encoded_df = encoded_df.drop(input_col)
    return encoded_df


def split_tt_dfs(encoded_df):
    """ Re-split the full encoded DF into train and test sets. """
    train_df_vectorized = encoded_df.filter(~encoded_df['is_test'])
    test_df_vectorized = encoded_df.filter(encoded_df['is_test'])

    train_df_vectorized = train_df_vectorized.drop('is_test')
    test_df_vectorized = test_df_vectorized.drop('is_test')

    return train_df_vectorized, test_df_vectorized


def transform_features(train_df_encoded, test_df_encoded):
    """ Modify the encoded train and test DFs to be input to Roby. """
    train_df_transformed = train_df_encoded.withColumnRenamed('sdg', 'label')
    test_df_tranformed = test_df_encoded.withColumnRenamed('sdg', 'label')
    features = test_df_tranformed.drop('label').columns

    vecAssembler = VectorAssembler(inputCols=features, outputCol='features')

    return train_df_transformed, test_df_tranformed, vecAssembler


def roby_train(train_df_transformed, vecAssembler, model_path):
    """ Train and save Roby model. """
    lr = LogisticRegression(maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                            threshold=0.5, standardization=True, aggregationDepth=2)
    pipeline = Pipeline(stages=[vecAssembler, lr])
    roby_model = pipeline.fit(train_df_transformed)

    if os.path.exists(model_path):
        roby_model.write().overwrite().save(model_path)
    else:
        roby_model.save(model_path)

    return roby_model


def roby_predict(roby_model, test_df_transformed, test_df):
    """
    Make predictions on test data, create predictions df with original indices and columns:
    - abstract
    - label
    - prediction
    - probability (of predicted label)
    """
    predictions = roby_model.transform(test_df_transformed)

    # Drop unnecessary cols, add abstracts and indices
    if 'faculty_encoded' in predictions.columns:
        predictions = predictions.drop('faculty_encoded', 'year_encoded')
    predictions = predictions.drop('abstract_vectorized', 'features', 'rawPrediction').toPandas()
    predictions['abstract'] = test_df['abstract']
    predictions.index = test_df.index

    # Correct coltypes
    predictions = predictions.astype({'prediction': 'int'})
    predictions['probability'] = np.array(predictions['probability'])

    # Reorder cols
    cols = ['abstract', 'label', 'prediction', 'probability']
    predictions = predictions[cols]

    return predictions
