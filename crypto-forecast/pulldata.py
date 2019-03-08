import configparser
import datetime
import os

import cbpro
import pandas as pd
from pyspark.sql import SparkSession
from tqdm import tqdm


def main():
    public_client = cbpro.PublicClient()
    config = configparser.ConfigParser()

    config.read(os.path.expanduser("~/.aws/credentials"))

    access_key = config.get('personal', "aws_access_key_id")
    secret_key = config.get('personal', "aws_secret_access_key")
    spark = SparkSession.builder.appName("myApp").config(
        "spark.jars.packages",
        "org.apache.hadoop:hadoop-aws:2.7.3,com.amazonaws:aws-java-sdk:1.7.4",
    ).config(
        "fs.s3a.access.key",
        access_key,
    ).config(
        "fs.s3a.secret.key",
        secret_key,
    ).getOrCreate()
    sdf = spark.read.parquet('s3a://dshurick-crypto/BTC-USD/')

    start = datetime.datetime(
        year=2019,
        month=1,
        day=1,
    )

    end = datetime.datetime.now()

    dti = pd.date_range(
        start=start,
        end=end,
        freq='1500 min',
    )

    all_rates = [
        public_client.get_product_historic_rates(
            product_id='BTC-USD',
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            granularity=300,
        ) for start_dt, end_dt in tqdm(list(zip(dti[:-1], dti[1:])))
    ]

    flat_list = [item for sublist in all_rates for item in sublist]

    df = pd.DataFrame(
        flat_list,
        columns=[
            'time',
            'low',
            'high',
            'open',
            'close',
            'volume',
        ])

    df['time'] = pd.to_datetime(df['time'], unit='s')

    df.reset_index(inplace=True)
    df.drop_duplicates(
        subset=[
            'product_id',
            'time',
        ], inplace=True)

    df['product_id'] = 'BTC-USD'

    df_old = sdf.toPandas()

    df_new = pd.concat([
        df,
        df_old,
    ])

    df_new.reset_index(inplace=True)

    df.drop_duplicates(
        subset=[
            'product_id',
            'time',
        ], inplace=True)

    sdf_new = spark.createDataFrame(df)

    sdf_new.write.parquet(
        's3a://dshurick-crypto/BTC-USD/',
        mode='overwrite',
    )


if __name__ == '__main__':
    main()
