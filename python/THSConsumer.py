from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
from operator import add
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from pyspark.sql import Row, SparkSession
from datetime import datetime, timedelta
from stop_words import get_stop_words
try:
    import json
except ImportError:
    import simplejson as json
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars $SPARK_HOME/jars/spark-streaming-kafka-0-8-assembly_2.11.jar pyspark-shell'

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    return globals()['sparkSessionSingletonInstance']

def consumer():
    context = StreamingContext(sc, 20)
    dStream = KafkaUtils.createDirectStream(context, ["thstweets"], {"metadata.broker.list": "localhost:9092"})
    dStream.foreachRDD(p1)
    context.start()
    context.awaitTermination()

def insertRawTweets(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], raw_json=str(x))))
            df.createOrReplaceTempView("raw_tweet")
            df = spark.sql("create database if not exists ths")
            df = spark.sql("use ths")
            df = spark.sql("select * from raw_tweet")
            df.write.mode("append").saveAsTable("raw_tweet")
            print("Inserted raw_tweet")
    else:
        print("No raw tweets avaliable to insert in hive")

def insertTweets(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], user_twiter_id=x["user"]["id_str"], text=x["text"], is_retweet="retweeted_status" in x, is_reply="in_reply_to_screen_name" in x, is_favorite=x["favorited"] == "true", is_decomposed=False)))
            df.createOrReplaceTempView("tweet")
            df = spark.sql("create database if not exists ths")
            df = spark.sql("use ths")
            df = spark.sql("select * from tweet")
            df.write.mode("append").saveAsTable("tweet")
            print("Inserted tweet")
    else:
        print("No tweets avaliable to insert in hive")

def insertUsers(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["user"]["id_str"], name=x["user"]["name"], screenname=x["user"]["screen_name"], description=x["user"]["description"], language=x["user"]["lang"], profile_picture=x["user"]["profile_image_url"], url=x["user"]["url"])))
            df.createOrReplaceTempView("twitter_user")
            df = spark.sql("create database if not exists ths")
            df = spark.sql("use ths")
            df = spark.sql("select * from twitter_user")
            df.write.mode("append").saveAsTable("twitter_user")
            print("Inserted user")
    else:
        print("No users avaliable to insert in hive")

def p1(time,rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    raw_tweets = rdd.collect() #Return a list with tweets
    spark = getSparkSessionInstance(rdd.context.getConf())

    # Raw Tweet
    insertRawTweets(raw_tweets, spark, time)
    insertTweets(raw_tweets, spark, time)
    insertUsers(raw_tweets, spark, time)

if __name__ == "__main__":
    print("Starting to read tweets")
    print("Startup at", datetime.now())
    sc = SparkContext(appName="THSConsumer")
    consumer()
