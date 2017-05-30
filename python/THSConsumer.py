from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
from operator import add
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from pyspark.sql import Row, SparkSession
from datetime import datetime, timedelta
import uuid
try:
    import json
except ImportError:
    import simplejson as json
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars $SPARK_HOME/jars/spark-streaming-kafka-0-8-assembly_2.11.jar pyspark-shell, $SPARK_HOME/jars/hiveUDFs.jar'

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
        rdd = rdd.filter(lambda x: x["user"]["lang"]=="en")
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], json=str(x))))
            df.createOrReplaceTempView("raw_tweet")
            df = spark.sql("create database if not exists ths2")
            df = spark.sql("use ths2")
            df = spark.sql("select * from raw_tweet")
            df.write.mode("append").saveAsTable("raw_tweet")
            print("Inserted raw_tweet")
    else:
        print("No raw tweets avaliable to insert in hive")

def insertTweets(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        rdd = rdd.filter(lambda x: x["user"]["lang"]=="en")
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], user_twiter_id=x["user"]["id_str"], text=x["text"], is_retweet="retweeted_status" in x, is_reply="in_reply_to_screen_name" in x, is_favorite=x["favorited"] == "true", is_decomposed=False)))
            df.createOrReplaceTempView("tweet")
            df = spark.sql("create database if not exists ths2")
            df = spark.sql("use ths2")
            df = spark.sql("select * from tweet")
            df.write.mode("append").saveAsTable("tweet")
            print("Inserted tweet")
    else:
        print("No tweets avaliable to insert in hive")

def insertUsers(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        rdd = rdd.filter(lambda x: x["user"]["lang"]=="en")
        if rdd.count() > 0:
            df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["user"]["id_str"], name=x["user"]["name"], screenname=x["user"]["screen_name"], description=x["user"]["description"], language=x["user"]["lang"], profile_picture=x["user"]["profile_image_url"], url=x["user"]["url"])))
            df.createOrReplaceTempView("twitter_user")
            df = spark.sql("create database if not exists ths2")
            df = spark.sql("use ths2")
            df = spark.sql("select * from twitter_user")
            df.write.mode("append").saveAsTable("twitter_user")
            print("Inserted user")
    else:
        print("No users avaliable to insert in hive")

def insertHashtags(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        rdd = rdd.filter(lambda y: y["user"]["lang"]=="en").filter(lambda x: len(x["entities"]["hashtags"]) > 0)
        rdd = rdd.map(lambda x: (x['id_str'], x['entities']['hashtags']))
        rdd = rdd.flatMapValues(lambda x: x)
        rdd = rdd.map(lambda x: (x[0], x[1]['text']))
        if rdd.count() > 0
            # This will be slow, figure out another way if necessary
            # Or move insertion to another process to run in background
            has_hashtag_rows = []
            spark.sql("create database if not exists ths2")
            spark.sql("use ths2")
            # Alphabetic order important in table definition
            spark.sql("create table if not exists hashtag(hashtag_text STRING, id STRING)")
            spark.sql("create table if not exists has_hashtag(hashtag_id STRING, tweet_id STRING)")
            for record in rdd.collect():
                query = "select * from hashtag where hashtag_text='" + record[1] + "'"
                search = spark.sql(query)
                if search.count() == 0:
                    uuidNew = str(uuid.uuid4())
                    uuidDf = spark.createDataFrame(sc.parallelize([Row(hashtag_text=record[1], id=uuidNew)]))
                    uuidDf.write.mode("append").insertInto("hashtag")
                else:
                    uuidNew = search.first()['id']
                has_hashtag_rows.append(Row(hashtag_id=uuidNew, tweet_id=record[0]))
            df = spark.createDataFrame(sc.parallelize(has_hashtag_rows))
            df.write.mode("append").insertInto("has_hashtag")
        else:
            print("No hashtags avaliable to insert in hive")
    else:
        print("No hashtags avaliable to insert in hive")

def p1(time,rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    raw_tweets = rdd.collect() #Return a list with tweets
    spark = getSparkSessionInstance(rdd.context.getConf())
    # Raw Tweet
    insertRawTweets(raw_tweets, spark, time)
    insertTweets(raw_tweets, spark, time)
    insertUsers(raw_tweets, spark, time)
    insertHashtags(raw_tweets, spark, time)


if __name__ == "__main__":
    print("Starting to read tweets")
    print("Startup at", datetime.now())
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED","123")
    sc = SparkContext(appName="ths2Consumer", conf=conf)
    consumer()
