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
        if rdd.count() > 0:
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

def insertKeywords(raw_tweets, spark, time):
    if raw_tweets:
        rdd = sc.parallelize(raw_tweets)
        rdd = rdd.filter(lambda x: x["user"]["lang"]=="en")
        rdd = rdd.map(lambda x: (x['id_str'], x["text"].lower().split()))
        rdd = rdd.flatMapValues(lambda x : x)
        rdd = rdd.filter(lambda x: x[1] != "a" and x[1] != "i'm" and x[1] != "a's" and x[1] != "ain't" and x[1] != "aren't" and x[1] != "c'mon" and x[1] != "c's" and x[1] != "can't" and x[1] != "couldn't" and x[1] != "didn't" and x[1] != "doesn't" and x[1] != "don't" and x[1] != "hadn't" and x[1] != "hasn't" and x[1] != "haven't" and x[1] != "he's" and x[1] != "here's" and x[1] != "i'd" and x[1] != "i'll" and x[1] != "i'm" and x[1] != "i've" and x[1] != "isn't" and x[1] != "it'd" and x[1] != "it'll" and x[1] != "it's" and x[1] != "let's" and x[1] != "shouldn't" and x[1] != "t's" and x[1] != "that's" and x[1] != "there's" and x[1] != "they'd" and x[1] != "they'll" and x[1] != "they're" and x[1] != "they've" and x[1] != "wasn't" and x[1] != "we'd" and x[1] != "we'll" and x[1] != "we're" and x[1] != "we've" and x[1] != "weren't" and x[1] != "what's" and x[1] != "where's" and x[1] != "who's" and x[1] != "won't" and x[1] != "wouldn't" and x[1] != "you'd" and x[1] != "you'll" and x[1] != "you're" and x[1] != "you've" and x[1] != "@" and x[1] != "rt" and x[1] != "'" and x[1] != "you're" and x[1] != "an" and x[1] != "and" and x[1] != "are" and x[1] != "as" and x[1] != "at" and x[1] != "be" and x[1] != "by" and x[1] != "for" and x[1] != "from" and x[1] != "has" and x[1] != "he" and x[1] != "in" and x[1] != "is" and x[1] != "its" and x[1] != "of" and x[1] != "on" and x[1] != "that" and x[1] != "the" and x[1] != "to" and x[1] != "was" and x[1] != "were" and x[1] != "will" and x[1] != "with")
        rdd = rdd.map(lambda x: (x[0], x[1]))
        if rdd.count() > 0:
            keywords_rows = []
            spark.sql("create database if not exists ths2")
            spark.sql("use ths2")
            spark.sql("create table if not exists keyword(id STRING, keyword_text STRING)")
            spark.sql("create table if not exists has_keyword(keyword_id STRING, tweet_id STRING)")
            i = 0
            for record in rdd.collect():
                if i < 10:
                    query = "select * from keyword where keyword_text='" + record[1] + "'"
                    search = spark.sql(query)
                    if search.count() == 0:
                        uuidNew = str(uuid.uuid4())
                        uuidDf = spark.createDataFrame(sc.parallelize([Row(id=uuidNew, keyword_text=record[1])]))
                        uuidDf.write.mode("append").insertInto("keyword")
                        i += 1
                    else:
                        uuidNew = search.first()['id']
                    keywords_rows.append(Row(keyword_id=uuidNew, tweet_id=record[0]))
                    
            print("Aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii Paso por aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
            df = spark.createDataFrame(sc.parallelize(keywords_rows))
            df.write.mode("append").insertInto("has_keyword")
        else:
            print("No keywords avaliable to insert in hive")
    else:
        print("No keywords avaliable to insert in hive")

def p1(time,rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    raw_tweets = rdd.collect() #Return a list with tweets
    spark = getSparkSessionInstance(rdd.context.getConf())
    # Raw Tweet
    #insertRawTweets(raw_tweets, spark, time)
    #insertTweets(raw_tweets, spark, time)
    #insertUsers(raw_tweets, spark, time)
    #insertHashtags(raw_tweets, spark, time)
    insertKeywords(raw_tweets, spark, time)


if __name__ == "__main__":
    print("Starting to read tweets")
    print("Startup at", datetime.now())
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED","123")
    sc = SparkContext(appName="ths2Consumer", conf=conf)
    consumer()
