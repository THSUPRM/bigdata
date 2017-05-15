from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
from operator import add
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from pyspark.sql import Row, SparkSession
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
    dStream.foreachRDD(p2)
    #dStream.foreachRDD(lambda rdd: rdd.foreach(p2))
    context.start()
    context.awaitTermination()

def p2(time, rdd):
    lines = rdd.map(lambda x: json.loads(x[1]))
    rec = lines.collect()
    spark = getSparkSessionInstance(rdd.context.getConf())
    tweets = [element for element in rec]
    for tweet in tweets:
        id_tweet = tweet.get("id")
        rddTweet = sc.parallelize(tweet)
        print("Elementooooooooooooooooooooooooooooo: " + str(tweet))
        print("Element IDDDDDDDDDDDDDDDDDDDDDDDDDDD: " + str(id_tweet))
        #Convert RDD[String] to RDD[Row] to DataFrame
        DF = spark.createDataFrame(rddTweet.map(lambda x: Row(twitter_id=id_tweet, json=x)))
        DF.createOrReplaceTempView("raw_tweet")
        DF = spark.sql("use ths")
        DF = spark.sql("select twitter_id, json from raw_tweet")
        DF.write.mode("append").saveAsTable("raw_tweet")
        print("Inserted raw_tweet FINISH")
        
        #keyword = tweet.get("text")
        #if keyword:
            #print("Elementooooooooooooooooooooooooooooo: " + str(tweet))
            #print("Keywordddddddddddddddddddddddddddddd: " + str(keyword))
            #hashtags = tweet.get("entities").get('hashtags')
            #print("Hashtagsssssssssssssssssssssssssssss: " + str(hashtags))

def insertText(records, spark, time):
    keywords = [element["text"] for element in records if "text" in element]
    if keywords:
        rddRecords = sc.parallelize(records)
        rddKeywords = sc.parallelize(keywords)
        rddKeywords = rddKeywords.map(lambda x: x.lower()).filter(lambda x: "trump" in x)
        if rddKeywords.count() > 0:
            print(rddRecords)
            print(rddKeywords)
            # Convert RDD[String] to RDD[Row] to DataFrame
            #keywordsDataFrame = spark.createDataFrame(rddKeywords.map(lambda x: Row(tweet=x, hora=time)))
            #keywordsDataFrame.createOrReplaceTempView("fastcapture")
            #keywordsDataFrame = spark.sql("use profe")
            #keywordsDataFrame = spark.sql("select tweet, hora from fastcapture")
            #keywordsDataFrame.write.mode("append").saveAsTable("fastcapture")
            #print("Inserted fastcapture FINISH")
    else:
        print("Is not keywords avaliables to insert in hive")

if __name__ == "__main__":
    print("Starting to read tweets")
    sc = SparkContext(appName="ConsumerTHS")
    consumer()