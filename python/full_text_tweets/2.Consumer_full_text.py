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
	context = StreamingContext(sc, 60)
	dStream = KafkaUtils.createDirectStream(context, ["thsFullTextTweets"], {"metadata.broker.list": "localhost:9092"})
	dStream.foreachRDD(p1)
	context.start()
	context.awaitTermination()

def insertRawTweets(rdd, spark, time):
	df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], json=str(x))))
	df.createOrReplaceTempView("raw_tweet")
	df = spark.sql("create database if not exists thsfulltext")
	df = spark.sql("use thsfulltext")
	df = spark.sql("select * from raw_tweet")
	df.write.mode("append").saveAsTable("raw_tweet")	
	print("Inserted raw_tweet")

def insertTweets(rdd, spark, time):
	rdd = rdd.filter(lambda x: ("zika" in str(x["extended_tweet"]["full_text"]).lower() or "flu" in str(x["extended_tweet"]["full_text"]).lower() or "ebola" in str(x["extended_tweet"]["full_text"]).lower() or "measles" in str(x["extended_tweet"]["full_text"]).lower() or "diarrhea" in str(x["extended_tweet"]["full_text"]).lower() ) == True )
	if(rdd.count()>0):
		df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], user_twitter_id=x["user"]["id_str"], full_text=str(x["extended_tweet"]["full_text"]), is_retweet=x["retweeted"], is_reply="in_reply_to_screen_name" in x, is_favorite=x["favorited"] == "true", is_decomposed=False)))
		df.createOrReplaceTempView("tweet")
		df = spark.sql("create database if not exists thsfulltext")
		df = spark.sql("use thsfulltext")
		df = spark.sql("select * from tweet")
		df.write.mode("append").saveAsTable("tweet")
		print("Inserted tweet")

def p1(time,rdd):
	rdd = rdd.map(lambda x: json.loads(x[1]))
	spark = getSparkSessionInstance(rdd.context.getConf())
	# Raw Tweet
	if rdd.count() > 0:
		tweets = rdd.filter(lambda y: (y["retweeted"] == False and "retweeted_status" not in y))
		if tweets.count() > 0:
			insertRawTweets(tweets, spark, time)
			insertTweets(tweets, spark, time)
		else:
			print("Just retweets to insert...")
	else:
		print("No raw tweets avaliable to insert in hive.")

if __name__ == "__main__":
	print("Starting to read tweets")
	print("Startup at", datetime.now())
	conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED","123")
	sc = SparkContext(appName="defaultConsumer", conf=conf)
	consumer()