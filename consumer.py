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

def read_credentials():
    file_name = "/home/garzoncristianc/credentials.json"
    try:
        with open(file_name) as data_file:
            return json.load(data_file)
    except:
        print ("Cannot load credentials.json")
        return None

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    return globals()['sparkSessionSingletonInstance']

def consumer():
    #context = StreamingContext.getOrCreate(checkpointDirectory, functionToCreateContext)
    context = StreamingContext(sc, 10)
    dStream = KafkaUtils.createDirectStream(context, ["pointab"], {"metadata.broker.list": "localhost:9092"})
    
    dStream.foreachRDD(p1)
    context.start()
    context.awaitTermination()

def p1(time,rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    records = rdd.collect() #Return a list with tweets
    records = [element["entities"]["hashtags"] for element in records if "entities" in element] #select only hashtags part
    records = [x for x in records if x] #remove empty hashtags
    records = [element[0]["text"] for element in records]
    if records:
        rdd = sc.parallelize(records)
        rdd = rdd.filter(lambda x: len(x) > 3)
        spark = getSparkSessionInstance(rdd.context.getConf())
        # Convert RDD[String] to RDD[Row] to DataFrame
        hashtagsDataFrame = spark.createDataFrame(rdd.map(lambda x: Row(hashtag=x, timestamp=time)))
        hashtagsDataFrame.createOrReplaceTempView("hashtags")
        hashtagsDataFrame = spark.sql("use bigdata")
        hashtagsDataFrame = spark.sql("select hashtag, timestamp, count(*) as total from hashtags group by hashtag, timestamp order by total desc limit 50")
        hashtagsDataFrame.write.mode("append").saveAsTable("hashtag")
    else:
        print("Lista Vacia")

if __name__ == "__main__":
    print("Starting to read tweets")
    credentials = read_credentials() 
    oauth = OAuth(credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'], credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
    twitter_stream = TwitterStream(auth=oauth)
    sc = SparkContext(appName="Project2")
    checkpointDirectory = "/checkpoint"
    consumer()
    