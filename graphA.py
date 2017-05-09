from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession
import pandas as pan
#import matplotlib.pyplot as plot
try:
    import json
except ImportError:
    import simplejson as json
import os 
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars $SPARK_HOME/jars/spark-streaming-kafka-0-8-assembly_2.11.jar pyspark-shell'

def graph(datetimetest):
    spark = SparkSession.builder.config(conf=sc.getConf()).enableHiveSupport().getOrCreate()
    hashtagQuery = spark.sql("use bigdata")
    #hashtagQuery = spark.sql("SELECT hashtag, SUM(total) as count FROM hashtag WHERE timestamp BETWEEN timestamp -1 AND  GROUP BY hashtag ORDER BY 2 DESC LIMIT 10")
    query = "SELECT * FROM hashtag WHERE timestamp BETWEEN (from_utc_timestamp({}, 'PDT') - interval 1 hour) AND from_utc_timestamp({}, 'PDT')".format(datetimetest, datetimetest)
    #query = "SELECT * FROM hashtag WHERE timestamp BETWEEN (cast('2017-05-03 22:00:00' AS timestamp) - interval 1 hour) AND cast('2017-05-03 22:00:00' AS timestamp)"
    hashtagQuery = spark.sql(query)
    hashtagQuery.show()

if __name__ == "__main__":
    print("Starting to graph point A")
    sc = SparkContext(appName="GraphA")
    checkpointDirectory = "/checkpoint"
    graph('2017-05-03 22:00:00')