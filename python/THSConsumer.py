from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
from operator import add
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from pyspark.sql import Row, SparkSession
from datetime import datetime, timedelta
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
        rdd = rdd.flatMap(lambda text: text["text"].split(" ")).filter(lambda word: word.lower().startswith("#"))
        rdd = rdd.collect()
        #htgs = ([element["id_str"] for element in rdd], [element["entities"]["hashtags"] for element in rdd if "entities" in element])
        #htgs = (1, [element[0]["text"] for element in htgs])
        '''
        ['867446764349116416', '867446764378484737', '867446768572788738', '867446768543428610', '867446768543428609', '867446768568594432', '867446768572739585', '867446768547450880', '867446768560164864', '867446768539234310', 
        '867446772729344004', '867446772746121218', '867446772729344006', '867446776940208128', '867446776936230915', '867446776957202434', '867446781155651584', '867446785333055488', '867446785345806337', '867446789514723328', 
        '867446789514948608', '867446793709027328', '867446793705058309', '867446793717641216', '867446797932707842', '867446802089472003', '867446802097852416', '867446802127220738', '867446806321307648', '867446806296350720', 
        '867446806292164609', '867446806296309761', '867446806317330435', '867446806304751616', '867446810478092289', '867446810478080003', '867446810511417345', '867446810499051520', '867446814693355521', '867446818866704384', 
        '867446818887659527', '867446823060951041', '867446823061008385', '867446823077769219', '867446827284656132', '867446827284656133', '867446831470571524', '867446831449600003', '867446831449604097', '867446835652116481', 
        '867446835668893696', '867446835677462528', '867446835669073920', '867446835681652739', '867446839838203905', '867446839850786816', '867446839854997504', '867446839838208005', '867446844032294912', '867446844057677833', 
        '867446844040900608', '867446848260235264', '867446848251981825'], [[{'text': 'ETFs', 'indices': [115, 120]}], [{'text': 'الامريكية', 'indices': [10, 20]}, {'text': 'الاسرائيلية', 'indices': [23, 35]}, {'text': 'دول_الخليج', 
        'indices': [60, 71]}, {'text': 'قطر', 'indices': [76, 80]}, {'text': 'الإرهاب', 'indices': [85, 93]}, {'text': 'الاخوان', 'indices': [96, 104]}], [{'text': 'hayatımınteklifi', 'indices': [0, 17]}], [{'text': 'hayatımınteklifi', 
        'indices': [0, 17]}], [{'text': 'HOT', 'indices': [20, 24]}, {'text': 'PHARMA', 'indices': [25, 32]}, {'text': 'MIDDAY', 'indices': [33, 40]}, {'text': 'STOCKALERT', 'indices': [41, 52]}], [{'text': 'hayatımınteklifi', 
        'indices': [0, 17]}], [{'text': 'hayatımınteklifi', 'indices': [0, 17]}], [{'text': 'DafBama2017_EXO', 'indices': [16, 32]}],
        '''
        print(rdd)
        #rdd = rdd.map(lambda x: (x["id_str"], list([hashtag['text'].lower() for hashtag in x['entities']['hashtags']])))
        #rdd = rdd.flatMap(lambda tweet: [hashtag['text'].lower() for hashtag in tweet['entities']['hashtags']])
        #htgs = [element["id"], element["entities"]["hashtags"] for element in rdd if "entities" in element]
        #htgs = [x for x in htgs if x] #remove empty hashtags
        #htgs = [element[0], element[1]["text"] for element in htgs]
        #print(rdd.collect())
        #if rdd.count() > 0:
            #df = spark.createDataFrame(rdd.map(lambda x: Row(hashtag=x["entities"]["hashtags"])))
            #df = spark.createDataFrame(rdd.flatMap(lambda x: Row(hashtag=(list([hashtag['text'].lower() for hashtag in x['entities']['hashtags']])))))
            #df.createOrReplaceTempView("hashtag")
            #df = spark.sql("create database if not exists ths2")
            #df = spark.sql("use ths2")
            #df = spark.sql("create temporary function row_sequence as 'org.apache.hadoop.hive.contrib.udf.UDFRowSequence'")
            #df = spark.sql("select * from hashtag")
            #df.write.mode("append").saveAsTable("hashtag")
    else:
        print("No emojis avaliable to insert in hive")

def p1(time,rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    raw_tweets = rdd.collect() #Return a list with tweets
    spark = getSparkSessionInstance(rdd.context.getConf())
    # Raw Tweet
    #insertRawTweets(raw_tweets, spark, time)
    #insertTweets(raw_tweets, spark, time)
    #insertUsers(raw_tweets, spark, time)
    insertHashtags(raw_tweets, spark, time)


if __name__ == "__main__":
    print("Starting to read tweets")
    print("Startup at", datetime.now())
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED","123")
    sc = SparkContext(appName="ths2Consumer", conf=conf)
    consumer()