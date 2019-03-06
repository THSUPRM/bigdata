from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
from operator import add
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import *
from datetime import datetime, timedelta
import uuid
from geopy.geocoders import Nominatim
import ast
import time

try:
    import json
except ImportError:
    import simplejson as json
import os

os.environ[
    'PYSPARK_SUBMIT_ARGS'] = '--jars $SPARK_HOME/jars/spark-streaming-kafka-0-8-assembly_2.11.jar pyspark-shell, $SPARK_HOME/jars/hiveUDFs.jar'


def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession.builder.config(
            conf=sparkConf).enableHiveSupport().getOrCreate()
    return globals()['sparkSessionSingletonInstance']


def consumer():
    context = StreamingContext(sc, 540)
    dStream = KafkaUtils.createDirectStream(context, ["thsFullTextTweets"], {"metadata.broker.list": "localhost:9092"})
    dStream.foreachRDD(p1)
    context.start()
    context.awaitTermination()


def insertRawTweets(rdd, spark, time):
    df = spark.createDataFrame(
        rdd.map(lambda x: Row(twitter_id=x["id_str"], json=str(x), inserted_tweet=str(datetime.now()))))
    df.createOrReplaceTempView("raw_tweet")
    df = spark.sql("create database if not exists thsfulltext")
    df = spark.sql("use thsfulltext")
    df = spark.sql("select * from raw_tweet")
    df.write.mode("append").saveAsTable("raw_tweet")
    print("Inserted raw_tweet")


def insertTweets(rdd, spark, time):
    rdd = rdd.filter(lambda x: ("zika" in str(x["extended_tweet"]["full_text"]).lower() or "flu" in str(
        x["extended_tweet"]["full_text"]).lower() or "ebola" in str(
        x["extended_tweet"]["full_text"]).lower() or "measles" in str(
        x["extended_tweet"]["full_text"]).lower() or "diarrhea" in str(
        x["extended_tweet"]["full_text"]).lower()) == True)
    if (rdd.count() > 0):
        df = spark.createDataFrame(rdd.map(lambda x: Row(twitter_id=x["id_str"], user_twitter_id=x["user"]["id_str"],
                                                         full_text=str(x["extended_tweet"]["full_text"]),
                                                         is_retweet=x["retweeted"],
                                                         is_reply="in_reply_to_screen_name" in x,
                                                         is_favorite=x["favorited"] == "true", is_decomposed=False,
                                                         inserted_tweet=str(datetime.now()), used_tweet="")))
        df.createOrReplaceTempView("tweet")
        df = spark.sql("create database if not exists thsfulltext")
        df = spark.sql("use thsfulltext")
        df = spark.sql("select * from tweet")
        df.write.mode("append").saveAsTable("tweet")
        print("Inserted tweet")


def insertTweet(data, spark):
    data = data.map(lambda x: (
        x['id_str'], x['user']['screen_name'], x['created_at'], x['extended_tweet']['full_text'], x['favorited'],
        x['in_reply_to_screen_name'], x['retweeted']))
    data = data.map(lambda x: (x[0], x[1], x[2], x[3].replace('#', ' '), x[4], x[5], x[6]))
    data = data.map(lambda x: (x[0], x[1], x[2], x[3].replace(',', ' '), x[4], x[5], x[6]))
    data = data.map(lambda x: (x[0], x[1], x[2], x[3].replace('.', ' '), x[4], x[5], x[6]))

    illness1 = data.map(lambda sn: (
        sn[0], sn[1], sn[2], sn[3], ' '.join(filter(lambda x: 'zika' == x.lower(), sn[3].split())).lower(), sn[4],
        sn[5],
        sn[6]))
    illness1 = illness1.filter(lambda x: x[4] != '')
    illness1 = illness1.map(lambda x: (x[0], x[1], x[2], x[3], x[4].split(' ', 1)[0], x[5], x[6], x[7]))

    illness2 = data.map(lambda sn: (
        sn[0], sn[1], sn[2], sn[3], ' '.join(filter(lambda x: 'flu' == x.lower(), sn[3].split())).lower(), sn[4], sn[5],
        sn[6]))
    illness2 = illness2.filter(lambda x: x[4] != '')
    illness2 = illness2.map(lambda x: (x[0], x[1], x[2], x[3], x[4].split(' ', 1)[0], x[5], x[6], x[7]))

    illness3 = data.map(lambda sn: (
        sn[0], sn[1], sn[2], sn[3], ' '.join(filter(lambda x: 'ebola' == x.lower(), sn[3].split())).lower(), sn[4],
        sn[5],
        sn[6]))
    illness3 = illness3.filter(lambda x: x[4] != '')
    illness3 = illness3.map(lambda x: (x[0], x[1], x[2], x[3], x[4].split(' ', 1)[0], x[5], x[6], x[7]))

    illness4 = data.map(lambda sn: (
        sn[0], sn[1], sn[2], sn[3], ' '.join(filter(lambda x: 'measles' == x.lower(), sn[3].split())).lower(), sn[4],
        sn[5],
        sn[6]))
    illness4 = illness4.filter(lambda x: x[4] != '')
    illness4 = illness4.map(lambda x: (x[0], x[1], x[2], x[3], x[4].split(' ', 1)[0], x[5], x[6], x[7]))

    illness5 = data.map(lambda sn: (
        sn[0], sn[1], sn[2], sn[3], ' '.join(filter(lambda x: 'diarrhea' == x.lower(), sn[3].split())).lower(), sn[4],
        sn[5], sn[6]))
    illness5 = illness5.filter(lambda x: x[4] != '')
    illness5 = illness5.map(lambda x: (x[0], x[1], x[2], x[3], x[4].split(' ', 1)[0], x[5], x[6], x[7]))

    data = illness1.union(illness2).union(illness3).union(illness4).union(illness5)
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("user", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("tweet", StringType(), True),
        StructField("disease", StringType(), True),
        StructField("favorited", BooleanType(), True),
        StructField("reply", StringType(), True),
        StructField("retweeted", BooleanType(), True)
    ])
    df = spark.createDataFrame(data.map(
        lambda x: Row(id=x[0], user=x[1], timestamp=x[2], tweet=x[3], disease=x[4], favorited=x[5], reply=x[6],
                      retweeted=x[7])), schema)
    df.createOrReplaceTempView("tweet")
    df = spark.sql("create database if not exists thsfulltext")
    df = spark.sql("use thsfulltext")
    df = spark.sql("SELECT id, user, timestamp, tweet, disease, favorited, reply, retweeted FROM tweet")
    df.write.mode("append").saveAsTable("tb_tweet")
    print("tb_tweet inserted")


def insertUser(data, spark):
    data = data.map(lambda x: (x['id_str'], x['user'], x['created_at'])).map(lambda x: (
        x[0], x[1]['id_str'], x[1]['name'], x[1]['screen_name'], x[1]['description'], x[1]['lang'], x[1]['location'],
        x[2]))
    datalocyes = data.filter(lambda x: x[6] != None)
    datalocno = data.filter(lambda x: x[6] == None)

    info = datalocyes.collect()
    infofinal = []
    count = 0
    for a in info:
        for attempt in range(3):
            try:
                lst = list(a)
                lst[6] = geolocator.geocode(lst[6])
                time.sleep(1.2)
                if lst[6] != None:
                    lst[6] = geolocator.reverse([lst[6].latitude, lst[6].longitude], language='en').raw['address'][
                        'country_code']
                    time.sleep(1)
                a = tuple(lst)
                infofinal.append(a)
                print(count)
                count = count + 1
            except:
                time.sleep(1)
            else:
                break
        else:
            lst[6] = None
            print("Error getting geo data. Retried 3 times.")
            a = tuple(lst)
            infofinal.append(a)

    datalocyes = sc.parallelize(infofinal)
    data = datalocyes.union(datalocno)

    schema = StructType([
        StructField("id_tweet", StringType(), True),
        StructField("id_user", StringType(), True),
        StructField("name", StringType(), True),
        StructField("screen_name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("lang", StringType(), True),
        StructField("location", StringType(), True),
        StructField("timestamp", StringType(), True)
    ])
    df = spark.createDataFrame(data.map(
        lambda x: Row(id_tweet=x[0], id_user=x[1], name=x[2], screen_name=x[3], description=x[4], lang=x[5],
                      location=x[6], timestamp=x[7])), schema)

    df.createOrReplaceTempView("user")
    df = spark.sql("create database if not exists thsfulltext")
    df = spark.sql("use thsfulltext")
    df = spark.sql("SELECT id_tweet, id_user, name, screen_name, description, lang, location, timestamp FROM user")
    df.write.mode("append").saveAsTable("tb_user")
    print("tb_user inserted")


def insertHashtags(data, spark):
    data = data.filter(lambda x: len(x["extended_tweet"]["entities"]["hashtags"]) > 0)
    if (data.count() > 0):
        dates = data.map(lambda x: (x['id_str'], x['created_at']))
        hashtags = data.map(lambda x: (x['id_str'], x["extended_tweet"]['entities']['hashtags']))
        hashtags = hashtags.flatMapValues(lambda x: x)
        hashtags = hashtags.map(lambda x: (x[0], x[1]['text']))
        data = dates.join(hashtags).map(lambda x: (x[0], x[1][1], x[1][0]))

        df = spark.createDataFrame(data.map(lambda x: Row(id_tweet=x[0], hashtag=x[1], timestamp=x[2])))
        df.createOrReplaceTempView("hashtag")
        df = spark.sql("create database if not exists thsfulltext")
        df = spark.sql("use thsfulltext")
        df = spark.sql("SELECT id_tweet, hashtag, timestamp FROM hashtag")
        df.show()
        df.write.mode("append").saveAsTable("tb_hashtag")
        print("tb_hashtag inserted")
    else:
        print("No hashtags to be inserted")


def insertKeyword(data, spark):
    dates = data.map(lambda x: (x['id_str'], x['created_at']))
    keyword = data.map(lambda x: (x['id_str'], x["extended_tweet"]['full_text']))
    keyword = keyword.map(
        lambda sn: [sn[0], ' '.join(filter(lambda x: x.startswith(('@', 'http', 'rt', '&')) == False, sn[1].split()))])
    keyword = keyword.map(lambda sn: [sn[0], ' '.join(filter(lambda x: x.endswith(('"')) == False, sn[1].split()))])
    keyword = keyword.map(lambda sn: [sn[0], ' '.join(filter(lambda x: len(x) > 2, sn[1].split()))])
    keyword = keyword.map(lambda x: (x[0], x[1].replace(',', ' '))).map(lambda x: (x[0], x[1].replace('?', ' '))).map(
        lambda x: (x[0], x[1].replace('/', ' ')))
    keyword = keyword.map(lambda x: (x[0], x[1].replace(')', ' '))).map(lambda x: (x[0], x[1].replace('(', ' '))).map(
        lambda x: (x[0], x[1].replace('!', ' ')))
    keyword = keyword.map(lambda x: (x[0], x[1].replace(']', ' '))).map(lambda x: (x[0], x[1].replace('[', ' '))).map(
        lambda x: (x[0], x[1].replace('{', ' ')))
    keyword = keyword.map(lambda x: (x[0], x[1].replace('}', ' '))).map(lambda x: (x[0], x[1].replace('#', ' '))).map(
        lambda x: (x[0], x[1].replace('"', ' ')))
    keyword = keyword.map(lambda x: (x[0], x[1].replace('—', ' '))).map(lambda x: (x[0], x[1].replace('|', ' '))).map(
        lambda x: (x[0], x[1].replace('.', ' ')))
    keyword = keyword.map(lambda x: (x[0], x[1].replace(':', ' '))).map(lambda x: (x[0], x[1].replace('–', ' '))).map(
        lambda x: (x[0], x[1].upper())).map(lambda x: (x[0], " ".join(x[1].split())))
    keyword = keyword.map(lambda x: (x[0], (x[1].split()))).flatMapValues(lambda x: x)
    keyword = keyword.filter(lambda x: (
        "A'S" != x[1] and x[1] != "ABLE" and x[1] != "ABOUT" and x[1] != "ABOVE" and x[1] != "ACCORDING" and x[
            1] != "ACCORDINGLY" and x[1] != "ACROSS" and x[1] != "ACTUALLY" and x[1] != "AFTER" and x[
            1] != "AFTERWARDS" and
        x[1] != "AGAIN" and x[1] != "AGAINST" and x[1] != "AIN'T" and x[1] != "ALL" and x[1] != "ALLOW" and x[
            1] != "ALLOWS" and x[1] != "ALMOST" and x[1] != "ALONE" and x[1] != "ALONG" and x[1] != "ALREADY" and x[
            1] != "ALSO" and x[1] != "ALTHOUGH" and x[1] != "ALWAYS" and x[1] != "AM" and x[1] != "AMONG" and x[
            1] != "AMONGST" and x[1] != "AN" and x[1] != "AND" and x[1] != "ANOTHER" and x[1] != "ANY" and x[
            1] != "ANYBODY" and x[1] != "ANYHOW" and x[1] != "ANYONE" and x[1] != "ANYTHING" and x[1] != "ANYWAY" and x[
            1] != "ANYWAYS" and x[1] != "ANYWHERE" and x[1] != "APART" and x[1] != "APPEAR" and x[1] != "APPRECIATE" and
        x[
            1] != "APPROPRIATE" and x[1] != "ARE" and x[1] != "AREN'T" and x[1] != "AROUND" and x[1] != "AS" and x[
            1] != "ASIDE" and x[1] != "ASK" and x[1] != "ASKING" and x[1] != "ASSOCIATED" and x[1] != "AT" and x[
            1] != "AVAILABLE" and x[1] != "AWAY" and x[1] != "AWFULLY" and x[1] != "B" and x[1] != "BE" and x[
            1] != "BECAME" and x[1] != "BECAUSE" and x[1] != "BECOME" and x[1] != "BECOMES" and x[1] != "BECOMING" and
        x[
            1] != "BEEN" and x[1] != "BEFORE" and x[1] != "BEFOREHAND" and x[1] != "BEHIND" and x[1] != "BEING" and x[
            1] != "BELIEVE" and x[1] != "BELOW" and x[1] != "BESIDE" and x[1] != "BESIDES" and x[1] != "BEST" and x[
            1] != "BETTER" and x[1] != "BETWEEN" and x[1] != "BEYOND" and x[1] != "BOTH" and x[1] != "BRIEF" and x[
            1] != "BUT" and x[1] != "BY" and x[1] != "C" and x[1] != "C'MON" and x[1] != "C'S" and x[1] != "CAME" and x[
            1] != "CAN" and x[1] != "CAN'T" and x[1] != "CANNOT" and x[1] != "CANT" and x[1] != "CAUSE" and x[
            1] != "CAUSES" and x[1] != "CERTAIN" and x[1] != "CERTAINLY" and x[1] != "CHANGES" and x[1] != "CLEARLY" and
        x[
            1] != "CO" and x[1] != "COM" and x[1] != "COME" and x[1] != "COMES" and x[1] != "CONCERNING" and x[
            1] != "CONSEQUENTLY" and x[1] != "CONSIDER" and x[1] != "CONSIDERING" and x[1] != "CONTAIN" and x[
            1] != "CONTAINING" and x[1] != "CONTAINS" and x[1] != "CORRESPONDING" and x[1] != "COULD" and x[
            1] != "COULDN'T" and x[1] != "COURSE" and x[1] != "CURRENTLY" and x[1] != "D" and x[1] != "DEFINITELY" and
        x[
            1] != "DESCRIBED" and x[1] != "DESPITE" and x[1] != "DID" and x[1] != "DIDN'T" and x[1] != "DIFFERENT" and
        x[
            1] != "DO" and x[1] != "DOES" and x[1] != "DOESN'T" and x[1] != "DOING" and x[1] != "DON'T" and x[
            1] != "DONE" and x[1] != "DOWN" and x[1] != "DOWNWARDS" and x[1] != "DURING" and x[1] != "E" and x[
            1] != "EACH" and x[1] != "EDU" and x[1] != "EG" and x[1] != "EIGHT" and x[1] != "EITHER" and x[
            1] != "ELSE" and
        x[1] != "ELSEWHERE" and x[1] != "ENOUGH" and x[1] != "ENTIRELY" and x[1] != "ESPECIALLY" and x[1] != "ET" and x[
            1] != "ETC" and x[1] != "EVEN" and x[1] != "EVER" and x[1] != "EVERY" and x[1] != "EVERYBODY" and x[
            1] != "EVERYONE" and x[1] != "EVERYTHING" and x[1] != "EVERYWHERE" and x[1] != "EX" and x[
            1] != "EXACTLY" and x[
            1] != "EXAMPLE" and x[1] != "EXCEPT" and x[1] != "F" and x[1] != "FAR" and x[1] != "FEW" and x[
            1] != "FIFTH" and
        x[1] != "FIRST" and x[1] != "FIVE" and x[1] != "FOLLOWED" and x[1] != "FOLLOWING" and x[1] != "FOLLOWS" and x[
            1] != "FOR" and x[1] != "FORMER" and x[1] != "FORMERLY" and x[1] != "FORTH" and x[1] != "FOUR" and x[
            1] != "FROM" and x[1] != "FURTHER" and x[1] != "FURTHERMORE" and x[1] != "G" and x[1] != "GET" and x[
            1] != "GETS" and x[1] != "GETTING" and x[1] != "GIVEN" and x[1] != "GIVES" and x[1] != "GO" and x[
            1] != "GOES" and x[1] != "GOING" and x[1] != "GONE" and x[1] != "GOT" and x[1] != "GOTTEN" and x[
            1] != "GREETINGS" and x[1] != "H" and x[1] != "HAD" and x[1] != "HADN'T" and x[1] != "HAPPENS" and x[
            1] != "HARDLY" and x[1] != "HAS" and x[1] != "HASN'T" and x[1] != "HAVE" and x[1] != "HAVEN'T" and x[
            1] != "HAVING" and x[1] != "HE" and x[1] != "HE'S" and x[1] != "HELLO" and x[1] != "HELP" and x[
            1] != "HENCE" and x[1] != "HER" and x[1] != "HERE" and x[1] != "HERE'S" and x[1] != "HEREAFTER" and x[
            1] != "HEREBY" and x[1] != "HEREIN" and x[1] != "HEREUPON" and x[1] != "HERS" and x[1] != "HERSELF" and x[
            1] != "HI" and x[1] != "HIM" and x[1] != "HIMSELF" and x[1] != "HIS" and x[1] != "HITHER" and x[
            1] != "HOPEFULLY" and x[1] != "HOW" and x[1] != "HOWBEIT" and x[1] != "HOWEVER" and x[1] != "I" and x[
            1] != "I'D" and x[1] != "I'LL" and x[1] != "I'M" and x[1] != "I'VE" and x[1] != "IE" and x[1] != "IF" and x[
            1] != "IGNORED" and x[1] != "IMMEDIATE" and x[1] != "IN" and x[1] != "INASMUCH" and x[1] != "INC" and x[
            1] != "INDEED" and x[1] != "INDICATE" and x[1] != "INDICATED" and x[1] != "INDICATES" and x[
            1] != "INNER" and x[
            1] != "INSOFAR" and x[1] != "INSTEAD" and x[1] != "INTO" and x[1] != "INWARD" and x[1] != "IS" and x[
            1] != "ISN'T" and x[1] != "IT" and x[1] != "IT'D" and x[1] != "IT'LL" and x[1] != "IT'S" and x[
            1] != "ITS" and
        x[1] != "ITSELF" and x[1] != "J" and x[1] != "JUST" and x[1] != "K" and x[1] != "KEEP" and x[1] != "KEEPS" and
        x[
            1] != "KEPT" and x[1] != "KNOW" and x[1] != "KNOWN" and x[1] != "KNOWS" and x[1] != "L" and x[
            1] != "LAST" and
        x[1] != "LATELY" and x[1] != "LATER" and x[1] != "LATTER" and x[1] != "LATTERLY" and x[1] != "LEAST" and x[
            1] != "LESS" and x[1] != "LEST" and x[1] != "LET" and x[1] != "LET'S" and x[1] != "LIKE" and x[
            1] != "LIKED" and
        x[1] != "LIKELY" and x[1] != "LITTLE" and x[1] != "LOOK" and x[1] != "LOOKING" and x[1] != "LOOKS" and x[
            1] != "LTD" and x[1] != "M" and x[1] != "MAINLY" and x[1] != "MANY" and x[1] != "MAY" and x[
            1] != "MAYBE" and x[
            1] != "ME" and x[1] != "MEAN" and x[1] != "MEANWHILE" and x[1] != "MERELY" and x[1] != "MIGHT" and x[
            1] != "MORE" and x[1] != "MOREOVER" and x[1] != "MOST" and x[1] != "MOSTLY" and x[1] != "MUCH" and x[
            1] != "MUST" and x[1] != "MY" and x[1] != "MYSELF" and x[1] != "N" and x[1] != "NAME" and x[
            1] != "NAMELY" and
        x[1] != "ND" and x[1] != "NEAR" and x[1] != "NEARLY" and x[1] != "NECESSARY" and x[1] != "NEED" and x[
            1] != "NEEDS" and x[1] != "NEITHER" and x[1] != "NEVER" and x[1] != "NEVERTHELESS" and x[1] != "NEW" and x[
            1] != "NEXT" and x[1] != "NINE" and x[1] != "NO" and x[1] != "NOBODY" and x[1] != "NON" and x[
            1] != "NONE" and
        x[1] != "NOONE" and x[1] != "NOR" and x[1] != "NORMALLY" and x[1] != "NOT" and x[1] != "NOTHING" and x[
            1] != "NOVEL" and x[1] != "NOW" and x[1] != "NOWHERE" and x[1] != "O" and x[1] != "OBVIOUSLY" and x[
            1] != "OF" and x[1] != "OFF" and x[1] != "OFTEN" and x[1] != "OH" and x[1] != "OK" and x[1] != "OKAY" and x[
            1] != "OLD" and x[1] != "ON" and x[1] != "ONCE" and x[1] != "ONE" and x[1] != "ONES" and x[1] != "ONLY" and
        x[
            1] != "ONTO" and x[1] != "OR" and x[1] != "OTHER" and x[1] != "OTHERS" and x[1] != "OTHERWISE" and x[
            1] != "OUGHT" and x[1] != "OUR" and x[1] != "OURS" and x[1] != "OURSELVES" and x[1] != "OUT" and x[
            1] != "OUTSIDE" and x[1] != "OVER" and x[1] != "OVERALL" and x[1] != "OWN" and x[1] != "P" and x[
            1] != "PARTICULAR" and x[1] != "PARTICULARLY" and x[1] != "PER" and x[1] != "PERHAPS" and x[
            1] != "PLACED" and
        x[1] != "PLEASE" and x[1] != "PLUS" and x[1] != "POSSIBLE" and x[1] != "PRESUMABLY" and x[1] != "PROBABLY" and
        x[
            1] != "PROVIDES" and x[1] != "Q" and x[1] != "QUE" and x[1] != "QUITE" and x[1] != "QV" and x[1] != "R" and
        x[
            1] != "RATHER" and x[1] != "RD" and x[1] != "RE" and x[1] != "REALLY" and x[1] != "REASONABLY" and x[
            1] != "REGARDING" and x[1] != "REGARDLESS" and x[1] != "REGARDS" and x[1] != "RELATIVELY" and x[
            1] != "RESPECTIVELY" and x[1] != "RIGHT" and x[1] != "S" and x[1] != "SAID" and x[1] != "SAME" and x[
            1] != "SAW" and x[1] != "SAY" and x[1] != "SAYING" and x[1] != "SAYS" and x[1] != "SECOND" and x[
            1] != "SECONDLY" and x[1] != "SEE" and x[1] != "SEEING" and x[1] != "SEEM" and x[1] != "SEEMED" and x[
            1] != "SEEMING" and x[1] != "SEEMS" and x[1] != "SEEN" and x[1] != "SELF" and x[1] != "SELVES" and x[
            1] != "SENSIBLE" and x[1] != "SENT" and x[1] != "SERIOUS" and x[1] != "SERIOUSLY" and x[1] != "SEVEN" and x[
            1] != "SEVERAL" and x[1] != "SHALL" and x[1] != "SHE" and x[1] != "SHOULD" and x[1] != "SHOULDN'T" and x[
            1] != "SINCE" and x[1] != "SIX" and x[1] != "SO" and x[1] != "SOME" and x[1] != "SOMEBODY" and x[
            1] != "SOMEHOW" and x[1] != "SOMEONE" and x[1] != "SOMETHING" and x[1] != "SOMETIME" and x[
            1] != "SOMETIMES" and
        x[1] != "SOMEWHAT" and x[1] != "SOMEWHERE" and x[1] != "SOON" and x[1] != "SORRY" and x[1] != "SPECIFIED" and x[
            1] != "SPECIFY" and x[1] != "SPECIFYING" and x[1] != "STILL" and x[1] != "SUB" and x[1] != "SUCH" and x[
            1] != "SUP" and x[1] != "SURE" and x[1] != "T" and x[1] != "T'S" and x[1] != "TAKE" and x[1] != "TAKEN" and
        x[
            1] != "TELL" and x[1] != "TENDS" and x[1] != "TH" and x[1] != "THAN" and x[1] != "THANK" and x[
            1] != "THANKS" and x[1] != "THANX" and x[1] != "THAT" and x[1] != "THAT'S" and x[1] != "THATS" and x[
            1] != "THE" and x[1] != "THEIR" and x[1] != "THEIRS" and x[1] != "THEM" and x[1] != "THEMSELVES" and x[
            1] != "THEN" and x[1] != "THENCE" and x[1] != "THERE" and x[1] != "THERE'S" and x[1] != "THEREAFTER" and x[
            1] != "THEREBY" and x[1] != "THEREFORE" and x[1] != "THEREIN" and x[1] != "THERES" and x[
            1] != "THEREUPON" and
        x[1] != "THESE" and x[1] != "THEY" and x[1] != "THEY'D" and x[1] != "THEY'LL" and x[1] != "THEY'RE" and x[
            1] != "THEY'VE" and x[1] != "THINK" and x[1] != "THIRD" and x[1] != "THIS" and x[1] != "THOROUGH" and x[
            1] != "THOROUGHLY" and x[1] != "THOSE" and x[1] != "THOUGH" and x[1] != "THREE" and x[1] != "THROUGH" and x[
            1] != "THROUGHOUT" and x[1] != "THRU" and x[1] != "THUS" and x[1] != "TO" and x[1] != "TOGETHER" and x[
            1] != "TOO" and x[1] != "TOOK" and x[1] != "TOWARD" and x[1] != "TOWARDS" and x[1] != "TRIED" and x[
            1] != "TRIES" and x[1] != "TRULY" and x[1] != "TRY" and x[1] != "TRYING" and x[1] != "TWICE" and x[
            1] != "TWO" and x[1] != "U" and x[1] != "UN" and x[1] != "UNDER" and x[1] != "UNFORTUNATELY" and x[
            1] != "UNLESS" and x[1] != "UNLIKELY" and x[1] != "UNTIL" and x[1] != "UNTO" and x[1] != "UP" and x[
            1] != "UPON" and x[1] != "US" and x[1] != "USE" and x[1] != "USED" and x[1] != "USEFUL" and x[
            1] != "USES" and
        x[1] != "USING" and x[1] != "USUALLY" and x[1] != "UUCP" and x[1] != "V" and x[1] != "VALUE" and x[
            1] != "VARIOUS" and x[1] != "VERY" and x[1] != "VIA" and x[1] != "VIZ" and x[1] != "VS" and x[1] != "W" and
        x[
            1] != "WANT" and x[1] != "WANTS" and x[1] != "WAS" and x[1] != "WASN'T" and x[1] != "WAY" and x[
            1] != "WE" and
        x[1] != "WE'D" and x[1] != "WE'LL" and x[1] != "WE'RE" and x[1] != "WE'VE" and x[1] != "WELCOME" and x[
            1] != "WELL" and x[1] != "WENT" and x[1] != "WERE" and x[1] != "WEREN'T" and x[1] != "WHAT" and x[
            1] != "WHAT'S" and x[1] != "WHATEVER" and x[1] != "WHEN" and x[1] != "WHENCE" and x[1] != "WHENEVER" and x[
            1] != "WHERE" and x[1] != "WHERE'S" and x[1] != "WHEREAFTER" and x[1] != "WHEREAS" and x[1] != "WHEREBY" and
        x[
            1] != "WHEREIN" and x[1] != "WHEREUPON" and x[1] != "WHEREVER" and x[1] != "WHETHER" and x[1] != "WHICH" and
        x[
            1] != "WHILE" and x[1] != "WHITHER" and x[1] != "WHO" and x[1] != "WHO'S" and x[1] != "WHOEVER" and x[
            1] != "WHOLE" and x[1] != "WHOM" and x[1] != "WHOSE" and x[1] != "WHY" and x[1] != "WILL" and x[
            1] != "WILLING" and x[1] != "WISH" and x[1] != "WITH" and x[1] != "WITHIN" and x[1] != "WITHOUT" and x[
            1] != "WON'T" and x[1] != "WONDER" and x[1] != "WOULD" and x[1] != "WOULDN'T" and x[1] != "X" and x[
            1] != "Y" and x[1] != "YES" and x[1] != "YET" and x[1] != "YOU" and x[1] != "YOU'D" and x[1] != "YOU'LL" and
        x[
            1] != "YOU'RE" and x[1] != "YOU'VE" and x[1] != "YOUR" and x[1] != "YOURS" and x[1] != "YOURSELF" and x[
            1] != "YOURSELVES" and x[1] != "Z" and x[1] != "ZERO"))
    keyword = keyword.map(lambda x: (x[0], x[1].lower()))
    data = keyword.join(dates).map(lambda x: (x[0], x[1][0], x[1][1]))

    df = spark.createDataFrame(data.map(lambda x: Row(id_tweet=x[0], keyword=x[1], timestamp=x[2])))
    df.createOrReplaceTempView("keyword")
    df = spark.sql("create database if not exists thsfulltext")
    df = spark.sql("use thsfulltext")
    df = spark.sql("SELECT id_tweet, keyword, timestamp FROM keyword")
    df.write.mode("append").saveAsTable("tb_keyword")
    print("tb_keyword inserted")


def p1(time, rdd):
    rdd = rdd.map(lambda x: json.loads(x[1]))
    spark = getSparkSessionInstance(rdd.context.getConf())
    # Raw Tweet
    if rdd.count() > 0:
        tweets = rdd.filter(lambda y: (y["retweeted"] == False and "retweeted_status" not in y))
        if tweets.count() > 0:
            insertRawTweets(tweets, spark, time)
            insertTweets(tweets, spark, time)
            insertUser(tweets, spark)
            insertHashtags(tweets, spark)
            insertTweet(tweets, spark)
            insertKeyword(tweets, spark)
        else:
            print("Just retweets to insert...")
    else:
        print("No raw tweets avaliable to insert in hive.")


if __name__ == "__main__":
    print("Starting to read tweets")
    print("Startup at", datetime.now())
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED", "123")
    sc = SparkContext(appName="defaultConsumer", conf=conf)
    geolocator = Nominatim(timeout=None)
    consumer()
