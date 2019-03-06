from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row
import psycopg2
from configparser import ConfigParser
import sys
from datetime import datetime, timedelta


# run with: spark-submit code/update_store.py
def getSparkSessionInstance(sparkConf):
    if ('sparkSessionInstance' not in globals()):
        globals()['sparkSessionInstance'] = SparkSession.builder.config(conf=sparkConf) \
                                            .enableHiveSupport().getOrCreate()
    return globals()['sparkSessionInstance']


def config(filename='/home/manuelr/code/database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def update(fromtime, totime):
    spark = getSparkSessionInstance(sc.getConf())
    spark.sql("use thsfulltext")

    # Tweet count
    df = spark.sql('''
                    SELECT count(*) AS ct
                    FROM tb_tweet
                    WHERE unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') >= unix_timestamp('{0}', 'yyyy-MM-dd HH:mm:ss Z')
                    AND unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') < unix_timestamp('{1}', 'yyyy-MM-dd HH:mm:ss Z')
                    '''.format(fromtime, totime))
    tweetct = df.collect()

    # Tweet count by disease
    df = spark.sql('''
                    SELECT disease, count(*) AS ct
                    FROM tb_tweet
                    WHERE unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') >= unix_timestamp('{0}', 'yyyy-MM-dd HH:mm:ss Z')
                    AND unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') < unix_timestamp('{1}', 'yyyy-MM-dd HH:mm:ss Z')
                    GROUP BY disease
                    '''.format(fromtime, totime))
    tweetctbd = df.collect()

    # User count
    df = spark.sql('''
                    SELECT id_user, first_value(screen_name) AS sn, first_value(location) as lc, count(*) AS ct
                    FROM tb_user
                    WHERE unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') >= unix_timestamp('{0}', 'yyyy-MM-dd HH:mm:ss Z')
                    AND unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') < unix_timestamp('{1}', 'yyyy-MM-dd HH:mm:ss Z')
                    GROUP BY id_user
                    '''.format(fromtime, totime))
    users = df.collect()
    df.show()

    # Keyword count
    df = spark.sql('''
                    SELECT keyword, count(*) AS ct
                    FROM tb_keyword
                    WHERE unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') >= unix_timestamp('{0}', 'yyyy-MM-dd HH:mm:ss Z')
                    AND unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') < unix_timestamp('{1}', 'yyyy-MM-dd HH:mm:ss Z')
                    GROUP BY keyword
                    '''.format(fromtime, totime))
    kw = df.collect()

    # Hashtag count
    df = spark.sql('''
                        SELECT hashtag, count(*) AS ct
                        FROM tb_hashtag
                        WHERE unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') >= unix_timestamp('{0}', 'yyyy-MM-dd HH:mm:ss Z')
                        AND unix_timestamp(`timestamp`, 'EEE MMM dd HH:mm:ss Z yyyy') < unix_timestamp('{1}', 'yyyy-MM-dd HH:mm:ss Z')
                        GROUP BY hashtag
                        '''.format(fromtime, totime))
    ht = df.collect()

    # Connect to the PostgreSQL database server
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # add to tweet_count
        for row in tweetct:
            cur.execute("INSERT INTO tweet_count(tcount, timestamp) VALUES(%s,%s);", (row.ct, totime))

        # add to disease
        for row in tweetctbd:
            cur.execute("INSERT INTO disease(disease_name, mentions, timestamp) VALUES(%s,%s,%s);", (row.disease, row.ct, totime))

        # add to user
        for row in users:
            cur.execute("INSERT INTO users(user_id, user_name, user_location, mentions, timestamp) VALUES(%s,%s,%s,%s,%s);", (row.id_user, row.sn, row.lc, row.ct, totime))

        # add to keyword
        for row in kw:
            cur.execute("INSERT INTO keywords(keyword_text, mentions, timestamp) VALUES(%s,%s,%s);", (row.keyword, row.ct, totime))

        # add to keyword
        for row in ht:
            cur.execute("INSERT INTO hashtags(hashtag_text, mentions, timestamp) VALUES(%s,%s,%s);", (row.hashtag, row.ct, totime))

        # close the communication with the PostgreSQL
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# Iterate in 6 hour intervals
def datetimerange(start_datetime, end_datetime):
    for n in range(int((end_datetime - start_datetime).days)*4):
        yield start_datetime + timedelta(hours=6*n)

if __name__ == "__main__":
    # spark-submit update-store.py fromtimeyear fromtimemonth fromtimeday fromtimehour totimeyear totimemonth totimeday totimehour
    sc = SparkContext(appName="Update Store")
    # Run retroactively with hardcoded dates in 6 hour intervals
    if len(sys.argv) == 1:
        start_datetime = datetime(2018, 10, 16)
        end_datetime = datetime(2018, 10, 21)
        for single_datetime in datetimerange(start_datetime, end_datetime):
            fromtime = single_datetime
            totime = fromtime + timedelta(hours=6)
            print("Working on " + fromtime.strftime("%Y-%m-%d %H:%M:%S") + " to " + totime.strftime("%Y-%m-%d %H:%M:%S"))
            update(fromtime.isoformat().replace("T", " ") + " -0400", totime.isoformat().replace("T", " ") + " -0400")

    # Run once with parametrized dates
    elif len(sys.argv) == 9:
        fromtimetup = (int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
        totimetup = (int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]))
        fromtime = datetime(fromtimetup[0], fromtimetup[1], fromtimetup[2], fromtimetup[3], 0, 0).isoformat().replace("T", " ") + " -0400"
        totime = datetime(totimetup[0], totimetup[1], totimetup[2], totimetup[3], 0, 0).isoformat().replace("T", " ") + " -0400"
        update(fromtime, totime)

    # Run once with to time, for the 6 hours before
    elif len(sys.argv) == 5:
        totimetup = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        totime_datetime = datetime(totimetup[0], totimetup[1], totimetup[2], totimetup[3], 0, 0)
        fromtime_datetime = totime_datetime - timedelta(hours=6)
        fromtime = fromtime_datetime.isoformat().replace("T", " ") + " -0400"
        totime = totime_datetime.isoformat().replace("T", " ") + " -0400"
        update(fromtime, totime)

    else:
        sys.exit("Error: Invalid syntax")
