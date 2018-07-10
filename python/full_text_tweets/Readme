In this folder is all code necessary to collect the Tweets from Twitter Streaming API save in Hive, process the data collected and create the new tables in Hive, save the aggregate counts in some new tables in Postgresql to show in the dashboard to the final client, all those scripts could run in the client node 136.145.115.135.

1.	Run “1.Producer_full_text.py” with this command: 
“nohup spark-submit code/1.Producer_full_text.py > logs/logProducerFullText.out 2> logs/logProducerFullText.err &”
2.	Run “2.Consumer_full_text.py” with this command:
“nohup spark-submit code/2.Consumer_full_text.py > logs/logConsumerFullText.out 2> logs/logConsumerFullText.err &”
3.	Once the number of tweets are collected and saved table raw_tweet in hive, is necessary run “3.Save_data_from_raw_tweet_to_tables_Hive.py” with this command:
“spark-submit code/3.Save_data_from_raw_tweet_to_tables_Hive.py”
4.	When the data is distributed in all the tables is necessary to extract the aggregate numbers of each to show just the important indicators to the final client, to do this is necessary run “4.Postgresql_create_tables.py” that create all the tables necessary to save the data. It runs with this command: “spark-submit code/4.Postgresql_create_tables.py”
5.	 Run “5.Extract_from_Hive_to_Postgresql.py” with this command: “spark-submit code/5.Extract_from_Hive_to_Postgresql.py” 
