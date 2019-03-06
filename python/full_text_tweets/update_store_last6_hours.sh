#!/usr/bin/env bash
export PATH=/home/manuelr/sqoop/bin:/home/manuelr/spark2/bin:/home/manuelr/hadoop-2.7.3/bin:/home/manuelr/hadoop-2.7.3/sbin:/usr/lib/jvm/java-8-oracle/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/manuelr/hive2/bin:/home/manuelr/sqoop/binexport:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin
export PYTHONPATH=/tmp/spark-2943518c-1d74-47f0-b061-012122b12027/userFiles-a3c44577-b88e-4db3-aa0f-b4b9dde61c18:/usr/lib/python3/dist-packages:/home/manuelr/spark2/python/lib/py4j-0.10.4-src.zip:/home/manuelr/spark2/python:/home/manuelr/logs:/usr/lib/python35.zip:/usr/lib/python3.5:/usr/lib/python3.5/plat-x86_64-linux-gnu:/usr/lib/python3.5/lib-dynload:/home/manuelr/.local/lib/python3.5/site-packages:/usr/local/lib/python3.5/dist-packages
export SPARK_MASTER_HOST=136.145.115.130
export JAVA_HOME=/usr/lib/jvm/java-8-oracle/jre
export HADOOP_CONF_DIR=/home/manuelr/hadoop-2.7.3/etc/hadoop
export SPARK_WORKER_CORES=4
export PYSPARK_PYTHON=/usr/bin/python3.5
echo 'Starting'
TO_YEAR=$(date +'%Y')
TO_MONTH=$(date +'%m')
TO_DAY=$(date +'%d')
TO_HOUR=$(date +'%H')
/home/manuelr/spark2/bin/spark-submit /home/manuelr/code/update_store.py ${TO_YEAR} ${TO_MONTH} ${TO_DAY} ${TO_HOUR} > /home/manuelr/logs/logUpdateStore.out
echo 'Finished'