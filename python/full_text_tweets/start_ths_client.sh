#!/usr/bin/env bash

nohup hive --service metastore > logs/logMetastore.out 2> logs/logMetastore.err &
nohup hiveserver2 > logs/logServer.out 2> logs/logServer.err &

sudo systemctl start ths-producer
sleep 15s
sudo systemctl start ths-consumer
sleep 5s
