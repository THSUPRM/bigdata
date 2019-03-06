#!/usr/bin/env bash

nohup kafka/bin/kafka-server-start.sh kafka/config/server.properties > logs/KafkaServer.out 2> logs/KafkaServer.err &
