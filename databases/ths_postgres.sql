DROP DATABASE IF EXISTS twitter_streaming;
CREATE DATABASE twitter_streaming WITH OWNER = postgres ENCODING = 'UTF8';
\connect twitter_streaming

create table end_user(
	id SERIAL PRIMARY KEY NOT NULL,
	last_name TEXT NOT NULL, 
	first_name TEXT NOT NULL,
	email TEXT NOT NULL,
	username TEXT NOT NULL, 
	password TEXT NOT NULL, 
	phone_number TEXT
);

create table end_user_profile(
	id SERIAL NOT NULL, 
	end_user_id INTEGER,
	address TEXT, 
	date_birth TEXT,
	gender VARCHAR(6),
	role TEXT,
	PRIMARY KEY(id, end_user_id),
    FOREIGN KEY(end_user_id) REFERENCES end_user(id)
);

-- raw_tweet_id from hive table tweet
create table study(
	id BIGSERIAL PRIMARY KEY NOT NULL,
	query TEXT NOT NULL, 
	is_spark BOOLEAN
);

create table owns(
	end_user_id BIGINT, 
	study_id BIGINT,
	PRIMARY KEY(end_user_id, study_id),
	FOREIGN KEY(end_user_id) REFERENCES end_user(id),
    FOREIGN KEY(study_id) REFERENCES study(id)
);

create table tweet_is_part_of_study(
	id_study BIGINT NOT NULL, 
	id_tweet_hive BIGINT NOT NULL,
	PRIMARY KEY(id_study, id_tweet_hive),
    FOREIGN KEY(id_study) REFERENCES study(id)
);

create table training_window(
	id BIGSERIAL PRIMARY KEY, 
	date TEXT,
	consecutive INTEGER
);

create table trained_tweet(
	id BIGSERIAL PRIMARY KEY NOT NULL, 
	result TEXT, 
	is_completed BOOLEAN,
	id_tweet_hive BIGINT NOT NULL
);

create table has_trained_tweet(
	id_training_window BIGINT,
	trained_tweet_id BIGINT,
	PRIMARY KEY(id_training_window, trained_tweet_id),
	FOREIGN KEY(id_training_window) REFERENCES training_window(id),
	FOREIGN KEY(trained_tweet_id) REFERENCES trained_tweet(id)
);

create table trainer_of(
	end_user_id BIGINT,
	trained_tweet_id BIGINT ,
	PRIMARY KEY(end_user_id, trained_tweet_id),
	FOREIGN KEY(end_user_id) REFERENCES end_user(id),
    FOREIGN KEY(trained_tweet_id) REFERENCES trained_tweet(id)
);
		
