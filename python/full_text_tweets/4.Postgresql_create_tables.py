import psycopg2
from configparser import ConfigParser

def config(filename='database.ini', section='postgresql'):
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


def create_tables():
    commands = (
        """
        CREATE TABLE IF NOT EXISTS disease (
            disease_id SERIAL PRIMARY KEY,
            disease_name VARCHAR(255) NOT NULL,
            mentions INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tweet_count (
            id SERIAL PRIMARY KEY,
            tcount INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS hashtags (
            serial_id SERIAL PRIMARY KEY,
            hashtag_text VARCHAR(255) NOT NULL,
            mentions INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS keywords (
            serial_id SERIAL PRIMARY KEY,
            keyword_text VARCHAR(255) NOT NULL,
            mentions INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            serial_id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            user_name VARCHAR(255) NOT NULL,
            user_location VARCHAR(255),
            mentions INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    create_tables()