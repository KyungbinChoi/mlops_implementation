# data_insertion_loop.py
from argparse import ArgumentParser
import psycopg2
import psycopg2.extras as extras
import pandas as pd
from sklearn.datasets import fetch_california_housing

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS housing_data (
        id SERIAL PRIMARY KEY,
        MedInc float8, 
        HouseAge int, 
        AveRooms float8,
        AveBedrms float8,
        Population int,
        AveOccup float8,
        Latitude float8,
        Longitude float8,
        MedHouseVal float8 
    );"""
    # print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()
        print('Created table')

def get_data():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    df = pd.concat([X, y], axis="columns")
    return df

def execute_batch(conn, df, table, page_size=100):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s)" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_batch(cursor, query, tuples, page_size)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_batch() done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()

    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host=args.db_host,
        port=5432,
        database="mydatabase"
    )
    create_table(db_connect)
    df = get_data()
    execute_batch(db_connect, df, "housing_data", page_size=100)

