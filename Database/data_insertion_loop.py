# data_insertion_loop.py
import psycopg2
import psycopg2.extras as extras
import pandas as pd
from sklearn.datasets import fetch_california_housing

def get_data():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    df = pd.concat([X, y], axis="columns")
    return df

#### origin code lines for insertion
# def insert_data(db_connect, data):
#     insert_row_query = f"""
#     INSERT INTO housing_Data
#         (timestamp, MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
#         Latitude, Longitude, MedHouseVal)
#         VALUES (
#             NOW(),
#             {data.MedInc},
#             {data.HouseAge},
#             {data.AveRooms},
#             {data.AveBedrms},
#             {data.Population},
#             {data.AveOccup},
#             {data.Latitude},
#             {data.Longitude},
#             {data.MedHouseVal}
#         );
#     """
#     # print(insert_row_query)
#     with db_connect.cursor() as cur:
#         cur.execute(insert_row_query)
#         db_connect.commit()
#         print('Complete insertion')

# def generate_data(db_connect, df):
#     while True:
#         insert_data(db_connect, df.sample(1).squeeze())
#         time.sleep(random.uniform(0, 1))

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
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5432,
        database="mydatabase",
    )
    df = get_data()
    execute_batch(db_connect, df, "housing_data", page_size=100)