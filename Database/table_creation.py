import psycopg2

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


if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect)


