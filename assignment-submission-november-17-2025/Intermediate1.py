import pymysql

#connecting to MySQL
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="host12345678",
    database="f1",
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:

        query = """
        SELECT 
            c.constructorId,
            c.name AS constructor_name,
            SUM(r.points) AS total_points_2022
        FROM results r
        JOIN constructors c ON r.constructorId = c.constructorId
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.year = 2022
        GROUP BY c.constructorId, c.name
        ORDER BY total_points_2022 DESC;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        print("\nConstructor Points (2022 Season):")
        for row in results:
            print(f"{row['constructor_name']}: {row['total_points_2022']} points")

finally:
    connection.close()
