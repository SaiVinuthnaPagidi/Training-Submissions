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

        #writing query to identiy total points by driver in 2021
        query = """
        SELECT 
            d.driverId,
            d.forename,
            d.surname,
            SUM(r.points) AS total_points_2021
        FROM results r
        JOIN drivers d ON r.driverId = d.driverId
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.year = 2021
        GROUP BY d.driverId, d.forename, d.surname
        ORDER BY total_points_2021 DESC;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        # 3. printing results
        print("\nDRIVER POINTS (2021 SEASON):")
        for row in results:
            print(f"{row['forename']} {row['surname']}: {row['total_points_2021']} points")

finally:
    #closing connection
    connection.close()
