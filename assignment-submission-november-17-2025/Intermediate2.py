import pymysql

#MySQL Connection
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="host12345678",
    database="f1",
    cursorclass=pymysql.cursors.DictCursor
)

cursor = conn.cursor()

# witing Query for most positions Gained in 2022
query = """
WITH race_positions AS (
    SELECT 
        d.driverId,
        CONCAT(d.forename, ' ', d.surname) AS driver_name,
        ra.year,
        r.grid,
        r.position,
        (r.grid - r.position) AS positions_gained
    FROM results r
    JOIN drivers d ON d.driverId = r.driverId
    JOIN races ra ON ra.raceId = r.raceId
    WHERE ra.year = 2022
      AND r.position > 0   -- ignore DNFs
      AND r.grid > 0       -- ignore pit-lane or undefined starts
)
SELECT 
    driverId,
    driver_name,
    AVG(positions_gained) AS avg_positions_gained,
    COUNT(*) AS races_count,
    RANK() OVER (ORDER BY AVG(positions_gained) DESC) AS gain_rank
FROM race_positions
GROUP BY driverId, driver_name
ORDER BY gain_rank;
"""

cursor.execute(query)
rows = cursor.fetchall()

#cleaning the table output

headers = [
    "Rank", "Driver", "Avg Gained", "Races"
]

print("\n Drivers Who Gained the Most Positions (2022 Season) \n")
print(f"{headers[0]:<6}{headers[1]:<25}{headers[2]:<12}{headers[3]:<8}")
print("-" * 60)

for row in rows:
    print(
        f"{row['gain_rank']:<6}"
        f"{row['driver_name']:<25}"
        f"{float(row['avg_positions_gained']):<12.2f}"
        f"{row['races_count']:<8}"
    )

cursor.close()
conn.close()
