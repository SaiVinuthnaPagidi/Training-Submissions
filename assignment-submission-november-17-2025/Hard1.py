import pymysql

#connecting to MysQL
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="host12345678",
    database="f1",
    cursorclass=pymysql.cursors.DictCursor
)

cursor = conn.cursor()

query = """
WITH podium_times AS (
    SELECT
        ra.raceId,
        cir.name AS circuit_name,
        r.positionOrder,
        r.milliseconds,
        ra.year
    FROM results r
    JOIN races ra ON r.raceId = ra.raceId
    JOIN circuits cir ON ra.circuitId = cir.circuitId
    WHERE r.positionOrder IN (1, 2, 3)
      AND r.milliseconds IS NOT NULL
      AND ra.year BETWEEN 2013 AND 2022
),
podium_groups AS (
    SELECT
        circuit_name,
        raceId,
        MAX(CASE WHEN positionOrder = 1 THEN milliseconds END) AS p1_time,
        MAX(CASE WHEN positionOrder = 3 THEN milliseconds END) AS p3_time
    FROM podium_times
    GROUP BY circuit_name, raceId
),
podium_gaps AS (
    SELECT
        circuit_name,
        (p3_time - p1_time) / 1000 AS podium_gap_seconds
    FROM podium_groups
    WHERE p1_time IS NOT NULL AND p3_time IS NOT NULL
)
SELECT
    circuit_name,
    ROUND(AVG(podium_gap_seconds), 2) AS avg_podium_gap,
    COUNT(*) AS num_races
FROM podium_gaps
GROUP BY circuit_name
HAVING num_races >= 2
ORDER BY avg_podium_gap ASC
LIMIT 15;
"""

cursor.execute(query)
results = cursor.fetchall()

print("\nClosest Podium Battles (2013–2022)\n")
print(f"{'Circuit':35} {'Avg Gap (s)':>12} {'Races':>8}")
print("-" * 60)

for row in results:
    print(f"{row['circuit_name'][:33]:35} "
          f"{row['avg_podium_gap']:12} "
          f"{row['num_races']:8}")

cursor.close()
conn.close()

