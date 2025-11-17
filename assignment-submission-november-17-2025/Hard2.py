import pymysql

#connecting to MySQL database
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='host12345678',  
    database='f1_2025'
)

cursor = conn.cursor()

#writng query for position gains
query = """
SELECT 
    q.Track,
    q.Driver AS DriverName,
    q.Position AS QualifyingPos,
    r.Position AS RacePos,
    (q.Position - r.Position) AS PositionsGained
FROM f1_2025_qualifyingresults q
JOIN f1_2025_raceresults r
    ON q.Track = r.Track
    AND q.Driver = r.Driver
WHERE (q.Position - r.Position) > 0
ORDER BY PositionsGained DESC;
"""

cursor.execute(query)
results = cursor.fetchall()

print("\n2025 Qualifying vs Race Position Improvements\n")
print(f"{'Driver':25} {'Track':20} {'Q':>3} {'R':>3} {'Gain':>5}")
print("-" * 70)

for row in results:
    track, driver, qpos, rpos, gain = row
    print(f"{driver:25} {track:20} {qpos:>3} {rpos:>3} {gain:>5}")

#close Connection
cursor.close()
conn.close()
