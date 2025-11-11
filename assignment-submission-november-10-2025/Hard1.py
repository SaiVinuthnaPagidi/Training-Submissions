import pymysql
import time

def optimized_query():
    try:
        #connect to MySQL database
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678',
            db='health_data'
        )
        cur = conn.cursor()

        #get the earliest and latest years dynamically
        cur.execute("""
            SELECT 
                MIN(YEAR(start)) AS earliest_year,
                MAX(YEAR(start)) AS latest_year
            FROM conditions;
        """)
        years = cur.fetchone()
        earliest_year, latest_year = years[0], years[1]

        print(f"Dataset year range: {earliest_year} to {latest_year}\n")

        #run optimized query using the detected year range
        start_time = time.time()

        cur.execute(f"""
            SELECT description, COUNT(*) AS total_cases
            FROM conditions
            WHERE YEAR(start) BETWEEN {earliest_year} AND {latest_year}
            GROUP BY description
            ORDER BY total_cases DESC
            LIMIT 10;
        """)

        results = cur.fetchall()
        duration = round(time.time() - start_time, 4)

        #displaying results
        print("Top 10 Most Common Conditions:\n")
        print("{:<40} {:<15}".format("Condition", "Total Cases"))
        print("-" * 55)
        for row in results:
            print("{:<40} {:<15}".format(row[0], row[1]))

        print(f"\nQuery executed in {duration} seconds")

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()

if __name__ == "__main__":
    optimized_query()
