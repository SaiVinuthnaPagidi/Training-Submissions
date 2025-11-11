import pymysql

def sql_intermediate_1():
    try:
        #connecting to MySQL database
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678',
            db='health_data'
        )
        cur = conn.cursor()
        #Query joining patients and encounters
        cur.execute("""
            SELECT p.gender, COUNT(e.id) AS total_encounters
            FROM patients p
            JOIN encounters e ON p.patient = e.patient
            GROUP BY p.gender
            ORDER BY total_encounters DESC;
        """)

        #fetch and display results
        results = cur.fetchall()
        print("Total encounters by gender:\n")
        print("{:<8} {:<15}".format("Gender", "Total Encounters"))
        print("-" * 30)
        for row in results:
            print("{:<8} {:<15}".format(row[0], row[1]))

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()

if __name__ == "__main__":
    sql_intermediate_1()
