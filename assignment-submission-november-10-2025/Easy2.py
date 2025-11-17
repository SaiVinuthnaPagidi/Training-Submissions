#using pymysql 

import pymysql

def sql_toy_example():
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678',
            db='health_data'
        )
        cur = conn.cursor()

        # finding male patients born in 'Boston'
        cur.execute("""
            SELECT first, last, gender, birthplace
            FROM patients
            WHERE gender = 'M' AND birthplace = 'Boston MA US';
        """)

        results = cur.fetchall()

        print("Male patients born in Boston:\n")
        print("{:<12} {:<12} {:<8} {:<15}".format("First", "Last", "Gender", "Birthplace"))
        print("-" * 50)
        for row in results:
            print("{:<12} {:<12} {:<8} {:<15}".format(row[0], row[1], row[2], row[3]))

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()

if __name__ == "__main__":
    sql_toy_example()