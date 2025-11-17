import pymysql

def sql_data_quality_checks():
    try:
        # Step 1: Connect to database
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678', 
            db='health_data'
        )
        cur = conn.cursor()

        #check for duplicate patient IDs
        print("Checking for duplicate patient IDs\n")
        cur.execute("""
            SELECT patient, COUNT(*) AS record_count
            FROM patients
            GROUP BY patient
            HAVING COUNT(*) > 1
            ORDER BY record_count DESC;
        """)
        duplicates = cur.fetchall()
        if len(duplicates) == 0:
            print("No duplicate records found.\n")
        else:
            print("Duplicate records found:\n", duplicates, "\n")

        #check for inconsistent text formatting in gender or race
        print("Checking for inconsistent gender or race values\n")
        cur.execute("""
            SELECT DISTINCT TRIM(UPPER(gender)) AS formatted_gender
            FROM patients
            WHERE gender IS NOT NULL;
        """)
        genders = [row[0] for row in cur.fetchall()]
        print("Unique gender values (standardized):", genders, "\n")

        cur.execute("""
            SELECT DISTINCT TRIM(UPPER(race)) AS formatted_race
            FROM patients
            WHERE race IS NOT NULL;
        """)
        races = [row[0] for row in cur.fetchall()]
        print("Unique race values (standardized):", races, "\n")

        #check for invalid (future) birthdates
        print("Checking for invalid (future) birthdates:\n")
        cur.execute("""
            SELECT patient, birthdate
            FROM patients
            WHERE birthdate > CURDATE();
        """)
        invalid_dates = cur.fetchall()
        if len(invalid_dates) == 0:
            print("No invalid birthdates found.\n")
        else:
            print("Invalid birthdates detected:\n", invalid_dates, "\n")

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()

if __name__ == "__main__":
    sql_data_quality_checks()
