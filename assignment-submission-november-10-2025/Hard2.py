import pymysql

def chronic_meds_by_gender():
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678',
            db='health_data'
        )
        cur = conn.cursor()

        print("Analyzing chronic-condition medication patterns by gender\n")

        cur.execute("""
            SELECT 
                p.gender AS gender,
                c.DESCRIPTION AS condition_name,
                m.DESCRIPTION AS medication,
                COUNT(DISTINCT m.PATIENT) AS total_patients
            FROM medications m
            JOIN conditions c ON m.PATIENT = c.PATIENT
            JOIN patients p ON m.PATIENT = p.PATIENT
            WHERE (c.DESCRIPTION LIKE '%Hypertension%' 
                OR c.DESCRIPTION LIKE '%Diabetes%' 
                OR c.DESCRIPTION LIKE '%Prediabetes%')
              AND (
                m.DESCRIPTION LIKE '%Metformin%' 
                OR m.DESCRIPTION LIKE '%Insulin%' 
                OR m.DESCRIPTION LIKE '%Lisinopril%' 
                OR m.DESCRIPTION LIKE '%Amlodipine%' 
                OR m.DESCRIPTION LIKE '%Hydrochlorothiazide%' 
                OR m.DESCRIPTION LIKE '%Glipizide%' 
                OR m.DESCRIPTION LIKE '%Losartan%'
              )
            GROUP BY p.gender, c.DESCRIPTION, m.DESCRIPTION
            ORDER BY total_patients DESC
            LIMIT 10;
        """)

        print("{:<10} {:<35} {:<45} {:<10}".format("Gender", "Condition", "Medication", "Patients"))
        print("-" * 110)
        for row in cur.fetchall():
            print("{:<10} {:<35} {:<45} {:<10}".format(row[0], row[1], row[2], row[3]))

    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    chronic_meds_by_gender()
