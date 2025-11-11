import pymysql

def my_sql_connector():
    conn = pymysql.connect(
        host='localhost',
        user='root', 
        password = "host12345678",
        db='health_data',
        )
    
    cur = conn.cursor()
    cur.execute("select * FROM patients;")
    output = cur.fetchall()
    print(output)
    
    # To close the connection
    conn.close()

# Driver Code
if __name__ == "__main__" :
    my_sql_connector()


import pymysql

def show_table_columns():
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='host12345678',  # use your current password
            db='health_data'
        )
        cur = conn.cursor()

        # Show column details for the Patients table
        cur.execute("DESCRIBE Patients;")
        columns = cur.fetchall()

        print("Columns in Patients table:\n")
        print("{:<15} {:<15} {:<10}".format("Field", "Type", "Null"))
        print("-" * 40)
        for col in columns:
            print("{:<15} {:<15} {:<10}".format(col[0], col[1], col[2]))

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()

# Run the function
if __name__ == "__main__":
    show_table_columns()
