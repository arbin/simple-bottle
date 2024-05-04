import json
import psycopg2
import psycopg2.extras
import mysql.connector


class RegisterJobs:
    def __init__(self):
        pass

    def load_configuration(self, config_file):
        try:
            with open(config_file, 'r') as config_file:
                config = json.load(config_file)
                return config
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found.")

    def db_connect(self, db_name=None):
        config_file = "config.json"
        config = self.load_configuration(config_file)
        try:
            db_name = config.get("DATABASE_NAME", db_name)
            conn = psycopg2.connect(
                host=config["DATABASE_HOST"],
                database=db_name,
                user=config["DATABASE_USER"],
                password=config["DATABASE_PASS"]
            )
            return conn
        except (psycopg2.Error, Exception) as error:
            print(f"Error connecting to the database: {error}")

    def db_connect_mysql_server(self):
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'tpcprojects'
        }
        connection = mysql.connector.connect(**db_config)

    def cleaned_up_db(self):
        config_file = "config.json"
        config = self.load_configuration(config_file)
        try:
            with self.db_connect(config["DATABASE_NAME"]) as conn:
                with conn.cursor() as cur:
                    tables = ["masterdaily", "jobs", "clientregistration", "columnissuechanges", "columntrackchanges",
                              "processerrorlog", "processconflict", "eoddata"]
                    for table in tables:
                        cur.execute(f"DELETE FROM public.{table}")
                        conn.commit()
        except (psycopg2.Error, Exception) as error:
            print(f"Error cleaning up the database: {error}")

        print("Done!")


if __name__ == "__main__":
    cleaned_up = RegisterJobs()
    cleaned_up.cleaned_up_db()
