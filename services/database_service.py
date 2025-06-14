import cx_Oracle
import json
from datetime import datetime
from typing import Tuple, Dict, Optional

from config import DB_CONFIG, SCHEMA_FILE, LAST_FETCH_FILE

class DatabaseService:
    @staticmethod
    def fetch_schema() -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch database schema from Oracle"""
        if not all([DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["schema"]]):
            return None, "Database configuration incomplete"
        
        try:
            dsn_tns = cx_Oracle.makedsn(
                DB_CONFIG["host"], 
                DB_CONFIG["port"], 
                service_name=DB_CONFIG["service_name"]
            )
            
            connection = cx_Oracle.connect(
                user=DB_CONFIG["user"], 
                password=DB_CONFIG["password"], 
                dsn=dsn_tns
            )
            
            schema_name = DB_CONFIG["schema"].upper()  # fix here

            query = f"""
            SELECT 
                table_name, 
                column_name, 
                data_type, 
                data_length, 
                nullable 
            FROM 
                all_tab_columns 
            WHERE 
                owner = '{schema_name}'
            ORDER BY 
                table_name, column_id
            """
            
            cursor = connection.cursor()
            cursor.execute(query)
            
            tables = {}
            for row in cursor:
                table_name, column_name, data_type, data_length, nullable = row
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append({
                    'column_name': column_name,
                    'data_type': data_type,
                    'data_length': data_length,
                    'nullable': nullable
                })
            
            cursor.close()
            connection.close()
            
            print(f"Tables fetched from Oracle schema '{schema_name}': {list(tables.keys())}")  # debug log

            if not tables:
                return None, f"No tables found in schema '{schema_name}'"
            
            with open(SCHEMA_FILE, 'w') as json_file:
                json.dump(tables, json_file, indent=4)
            
            with open(LAST_FETCH_FILE, 'w') as f:
                f.write(datetime.now().isoformat())
            
            return tables, None
        
        except cx_Oracle.DatabaseError as e:
            return None, f"Database error: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
