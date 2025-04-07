
class SchemaInfo:
    def __init__(self, cursor):
        self.cursor = cursor

    # Function to get schema information dynamically from the database
    def get_schema_info(self):
        # Get all table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        schema_info = "Database Schema:\n"

        for table in tables:
            table_name = '"' + table[0] + '"'
            schema_info += f"\nTable: {table_name}\nColumns:\n"

            # Get columns for each table
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()

            for column in columns:
                column_name = column[1]
                schema_info += f"  - {column_name}\n"

        return schema_info