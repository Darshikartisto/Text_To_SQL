import re
from typing import List,Tuple

class Datasourcedeterminer:
    def __init__(self):
        self.mysql_tables = ['users', 'orders']
        self.csv_files = ['sales_data.csv']
        self.mysql_keywords = ['user', 'order', 'customer', 'product']
        self.csv_keywords = ['sale_id','product','sale_date','revenue','quantity,customer_id']

    def determine_source(self, query: str) -> List[str]:
        query = query.lower()
        sources = []

        for table in self.mysql_tables:
            if table in query:
                sources.append('mysql')
                break

        for file in self.csv_files:
            if file.split('.')[0] in query:
                sources.append('csv')
                break

        if not sources:
            mysql_score = sum(1 for keyword in self.mysql_keywords if keyword in query)
            csv_score = sum(1 for keyword in self.csv_keywords if keyword in query)

            if mysql_score > csv_score:
                sources.append('mysql')
            elif csv_score > mysql_score:
                sources.append('csv')
            else:
                sources = ['mysql', 'csv']

        if re.search(r'\b(join|foreign key)\b', query):
            if 'mysql' not in sources:
                sources.append('mysql')
        
        if re.search(r'\b(total sales|revenue)\b', query):
            if 'csv' not in sources:
                sources.append('csv')

        return sources

    def get_relevant_tables_or_files(self, query: str) -> Tuple[List[str], List[str]]:
        query = query.lower()
        mysql_tables = [table for table in self.mysql_tables if table in query]
        csv_files = [file for file in self.csv_files if file.split('.')[0] in query]
        return mysql_tables, csv_files
        