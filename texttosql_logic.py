from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import mysql.connector



app = Flask(__name__)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("suriya7/t5-base-text-to-sql")
model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/t5-base-text-to-sql")

# MySQL connection details
db_config = {
    "host": "localhost",
    "user": "root",  
    "password": "9528",  
    "database": "salesinfo"  
}

# Function to translate English query to SQL
def translate_to_sql_select(english_query):
    input_text = "translate English to SQL: " + english_query
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=100, truncation=True)
    outputs = model.generate(input_ids)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL query: {sql_query}") 
    return sql_query

# Function to execute SQL query on MySQL database
def execute_query(sql_query):
    try:
        print(f"Generated SQL query: {sql_query}") 
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)  # Fetch results as dictionaries
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    


# Define the /query endpoint
@app.route('/query', methods=['POST'])
def handle_query():
    # Get the natural language query from the request
    data = request.get_json()
    english_query = data.get("query")

    if not english_query:
        return jsonify({"error": "No query provided"}), 400

    # Translate the natural language query to SQL
    sql_query = translate_to_sql_select(english_query)

    # Execute the SQL query and get results
    results = execute_query(sql_query)

    if results is None:
        return jsonify({"error": "Failed to execute SQL query"}), 500

    # Return the results as a JSON response
    return jsonify({"sql_query": sql_query, "results": results})

if __name__ == '__main__':
    app.run(debug=True)
