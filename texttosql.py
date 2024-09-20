from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("suriya7/t5-base-text-to-sql")
model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/t5-base-text-to-sql")

# Function to translate English query to SQL
def translate_to_sql_select(english_query):
    input_text = "translate English to SQL: " + english_query
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=100, truncation=True)
    outputs = model.generate(input_ids)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# Example usage
english_query = "Show all employees with salary greater than $50000"
sql_query = translate_to_sql_select(english_query)
print("SQL Query:", sql_query)
