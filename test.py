import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16)

text = """CREATE TABLE work_orders (
    ID NUMBER,
    CREATED_AT TEXT,
    COST FLOAT,
    INVOICE_AMOUNT FLOAT,
    IS_DUE BOOLEAN,
    IS_OPEN BOOLEAN,
    IS_OVERDUE BOOLEAN,
    
    COUNTRY_NAME TEXT,
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- how many work orders are open?

SELECT"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=500)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
