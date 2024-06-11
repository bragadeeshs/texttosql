import openai
from openai import AzureOpenAI
from flask import Flask, request, render_template, jsonify

client = AzureOpenAI(
    api_key='<API KEY>',
    api_version="<API VERSION>",
    azure_endpoint='<END POINT>',
)

app = Flask(__name__)

# Example exchange rate: 1 USD = 75 INR
USD_TO_INR = 82

def count_tokens(prompt, response_text):
    return len(prompt.split()) + len(response_text.split())

def calculate_cost(tokens_used):
    # Example pricing: $0.002 per 1,000 tokens
    cost_per_1000_tokens = 0.002
    cost_in_usd = (tokens_used / 1000) * cost_per_1000_tokens
    cost_in_inr = cost_in_usd * USD_TO_INR
    return cost_in_usd, cost_in_inr

def extract_ner(sentence):
    prompt = f"Extract the named entities from the following sentence and categorize them:\n\nSentence: \"{sentence}\"\n\nEntities:"
    response = client.chat.completions.create(
        model="gpt-35-turbo",  # Specify the deployment name you have configured in Azure
        messages=[
            {"role": "system", "content": "You are an assistant that helps to extract named entities."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_text = response.choices[0].message.content.strip()
    tokens_used = count_tokens(prompt, response_text)
    cost_usd, cost_inr = calculate_cost(tokens_used)
    print(f"Tokens used: {tokens_used}, Cost: ${cost_usd:.4f}, Cost in INR: ₹{cost_inr:.2f}")
    return response_text, tokens_used, cost_usd, cost_inr

def map_entities_to_columns(entities, schema):
    entity_column_map = {}
    for entity in entities.split("\n"):
        entity = entity.strip()
        if entity:
            entity_name, entity_type = entity.split(':')
            entity_name = entity_name.strip()
            entity_type = entity_type.replace('entity type:', '').replace(')', '').strip()
            for column in schema:
                if entity_type.lower() in column['type'].lower():
                    entity_column_map[entity_name] = column['name']
                    break
    return entity_column_map

def generate_sql_query_and_extract_entities(natural_language_query, schema):
    # Construct the prompt to perform both NER and SQL generation
    prompt = (
        f"Extract the named entities and generate a SQL query for the following natural language query"
        f" using the provided schema:\n\n"
        f"Query: \"{natural_language_query}\"\n\n"
        f"Schema: {schema}\n\n"
        f"Entities and SQL Query:"
    )

    # Call Azure OpenAI to perform the task
    response = client.chat.completions.create(
        model="gpt-35-turbo",  # Specify the deployment name you have configured in Azure
        messages=[
            {"role": "system",
             "content": "You are an assistant that helps to extract named entities and generate SQL queries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_text = response.choices[0].message.content.strip()
    tokens_used = count_tokens(prompt, response_text)
    cost_usd, cost_inr = calculate_cost(tokens_used)
    print(f"Tokens used: {tokens_used}, Cost: ${cost_usd:.4f}, Cost in INR: ₹{cost_inr:.2f}")

    entities, sql_query = response_text.split("SQL Query:")
    entities = entities.replace("Entities:", "").strip()
    sql_query = sql_query.strip()
    return entities, sql_query, tokens_used, cost_usd, cost_inr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.json
    sentence = data['sentence']
    schema = [
        {"name": "customer_id", "type": "int"},
        {"name": "name", "type": "text"},
        {"name": "total_spent", "type": "money"},
        {"name": "last_purchase_date", "type": "date"}
    ]

    entities, sql_query, tokens_used, cost_usd, cost_inr = generate_sql_query_and_extract_entities(sentence, schema)

    return jsonify({
        'entities': entities,
        'sql_query': sql_query,
        'tokens_used': tokens_used,
        'cost_usd': cost_usd,
        'cost_inr': cost_inr
    })

if __name__ == '__main__':
    app.run(debug=True)
