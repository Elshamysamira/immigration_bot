import cohere

# Replace 'your-api-key' with your actual Cohere API key
cohere_api_key = "GVO78KFjnCgLq4jv7jvRd6SnuDTXbUOr03l4rcUo"
co = cohere.Client(cohere_api_key)

def get_prompt(instruction: str) -> str:
    # Define a system-like prompt that provides context
    system = "You are an AI assistant that gives helpful and concise answers."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    return prompt

# Example question
question = "What is a cat?"
prompt = get_prompt(question)

# Call the Cohere API to generate a response
response = co.generate(
    model='command-r-plus-08-2024',  # You can also use other models provided by Cohere
    prompt=prompt,
    max_tokens=50,
    temperature=0.6,
)

# Print the response from the API
print(response.generations[0].text)