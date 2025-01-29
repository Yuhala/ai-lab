import ollama
response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': 'Why is sky blue?',
    },
])
print(response['message']['content'])