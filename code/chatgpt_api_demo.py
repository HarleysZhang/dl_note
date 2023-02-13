import openai

# After set the API key, the code can run!
openai.api_key = "YOUR_API_KEY"

# Define the model and prompt
model_engine = "text-davinci-003"
prompt = "如何使用openai 官网的 chatgpt 服务，给出详细步骤"

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Get the response text
message = completion.choices[0].text

print(message)