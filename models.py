import dspy
import os


# azure gpt-4o-mini
mini_api_key = os.environ.get("4omini_api")
mini_endpoint = os.environ.get("4omini_endpoint")
lm_mini = dspy.LM('azure/gpt-4o-mini', api_key=mini_api_key, api_base=mini_endpoint, api_version="2024-12-01-preview")
dspy.configure(lm=lm_mini)

# deepseek-V3
dp_api_key = os.environ.get("deepseek_api")
lm = dspy.LM('deepseek/deepseek-chat', api_key=dp_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=lm)


# simple test
# response = lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
# print(response)
# response = lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
# print(response)


# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')
# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response)
print(response.answer)


# len(lm.history)  # e.g., 3 calls to the LM
# lm.history[-1].keys()  # access the last call to the LM, with all metadata