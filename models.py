import dspy
import os
import json


# azure gpt-4o-mini
api_config_path = os.environ.get("api_configs")
with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    api_key = api_configs['azure-4omini']['api_key']
    mini_endpoint = api_configs['azure-4omini']['base_url']
    mini_version = api_configs['azure-4omini']['api_version']

lm_mini = dspy.LM('azure/gpt-4o-mini', api_key=api_key, api_base=mini_endpoint, api_version=mini_version)
dspy.configure(lm=lm_mini)

# deepseek-V3
api_config_path = os.environ.get("api_configs")
with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    dp_api_key = api_configs['deepseek']['api_key']

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