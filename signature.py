from typing import Literal
import dspy
import os
import pydantic
import json

# attention: all deepseek models do not support mutilmodel input
api_config_path = os.environ.get("api_configs")
with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    dp_api_key = api_configs['deepseek']['api_key']

lm = dspy.LM('deepseek/deepseek-chat', api_key=dp_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=lm)

with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    api_key = api_configs['azure-4omini']['api_key']
    endpoint = api_configs['azure-4omini']['base_url']
    api_version = api_configs['azure-4omini']['api_version']
lm_mini = dspy.LM('azure/gpt-4o-mini', api_key=api_key, api_base=endpoint, api_version="2024-12-01-preview")
# dspy.configure(lm=lm_mini)


# add instructions to the signature
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)


# customize your types
# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float
signature = dspy.Signature("query: str -> result: QueryResult")
class Container:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float
signature = dspy.Signature("query: Container.Query -> score: Container.Score")


# sample classification
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
classify1 = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify2 = dspy.Predict(
    dspy.Signature(
        "sentence -> negative: bool",
        instructions="Mark as 'negative' if the sentence expresses negative emotion.",
    )
)
result1 = classify1(sentence=sentence)
result2 = classify2(sentence=sentence)
print(result1.sentiment)  # => True
print(result2.negative)  # => False


# summerization
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""
summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)
print(response)
print("Reasoning:", response.reasoning)
print(response.summary)


# classification, specifying the input and output fields
class Emotion(dspy.Signature):
    """Classify emotion."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion
classify = dspy.Predict(Emotion)
result = classify(sentence=sentence)
print(result)
print(result.sentiment) # => 'fear'


# A metric that evaluates faithfulness to citations
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")
context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
text = "Lee scored 3 goals for Colchester United."
faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
result = faithfulness(context=context, text=text)
print(result)


# image classification
class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")
image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
result = classify(image_1=dspy.Image.from_url(image_url))
print(result)