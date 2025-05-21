from typing import Literal
import dspy
import os
import pydantic

# attention: all deepseek models do not support mutilmodel input
dp_api_key = os.environ.get("deepseek_api")
lm = dspy.LM('deepseek/deepseek-chat', api_key=dp_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=lm)
mini_api_key = os.environ.get("4omini_api")
mini_endpoint = os.environ.get("4omini_endpoint")
lm_mini = dspy.LM('azure/gpt-4o-mini', api_key=mini_api_key, api_base=mini_endpoint, api_version="2024-12-01-preview")
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