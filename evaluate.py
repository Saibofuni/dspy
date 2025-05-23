import random
import dspy
from dspy.evaluate import SemanticF1
import os
from dspy.utils import download
import ujson

lm = dspy.LM('deepseek/deepseek-chat', api_key=os.environ.get("deepseek_api"), api_base="https://api.deepseek.com")
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)

cot = dspy.ChainOfThought('question -> response')

# Download the RAG-QA Arena dataset before running the example.
with open("E:\\program\\agent\\dspy\\ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

data = [dspy.Example(**d).with_inputs('question') for d in data]

example = data[2]
print(f"{example}\n\n")

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]


# -------------- use SemanticF1 to evaluate part of performance of the model------------
# Instantiate the metric.
metric = SemanticF1(decompositional=True)
# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())
# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")
# -------------- use SemanticF1 to evaluate part of performance of the model------------


# -------------- use dspy.Evaluate and SemanticF1 to evaluate all the performance of the model------------
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
# ---------------- use dspy.Evaluate and SemanticF1 to evaluate all the performance of the model------------