import random
import dspy
from dspy.evaluate import SemanticF1
import os
from dspy.utils import download
import ujson
import json



api_config_path = os.environ.get("api_configs")
with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    api_key = api_configs['deepseek']['api_key']

lm = dspy.LM('deepseek/deepseek-chat', api_key=api_key, api_base="https://api.deepseek.com")
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

# Instantiate the metric.
metric = SemanticF1(decompositional=True)


# -------------- use SemanticF1 to evaluate part of performance of the model------------
# # Produce a prediction from our `cot` module, using the `example` above as input.
# pred = cot(**example.inputs())
# # Compute the metric score for the prediction.
# score = metric(example, pred)

# print(f"Question: \t {example.question}\n")
# print(f"Gold Response: \t {example.response}\n")
# print(f"Predicted Response: \t {pred.response}\n")
# print(f"Semantic F1 Score: {score:.2f}")
# -------------- use SemanticF1 to evaluate part of performance of the model------------


# -------------- use dspy.Evaluate and SemanticF1 to evaluate all the performance of the model------------
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
# ---------------- use dspy.Evaluate and SemanticF1 to evaluate all the performance of the model------------


# -------------- use MIPROv2 to optimize the Chain-of-Thought program (COSTLY!!!)------------
# optimize the RAG prompt using dspy
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(cot, trainset=trainset,
                           max_bootstrapped_demos=2, max_labeled_demos=2,
                           requires_permission_to_run=False)

# save and load the optimized RAG module
optimized_rag.save("E:\\program\\agent\\dspy\\optimized_cot.json")

loaded_rag = cot
loaded_rag.load("E:\\program\\agent\\dspy\\optimized_cot.json")
evaluate(loaded_rag)