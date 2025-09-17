from dspy.evaluate import SemanticF1
import dspy
import os
import ujson
import random
import json


api_config_path = os.environ.get("api_configs")
with open(api_config_path, "r", encoding="utf-8") as f:
    api_configs = json.load(f)
    dp_api_key = api_configs['deepseek']['api_key']
    e_api_key = api_configs['azure-te3s']['api_key']
    e_endpoint = api_configs['azure-te3s']['base_url']
    e_version = api_configs['azure-te3s']['api_version']

LM = dspy.LM('deepseek/deepseek-chat', api_key=dp_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=LM)

max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query


with open("E:\\program\\agent\\dspy\\ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")



embedder = dspy.Embedder('azure/text-embedding-3-small', dimensions=512, api_key=e_api_key, api_base=e_endpoint, api_version=e_version)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)

class RAG(dspy.Module):
    """in the __init__ method, you declare any sub-module you'll need, which in this case is just a dspy.ChainOfThought('context, question -> response') module that takes retrieved context, a question, and produces a response."""
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    """In the forward method, you simply express any Python control flow you like, possibly using your modules. In this case, we first invoke the search function defined earlier and then invoke the self.respond ChainOfThought module."""
    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
    
rag = RAG()
rag(question="what are high memory and low memory on linux?")

# dspy.inspect_history()


# evaluate the RAG module on the RAG-QA Arena dataset.
with open("E:\\program\\agent\\dspy\\ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]
data = [dspy.Example(**d).with_inputs('question') for d in data]
random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]
metric = SemanticF1(decompositional=True)
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                          display_progress=True, display_table=2)

evaluate(RAG())


# optimize the RAG prompt using dspy
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(RAG(), trainset=trainset,
                           max_bootstrapped_demos=2, max_labeled_demos=2,
                           requires_permission_to_run=False)


# # show the cost. This function cannot be used with dspy 2.6.14
# cost = sum([x['cost'] for x in rag.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers
# cost = sum([x['cost'] for x in optimized_rag.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers
# print(f"Cost of RAG: {cost}")
# print(f"Cost of optimized RAG: {cost}")


# save and load the optimized RAG module
optimized_rag.save("E:\\program\\agent\\dspy\\optimized_rag.json")

# loaded_rag = RAG()
# loaded_rag.load("E:\\program\\agent\\dspy\\optimized_rag.json")

# loaded_rag(question="cmd+tab does not work on hidden or minimized windows")