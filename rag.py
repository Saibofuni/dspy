import dspy
import os
import ujson

LM = dspy.LM('deepseek/deepseek-chat', api_key=os.environ.get("deepseek_api"), api_base="https://api.deepseek.com")
dspy.configure(lm=LM)

max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query


with open("E:\\program\\agent\\dspy\\ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

embedder = dspy.Embedder('azure/text-embedding-3-small', dimensions=512, api_key=os.environ.get("t_em_3s_api"), api_base=os.environ.get("t_em_3s_endpoint"), api_version="2024-12-01-preview")
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

dspy.inspect_history()