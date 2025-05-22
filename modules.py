import dspy
import os

# attention: all deepseek models do not support mutilmodel input
dp_api_key = os.environ.get("deepseek_api")
lm = dspy.LM('deepseek/deepseek-chat', api_key=dp_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)
# mini_api_key = os.environ.get("4omini_api")
# mini_endpoint = os.environ.get("4omini_endpoint")
# lm_mini = dspy.LM('azure/gpt-4o-mini', api_key=mini_api_key, api_base=mini_endpoint, api_version="2024-12-01-preview")
# dspy.configure(lm=lm_mini)


# module Predict
sentence = "it's a charming and often affecting journey."
# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')
# 2) Call with input argument(s). 
response = classify(sentence=sentence)
# 3) Access the output.
print(response.sentiment)


# module ChainOfThought
question = "What's something great about the ColBERT retrieval model?"
# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', temperature=0.7)
# 2) Call with input argument.
response = classify(question=question)
# 3) Access the outputs.
print(response.completions.answer)


# # module ProgramOfThought
# question = "write a python function to calculate the H(x) of three numbers"
# classify = dspy.ProgramOfThought('question -> answer')
# response = classify(question=question)
# print(response.reasoning)
# print(response.answer)


# module MultiChainComparison
question = "What's something great about the ColBERT retrieval model?"
# generate completions
cot = dspy.ChainOfThought('question -> answer', temperature=0.7)
completions = [cot(question=question) for _ in range(3)]
classify = dspy.MultiChainComparison('question -> answer')
response = classify(question=question, completions=completions)
print(response.answer)


# # module ReAct
# question = "What's something great about the ColBERT retrieval model?"
# classify = dspy.ReAct('question -> answer')
# response = classify(question=question)
# print(response.completions.answer)


# module majority



# build RAG
# ATTENTION: the knowledge base should be set before using the RAG module, the example below is just for demonstration
def search(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What's the name of the castle that David Gregory inherited?"
result = rag(context=search(question), question=question)
print(result)
print(result.reasoning)
print(result.response)


# information extraction
text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities_and_metadata)


# Agent
def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)


# multi-hop search(combine multiple models)
def multi_hop_search(query: str, k: int = 3) -> list[str]:
    """
    使用 ColBERTv2 检索服务，根据 query 检索 Wikipedia 摘要，返回最相关的 k 篇文档内容。

    Args:
        query (str): 检索的查询语句。
        k (int): 检索时返回的文档数量（即搜索文档数），默认为3。

    Returns:
        list[str]: 检索到的文档内容列表。
    """
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=k)
    return [x['text'] for x in results]

class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = multi_hop_search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

        return dspy.Prediction(notes=notes, titles=list(set(titles)))
    
search_h = Hop(num_docs=3, num_hops=2)
response = search_h(claim="The ColBERT retrieval model is a great model.")
print(response.titles)


# # track LLM usage, only supported above dspy2.6.16
# # Define a simple program that makes multiple LM calls
# class MyProgram(dspy.Module):
#     def __init__(self):
#         self.predict1 = dspy.ChainOfThought("question -> answer")
#         self.predict2 = dspy.ChainOfThought("question, answer -> score")

#     def __call__(self, question: str) -> str:
#         answer = self.predict1(question=question)
#         score = self.predict2(question=question, answer=answer)
#         return score

# # Run the program and check usage
# program = MyProgram()
# output = program(question="What is the capital of France?")
# print(output.get_lm_usage())