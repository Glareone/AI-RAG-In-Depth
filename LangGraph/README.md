## LangGraph

List of Content
1. [LangGraph Patterns and their difference. ReACT, Reflection, Multi-Agent, Decision Tree, Plan-Execute](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/LangGraph-Patterns.md)
2. System Prompt Examples  
   a. [Decision Tree System Prompt](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/Prompt-Decision-Tree.md)    
   b. [Multi-Agent System Prompt](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/Prompt-Multi-Agent.md)  
   c. [Plan-Execute](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/Prompt-Plan-Execute.md)  
   d. [ReACT](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/Prompt-ReACT.md)  
   e. [Reflection](https://github.com/Glareone/AI-LLM-RAG-best-practices/blob/main/LangGraph/Prompt-Reflection.md)    
3. [ReACT Implementation Recommendations are here](https://til.simonwillison.net/llms/python-react-pattern)
4. [Prompts Hub. LangGraph Hub](https://smith.langchain.com/hub)

## Practical Examples
1. [LangGraph. ReAct. Tools. AML Anti-money Laundering example](https://github.com/Glareone/AI-RAG-In-Depth/tree/main/LangGraph/LangGraph_ReAct)  
2. [LangGraph. Human in the loop. MCP. Document Analysis example](https://github.com/Glareone/AI-RAG-In-Depth/tree/main/LangGraph/LangGraph_Human_in_the_loop)  

## Monitoring & Traceability examples
1. [Arize Phoenix. Open telemetry](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/LangGraph/LangGraph_ReAct/src/telemetry/setup.py)
2. [Arize Phoenix. AB testing](https://github.com/Glareone/AI-RAG-In-Depth/tree/main/LangGraph/LangGraph_ReAct/ab_testing)  

## LangGraph Patterns
![image](https://github.com/user-attachments/assets/336652b8-71ad-441b-a530-a333fd60a1cd)


### LangGraph usage
```python
from langgraph.graph import StateGraph

builder = StateGraph(StateType)
builder.add_node("react_node", react_function)
builder.add_edge("react_node", decision_node)
builder.set_entry_point("react_node")
graph = builder.compile()
```

### Simple usecase without LangGraph framework
![image](https://github.com/user-attachments/assets/75c79161-8868-4938-ac06-d4b2bf8267c9)

### Difference between pre-coded loop and LangGraph Cycling graph
The difference is not that big as you may expect. It acts very similar:  
![image](https://github.com/user-attachments/assets/5b99ab51-00bd-4a0b-9db3-a19e08cd249c)

Reflecting the previous logic we can just use the following image:  
<img width="667" alt="image" src="https://github.com/user-attachments/assets/1f37bade-665c-4e92-ab77-2d5511f462db" />
* **Annotated**: Annotated means that the value will not be overriden by the LangGraph, instead the last value will be added to the value list.
* State Could be simple or complex, depends on your logic
<img width="650" alt="image" src="https://github.com/user-attachments/assets/dfa1b205-51c0-4d83-805a-72794b233446" />



---
### Pros and Cons of using LangGraph over pre-coded loop
PROS:  
1. Traceability: having quite complex state you can control and understand why it goes this or that way
2. All-in-one: you have one tool to organize the whole flow which fits your needs. You dont need to mix things up.
