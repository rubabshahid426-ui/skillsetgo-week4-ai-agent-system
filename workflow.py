from typing import TypedDict
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

class State(TypedDict, total=False):
    question: str
    intent: str
    context: str
    answer: str

def classify(state):
    q = state.get("question", "").lower()
    return {"intent": "summarization" if "summarize" in q or "summary" in q else "question_answering"}

def retrieve(state):
    return {"context": "RAG retrieves relevant source information. LangGraph connects stateful workflow nodes. LangSmith helps inspect traces and failures."}

def respond(state):
    return {"answer": f"Intent: {state.get('intent')}\nQuestion: {state.get('question')}\nDemo context: {state.get('context')}\nVerify source evidence before submitting an answer."}

def run(question):
    initial = {"question": question}
    if StateGraph is None:
        state = dict(initial); state.update(classify(state)); state.update(retrieve(state)); state.update(respond(state))
        state["note"] = "LangGraph not installed; steps ran sequentially."
        return state
    graph = StateGraph(State)
    graph.add_node("classify", classify); graph.add_node("retrieve", retrieve); graph.add_node("respond", respond)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve"); graph.add_edge("retrieve", "respond"); graph.add_edge("respond", END)
    return graph.compile().invoke(initial)

if __name__ == "__main__":
    q = input("Enter a question: ").strip() or "What is RAG?"
    result = run(q)
    print("\nWorkflow output")
    for key, value in result.items(): print(f"{key}: {value}")
