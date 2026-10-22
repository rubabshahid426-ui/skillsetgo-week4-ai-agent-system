import ast, operator, re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Week 4 AI Agent", page_icon="🧠")
st.title("🧠 Capstone: AI Agent System")
st.caption("Retrieval + calculator tool + multi-step routing (demo mode)")
DEFAULT = """RAG stands for retrieval-augmented generation. It retrieves source information before answering.
LangGraph helps build stateful workflows from nodes and edges.
LangSmith helps inspect traces, response quality, latency, and failure cases.
Prompt engineering means designing and testing instructions for model outputs.
CNNs are commonly used for image classification."""
kb = st.text_area("Knowledge base", DEFAULT, height=140)
question = st.text_input("Ask a question or enter a calculation", placeholder="What is RAG? or calculate 12 * (5 + 2)")

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
def calc(expr):
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 8: raise ValueError("Exponent too large")
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.operand))
        raise ValueError("Only basic arithmetic is supported")
    return visit(ast.parse(expr, mode="eval"))

if st.button("Run agent", type="primary"):
    if not question.strip(): st.warning("Enter a question first.")
    elif re.search(r"\d", question) and any(x in question for x in ["+", "*", "/", "calculate", "Calculate", " - "]):
        st.subheader("1. Intent routing"); st.write("Calculator tool")
        expr = re.sub(r"^(calculate|Calculate)\s*", "", question).strip()
        try:
            result = calc(expr)
            st.subheader("2. Tool result"); st.code(str(result))
            st.subheader("3. Final response"); st.success(f"The result is {result}.")
        except Exception as e: st.error(f"Could not calculate: {e}")
    else:
        st.subheader("1. Intent routing"); st.write("Knowledge retrieval")
        chunks = [x.strip() for x in re.split(r"(?<=[.!?])\s+", kb) if x.strip()]
        if not chunks: st.warning("Add content to the knowledge base.")
        else:
            vec = TfidfVectorizer(stop_words="english")
            matrix = vec.fit_transform(chunks)
            scores = cosine_similarity(vec.transform([question]), matrix).ravel()
            ids = scores.argsort()[::-1][:3]
            hits = [chunks[i] for i in ids if scores[i] > 0]
            st.subheader("2. Retrieved evidence")
            if hits:
                for hit in hits: st.info(hit)
                st.subheader("3. Final response")
                st.write("Review the retrieved evidence above before composing a final answer. This demo does not call an LLM.")
            else: st.warning("No matching evidence found in the knowledge base.")

st.divider()
st.subheader("Workflow")
st.code("User Input → Intent Router → Calculator OR Knowledge Retrieval → Evidence Review → Response")
