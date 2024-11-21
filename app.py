import gradio as gr
from agent import agent
from langchain.schema import AIMessage, HumanMessage
import json

def answer(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    response = agent.invoke({"messages": history_langchain_format, "verbose": True})
    print(json.dumps([m.content for m in response['messages']], indent=2))
    return response['messages'][-1].content

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(answer, type="messages")
    
if __name__ == "__main__":
    demo.launch(debug=True)