from langchain_community.agent_toolkits.load_tools import load_tools

def websearch_tool(llm):
    searx_ng_url = 'http://localhost:8080/'
    tools = load_tools(["searx-search"], searx_host=searx_ng_url, llm=llm)
    return tools[0]