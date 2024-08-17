import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

# Add custom CSS
st.markdown("""
    <style>
    body {
        background-color: #F5F5DC;  /* Beige background for the whole page */
        color: #000000;            /* Black text color */
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: #D3D3D3;  /* Light grey background for buttons */
        color: #000000;            /* Black text color for buttons */
        border: 1px solid #000000;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: #C0C0C0;  /* Slightly darker grey background on hover */
        color: #000000;            /* Black text color on hover */
    }
    .stTextInput > div > div > input {
        color: #000000;            /* Black text color for input fields */
        background-color: #FFFFFF;  /* White background for input fields */
        border: 1px solid #000000;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: #000000;            /* Black text color for labels */
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader {
        color: #000000;            /* Black text color for headings */
    }
    .stApp {
        background-color: #F5F5DC;  /* Beige background for the app container */
    }
    .stChatMessage {
        background-color: #D3D3D3;  /* Light grey background for chat messages */
        color: #000000;            /* Black text color for chat messages */
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .stSuccess {
        color: #4CAF50;            /* Green color for success messages */
    }
    .stWarning {
        color: #FFC107;            /* Yellow color for warning messages */
    }
    .response-box {
        background-color: #F5F5DC;  /* Beige background for the response */
        color: #000000;            /* Black text color for the response */
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border: 1px solid #CCCCCC;
    }
    .intermediate-step {
        background-color: #F5F5DC;  /* Beige background for intermediate steps */
        color: #000000;            /* Black text color for intermediate steps */
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border: 1px solid #CCCCCC;
    }
    /* Sidebar styling */
    .css-1n7v3ny {
        background-color: #8B4513;  /* Brown background for the sidebar */
    }
    </style>
    """, unsafe_allow_html=True)

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned"
)

# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only input mathematical expressions need to be provided"
)

prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation
and display it pointwise for the question below:
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Start the interaction
question = st.text_area("Enter your question:", "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            
            st.write('### Response:')
            st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

            # Intermediate steps (assuming you have steps to display)
            # This is a placeholder for displaying intermediate steps
            intermediate_steps = "Here are the intermediate steps to solve the problem..."  # Replace with actual steps if available
            st.markdown(f'<div class="intermediate-step">{intermediate_steps}</div>', unsafe_allow_html=True)

    else:
        st.warning("Please enter a question")
