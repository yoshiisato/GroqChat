"""
Welcome to a locally hosted web interface powered by Groq's lightning-fast inference API, 
featuring state-of-the-art open-source models such as MixTRL and LLaMA. Experience the 
speed and efficiency of GroqChat, providing near-instantaneous responses to your queries. 
Give it a try and see the amazing performance firsthand!
"""

import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() # Loading environment variables from .env file

# Retrieving Groq API key from environmental variables, obtained from groqcloud
groq_api_key = os.environ['GroqInferenceProd']

def main():
    """
    This is the main function of the GroqChat application. It sets up the Streamlit interface,
    initializes the conversation chain with Groq's inference API, and handles user interactions.

    The function allows the user to select an LLM model and conversational memory length,
    displays chat history, and processes user input to generate and display responses from the chatbot.
    """
    st.title("GroqChat")
    st.sidebar.title("Select an LLM")

    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )

    conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)
    memory = ConversationBufferMemory(k=conversational_memory_length)

    # Initializing chat history if it doesn't exist in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Displays chat messages from history and saves them to memory for context
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # Using the chat_message container for each message
            st.markdown(message["content"]) # Displays the message content
        # Saves the context to memory for future responses
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initializes ChatGroq object with selected model and API key
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Initializez the ConversationChain with ChatGroq object and memory length
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    user_question = st.chat_input("Ask any question..")

    if user_question:
        response = conversation(user_question) # Obtain response from conversation chain
        st.chat_message("user").markdown(user_question) # Displays the user's question
        # Appends user question to session state messages
        st.session_state.messages.append({"role": "user", "content": user_question, "human": user_question, "AI": response['response']})

        with st.chat_message("assistant"): # Use chat_message container for the AI assistant's response
            st.markdown(response['response']) # Displays the chatbot's response
        # Appends chatbot response to session state messages
        st.session_state.messages.append({"role": "assistant", "content": response['response'], "human": user_question, "AI": response['response']})

if __name__ == "__main__":
    main()
