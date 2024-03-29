# GroqChat
Welcome to GroqChat, a locally hosted web interface powered by Groq's lightning-fast inference API, featuring state-of-the-art open-source models such as MixTRL and LLaMA. Experience the speed and efficiency of GroqChat, providing near-instantaneous responses to your queries. Give it a try and see the amazing performance firsthand!

![GroqChat Screenshot](images/groqchat_screenshot.png)

## Features
- Locally hosted web interface for easy access and customization
- Utilizes Groq's inference API for fast and efficient model inference
- Supports open-source models like MixTRL and LLaMA
- Interactive chat interface with conversational memory

## Installation
To set up GroqChat on your local machine, follow these steps:

1. Clone this repository:
`git clone https://github.com/yoshiisato/GroqChat.git`
2. Install the required Python packages:
`pip install -r requirements.txt`
3. Set up your .env file with your Groq API key: (get API key at https://console.groq.com/keys)
`GroqInferenceProd=your_groq_api_key`
4. Run the Streamlit app:
`streamlit run app.py`

## Usage
After starting the Streamlit app, navigate to the provided URL in your web browser. You can interact with GroqChat by selecting a model, adjusting the conversational memory length, and typing your questions into the chat input.

## Credits
- Groq for their inference API
- To the open-sourced Mixtral and Llama models
- Streamlit framework for making webdev a breeze
- Alexmrin for giving me the idea



