import os

# A dictionary representing the project structure: {path: content}
project_structure = {
    "config/default.json": """
{
  "system_prompt": "You are a helpful, generic AI assistant. Please ask the caller which company they are trying to reach.",
  "forwarding_number": "+15559999999",
  "voice_name": "en-US-Wavenet-F"
}
""",
    "config/_15551234567.json": """
{
  "system_prompt": "You are a flawless, voice-activated AI receptionist for a company named 'Innovate Tech'. Your primary tasks are to answer incoming calls, schedule appointments, and forward calls to the main office. Start the conversation by introducing yourself.",
  "forwarding_number": "+15551234567",
  "voice_name": "en-US-Wavenet-D"
}
""",
    "knowledge_base/faq.txt": """
Q: What is the typical timeframe to build a custom home?
A: A typical custom home build takes between 10 to 16 months from groundbreaking to completion, depending on the complexity and size of the project.

Q: Do you help with purchasing land?
A: While we do not broker land deals, we partner with a number of trusted local real estate agents and can provide recommendations to help you find the perfect lot for your new home.

Q: What is your price per square foot?
A: We do not use a fixed price per square foot, as every custom home is unique. The final cost depends on the architectural complexity, interior finishes, and specific features chosen by the client. We provide a detailed, itemized budget after the initial design consultation.
""",
    "knowledge_base/warranty.txt": """
Our company provides a 10-year structural warranty on all new custom homes. This covers the foundation and frame of the home. A separate 2-year warranty covers all plumbing and electrical systems. Appliances are covered by their respective manufacturer's warranties. The warranty does not cover cosmetic issues like paint or drywall cracks after the first 90 days.
""",
    "requirements.txt": """
twilio
flask
Flask-Sockets
gevent
gevent-websocket
python-dotenv
google-cloud-speech
google-cloud-texttospeech
langchain
langchain-google-genai
faiss-cpu
tiktoken
""",
    ".env.example": """
# Twilio Credentials
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token

# Google AI Credentials (from Google Cloud Console -> APIs & Services -> Credentials)
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Cloud Credentials (path to your service account JSON file)
GOOGLE_APPLICATION_CREDENTIALS=gcp-credentials.json
""",
    ".gitignore": """
# Virtual Environment
.venv/
venv/

# Environment variables
.env

# Credentials
*.json

# FAISS Index
faiss_index/

# Python cache
__pycache__/
*.pyc
""",
    "create_vectorstore.py": """
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Verify that the necessary environment variables are set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
     raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

# 1. Load documents from the knowledge_base directory
loader = DirectoryLoader('knowledge_base/', glob="**/*.txt")
docs = loader.load()
if not docs:
    raise ValueError("No documents found in 'knowledge_base' directory. Please add your .txt files.")
print(f"Loaded {len(docs)} documents.")

# 2. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks.")

# 3. Create Google Gemini embeddings
print("Creating Gemini embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. Create a FAISS vector store from the chunks and save it locally
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local("faiss_index")

print("Vector store created and saved locally in 'faiss_index' folder.")
""",
    "bot_logic.py": """
import json
import os
from google.cloud import texttospeech
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# --- Initialize API Clients, LLM, and Load Vector Store ---
tts_client = texttospeech.TextToSpeechClient()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

print("Loading FAISS index (Gemini)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
print("FAISS index loaded successfully.")

# --- Helper Functions ---
def load_config(twilio_phone_number):
    \"\"\"Loads a client's configuration based on their phone number.\"\"\"
    filename = f"config/{twilio_phone_number.replace('+', '_')}.json"
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        with open('config/default.json', 'r') as f:
            return json.load(f)

def get_grounded_ai_response(user_question, conversation_history):
    \"\"\"Retrieves context and generates a grounded response using the RAG pipeline.\"\"\"
    print(f"Searching for context related to: '{user_question}'")
    
    retrieved_docs = retriever.invoke(user_question)
    context = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
    
    original_system_prompt = conversation_history[0]['content']
    
    rag_prompt = f\"\"\"
    {original_system_prompt}

    You MUST answer the user's last question based ONLY on the following context.
    If the answer is not in the context, you MUST say "I do not have that specific information, but I can have someone from our team get back to you with an exact answer."
    NEVER invent or guess at details.

    --- CONTEXT ---
    {context}
    --- END CONTEXT ---
    \"\"\"

    messages = [SystemMessage(content=rag_prompt)]
    for message in conversation_history[1:]:
        if message['role'] == 'user':
            messages.append(HumanMessage(content=message['content']))
    messages.append(HumanMessage(content=user_question))

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "I am having trouble connecting. Please call back later."

def text_to_speech(text, voice_name="en-US-Wavenet-F"):
    \"\"\"Converts text to speech using Google TTS.\"\"\"
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MULAW,
        sample_rate_hertz=8000
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content
""",
    "app.py": """
import base64
import json
import os

from flask import Flask, request
from flask_sockets import Sockets
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
from dotenv import load_dotenv
from google.cloud import speech

from bot_logic import load_config, get_grounded_ai_response, text_to_speech

load_dotenv()

app = Flask(__name__)
sockets = Sockets(app)

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
speech_client = speech.SpeechClient()

conversation_history = {}

@app.route('/incoming_call', methods=['POST'])
def incoming_call():
    \"\"\"Handles incoming calls and starts the WebSocket stream.\"\"\"
    to_number = request.form.get('To')
    print(f"Incoming call to: {to_number}")
    
    response = VoiceResponse()
    start = Start()
    stream = start.stream(url=f'wss://{request.host}/audiostream')
    stream.parameter(name='to_number', value=to_number)
    response.append(start)
    
    response.say("Thank you for calling. Please wait while we connect you.", voice='alice')
    response.pause(length=5)

    return str(response), 200, {'Content-Type': 'text/xml'}

@sockets.route('/audiostream')
def audiostream(ws):
    \"\"\"Handles the real-time audio stream and AI interaction.\"\"\"
    print("WebSocket connection established.")
    call_sid, stream_sid, client_config = None, None, None

    def audio_generator():
        nonlocal call_sid, stream_sid, client_config
        while not ws.closed:
            message = ws.receive()
            if message is None: continue
            data = json.loads(message)

            if data['event'] == 'start':
                call_sid = data['start']['callSid']
                stream_sid = data['start']['streamSid']
                to_number = data['start']['customParameters']['to_number']
                client_config = load_config(to_number)
                
                print(f"Streaming started for call SID: {call_sid}")
                conversation_history[call_sid] = [{"role": "system", "content": client_config['system_prompt']}]

            elif data['event'] == 'media':
                yield base64.b64decode(data['media']['payload'])
            
            elif data['event'] == 'stop':
                print("Streaming stopped.")
                if call_sid and call_sid in conversation_history:
                    del conversation_history[call_sid]
                break

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(config=recognition_config, interim_results=False)

    responses = speech_client.streaming_recognize(
        config=streaming_config,
        requests=(speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator())
    )

    for response in responses:
        if not response.results or not response.results[0].alternatives: continue

        transcript = response.results[0].alternatives[0].transcript.strip()
        if not transcript: continue
        print(f"Transcript: {transcript}")

        if call_sid:
            ai_response_text = get_grounded_ai_response(transcript, conversation_history[call_sid])
            print(f"AI Response: {ai_response_text}")

            conversation_history[call_sid].append({"role": "user", "content": transcript})
            conversation_history[call_sid].append({"role": "assistant", "content": ai_response_text})
        
            if "forwarding your call" in ai_response_text.lower():
                print(f"AI forwarding call to {client_config['forwarding_number']}")
                forward_response = VoiceResponse()
                forward_response.dial(client_config['forwarding_number'])
                twilio_client.calls(call_sid).update(twiml=str(forward_response))
                ws.close()
                break

            audio_mulaw = text_to_speech(ai_response_text, client_config['voice_name'])
            
            ws.send(json.dumps({"event": "clear", "streamSid": stream_sid}))
            ws.send(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(audio_mulaw).decode('utf-8')}
            }))
            print("Sent audio response to Twilio.")

if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    
    print("Starting server on port 5000...")
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
""",
    "README.md": """
# AI Voice Receptionist

This project is a sophisticated, real-time AI voice receptionist that can answer phone calls, provide factual information from a knowledge base, and perform actions like call forwarding. It's built with Twilio for telephony, Google Gemini for conversational AI, and a RAG (Retrieval-Augmented Generation) pipeline to ensure factual, hallucination-free responses.

## Features

-   **Real-Time Conversation**: Engages in low-latency, back-and-forth conversation over the phone.
-   **RAG Pipeline**: Answers questions based on a provided knowledge base to prevent AI hallucinations and ensure accuracy.
-   **Dynamic Configuration**: Easily configurable for different clients (multi-tenant) with unique AI personas, voices, and business logic using simple JSON files.
-   **Call Forwarding**: Can intelligently forward calls to a human based on conversational triggers.

## Technology Stack

-   **Telephony**: Twilio Voice & Media Streams
-   **Backend Server**: Python with Flask & Flask-Sockets
-   **Conversational AI**: Google Gemini Pro via `langchain-google-genai`
-   **Speech-to-Text**: Google Cloud Speech-to-Text
-   **Text-to-Speech**: Google Cloud Text-to-Speech
-   **RAG/Vector Search**: LangChain with FAISS

## Setup and Installation

1.  **Clone the Repository & Navigate to Directory**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and Activate Virtual Environment**
    ```bash
    # Create
    python -m venv .venv
    # Activate (Windows)
    .venv\\Scripts\\activate
    # Activate (macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Credentials**
    -   Obtain a **Twilio Account SID & Auth Token**.
    -   From the **Google Cloud Console**, create a project, enable billing, and:
        -   Create a project-linked **API Key** (for Gemini).
        -   Create a **Service Account** and download the **JSON key file**. Rename it to `gcp-credentials.json` and place it in the project root.
    -   Create a `.env` file from `.env.example` and fill in all the credentials. Do not use quotes.

5.  **Create Knowledge Base**
    -   Create a folder named `knowledge_base`.
    -   Add your factual information in `.txt` files inside this folder.

6.  **Generate Vector Index**
    -   Run the indexing script once. This reads your knowledge base and prepares it for the AI.
    ```bash
    python create_vectorstore.py
    ```

## Running the Application

1.  **Start the Main Server**
    -   In your first terminal, run the Flask app.
    ```bash
    python app.py
    ```

2.  **Start the `ngrok` Tunnel**
    -   In a second terminal, expose your local server to the internet.
    ```bash
    ngrok http 5000
    ```

3.  **Configure Twilio**
    -   Copy the `https://...` forwarding URL from `ngrok`.
    -   Go to your Twilio Phone Number's settings and paste the URL into the "A CALL COMES IN" webhook field, adding `/incoming_call` to the end. Ensure the method is `HTTP POST`.

4.  **Test**
    -   Call your Twilio phone number and start talking to your AI receptionist!

## Configuration

-   **Secrets**: All API keys and secret tokens are managed in the `.env` file.
-   **Client Personas**: To add or change a client's AI persona, edit the JSON files in the `config/` directory. The application identifies the client based on the Twilio phone number they called and loads the corresponding `.json` file (e.g., a call to `+15551234567` will load `config/_15551234567.json`).
"""
}

def create_project(structure):
    """Creates the project structure based on the dictionary provided."""
    for path, content in structure.items():
        # Ensure the directory for the file exists
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            print(f"Created directory: {dir_name}")
        
        # Write the file content
        with open(path, 'w', encoding='utf-8') as f:
            # .strip() removes leading/trailing whitespace from the multiline strings
            f.write(content.strip())
        print(f"Created file:      {path}")

if __name__ == "__main__":
    print("Starting to create project structure...")
    create_project(project_structure)
    print("\nProject structure created successfully!")
    print("Next steps:")
    print("1. Create a virtual environment and activate it.")
    print("2. Run 'pip install -r requirements.txt'.")
    print("3. Create a '.env' file from '.env.example' and add your credentials.")