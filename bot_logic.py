import json
import os
# ADD THESE TWO LINES AT THE TOP
from dotenv import load_dotenv
load_dotenv()

from google.cloud import texttospeech
from google.oauth2 import service_account
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# --- Explicitly Load Credentials ---
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# This line will no longer fail because the variable is now loaded.
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# --- Initialize API Clients, LLM, and Load Vector Store ---
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

print("Loading FAISS index (Gemini)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
print("FAISS index loaded successfully.")

# ... (The rest of the file is identical) ...

def load_config(twilio_phone_number):
    """Loads a client's configuration based on their phone number."""
    filename = f"config/{twilio_phone_number.replace('+', '_')}.json"
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        with open('config/default.json', 'r') as f:
            return json.load(f)

def get_grounded_ai_response(user_question, conversation_history):
    """Retrieves context and generates a grounded response using the RAG pipeline."""
    print(f"Searching for context related to: '{user_question}'")
    
    retrieved_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    original_system_prompt = conversation_history[0]['content']
    
    rag_prompt = f"""
    {original_system_prompt}

    You MUST answer the user's last question based ONLY on the following context.
    If the answer is not in the context, you MUST say "I do not have that specific information, but I can have someone from our team get back to you with an exact answer."
    NEVER invent or guess at details.

    --- CONTEXT ---
    {context}
    --- END CONTEXT ---
    """

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
    """Converts text to speech using Google TTS."""
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