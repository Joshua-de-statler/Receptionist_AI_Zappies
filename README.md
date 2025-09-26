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
    .venv\Scripts\activate
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


**WHen initializing git for the first time**

```bash
git init
```
```bash
git add .
```
```bash
git commit -m "__commit_name__"
```
```bash
git remote add origin <link>
```
```bash
git branch -M main
```
```bash
git push origin main
```

**When adding files to local git**

```bash
git add .
```
```bash
git commit -m "__commit_name__"
```

**When pusing local changes to github**

```bash
git push origin main
```