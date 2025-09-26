import base64
import json
import os

from flask import Flask, request
from flask_sockets import Sockets
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
from dotenv import load_dotenv
from google.cloud import speech
from google.oauth2 import service_account

from bot_logic import load_config, get_grounded_ai_response, text_to_speech

load_dotenv()

app = Flask(__name__)
sockets = Sockets(app)

# --- Load Credentials ---
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and os.path.exists(credentials_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    speech_client = speech.SpeechClient(credentials=credentials)
else:
    speech_client = speech.SpeechClient()

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
conversation_history = {}

@app.route('/incoming_call', methods=['POST'])
def incoming_call():
    """Handles incoming calls and starts the WebSocket stream."""
    to_number = request.form.get('To')
    print(f"Incoming call to: {to_number}")
    
    response = VoiceResponse()
    start = Start()
    stream_url = f'wss://{request.host}/audiostream'
    stream = start.stream(url=stream_url)
    stream.parameter(name='to_number', value=to_number)
    response.append(start)
    
    response.say("Thank you for calling. Please wait while we connect you.", voice='alice')
    response.pause(length=1)

    return str(response), 200, {'Content-Type': 'text/xml'}

@sockets.route('/audiostream')
def audiostream(ws):
    """Handles the real-time audio stream and AI interaction."""
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

    try:
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
                ws.send(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(audio_mulaw).decode('utf-8')}
                }))
                print("Sent audio response to Twilio.")
    except Exception as e:
        print(f"An error occurred in the audio stream: {e}")
    finally:
        print("WebSocket connection closed.")

if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    
    print("Starting server on port 5000...")
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()