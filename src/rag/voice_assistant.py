import os
from gtts import gTTS
import speech_recognition as sr
import playsound

# --- MOCK LLM RAG PIPELINE ---
# In a full production app, this would use LangChain + OpenAI/Gemini
# taking the user's Tamil text, translating to English, querying our XGBoost model,
# and translating the answer back to Tamil.

def query_rag_pipeline(tamil_text):
    """
    Mock RAG pipeline. Simulates an LLM querying the models we just built.
    """
    print(f"\n[RAG Pipeline] Received Question: {tamil_text}")
    
    # Simple keyword matching to simulate an LLM deciding what to answer
    if "தக்காளி" in tamil_text or "விலை" in tamil_text:  # "Tomato" or "Price"
        # Mocking an answer based on our XGBoost model
        answer = "தக்காளி விலை அடுத்த ஏழு நாட்களில் கிலோவுக்கு 4 ரூபாய் அதிகரிக்க வாய்ப்புள்ளது. 5 நாட்கள் பொறுத்து விற்கவும்."
    elif "நோய்" in tamil_text or "பயிர்கள்" in tamil_text:  # "Disease" or "Crops"
        # Mocking an answer based on our Random Forest model
        answer = "அதிக ஈரப்பதம் காரணமாக இலை கருகல் நோய் வர வாய்ப்புள்ளது. முன்னெச்சரிக்கை நடவடிக்கைகளை எடுக்கவும்."
    else:
        answer = "மன்னிக்கவும், எனக்கு புரியவில்லை. தக்காளி விலை அல்லது பயிர் நோய்கள் பற்றி கேட்கவும்."
        
    print(f"[RAG Pipeline] Generated Answer: {answer}")
    return answer

# --- VOICE ASSISTANT ---

def listen_to_farmer():
    """
    Uses SpeechRecognition to capture audio from the microphone and 
    convert it to Tamil text using Google's free Speech-to-Text API.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Please speak in Tamil (e.g. 'தக்காளி விலை எப்படி இருக்கும்?')...")
        print("(Listening for 10 seconds. If it doesn't work, we will ask you to type it!)")
        # Lower threshold to ensure it picks up quiet speech
        recognizer.energy_threshold = 300 
        recognizer.dynamic_energy_threshold = True
        
        try:
            # Increased timeout so you have time to start speaking
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing audio...")
            text = recognizer.recognize_google(audio, language="ta-IN")
            print(f"\n🧑‍🌾 Farmer said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            
    # Fallback if mic fails
    print("\n[Microphone Issue Detected]")
    fallback = input("Since the microphone didn't pick you up, please copy-paste or type your Tamil question here: ")
    return fallback

def speak_to_farmer(tamil_text):
    """
    Uses Google Text-to-Speech (gTTS) to convert the Tamil response 
    into an audio file and plays it back to the farmer.
    """
    print("\n🔊 Speaking response...")
    try:
        tts = gTTS(text=tamil_text, lang='ta')
        filename = "response.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)  # Cleanup
    except Exception as e:
        print(f"Error playing audio: {e}")

def run_assistant():
    print("=== Village Farmer Intelligence System (Voice Assistant) ===")
    user_text = listen_to_farmer()
    
    if user_text:
        # Pass the Tamil text to our "LLM"
        response = query_rag_pipeline(user_text)
        
        # Speak the answer back in Tamil
        speak_to_farmer(response)

if __name__ == "__main__":
    run_assistant()
