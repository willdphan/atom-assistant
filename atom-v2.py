import gradio as gr
import openai
from playsound import playsound
from elevenlabslib import *
from pydub import AudioSegment
from pydub.playback import play
import io
import config
import time
import speech_recognition as sr

openai.api_key = config.OPENAI_API_KEY
api_key = config.ELEVEN_LABS_API_KEY
from elevenlabslib import ElevenLabsUser
user = ElevenLabsUser(api_key)

messages = ["You are an AI executive assistant named ATOM. Provide responses less than 15 words."]

def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    prompt = messages[-1]
    prompt += f"\nUser: {transcript['text']}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=80,
        n=1,
        stop=None,
        temperature=0.5,
    )

    system_message = response["choices"][0]["text"].replace("Alice:", "").strip()
    messages.append(f"{system_message}")

    voice = user.get_voices_by_name("Antoni")[0]
    audio = voice.generate_audio_bytes(system_message)

    audio = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
    audio.export("output.wav", format="wav")

    playsound("output.wav")

    chat_transcript = "\n".join(messages)
    return chat_transcript

def listen_and_respond():
    timeout_counter = 0
    while True:
        with sr.Microphone() as source:
            r = sr.Recognizer()
            print("Listening...")
            try:
                audio = r.listen(source, timeout=20)
            except sr.WaitTimeoutError:
                timeout_counter += 1
                messages.append("User: [No input]")
                voice = user.get_voices_by_name("Antoni")[0]

                if timeout_counter == 1:
                    message = "Is anyone there?"
                elif timeout_counter == 2:
                    message = "Helloooooooooo?"
                elif timeout_counter == 3:
                    message = "Getting bored here..."
                else:
                    message = "Okay, I'm going to take a nap... Wake me up if you need me."
                    break

                audio = voice.generate_audio_bytes(message)
                audio = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
                audio.export("output.wav", format="wav")
                playsound("output.wav")
                continue

            # Reset the timeout counter if a user speaks
            timeout_counter = 0
                
            with open("audio.wav", "wb") as f:
                print("Generating audio response...")
                f.write(audio.get_wav_data())
                print()
            transcribe("audio.wav")

listen_and_respond()