import gradio as gr
import openai
from playsound import playsound
from elevenlabslib import *
from pydub import AudioSegment
from pydub.playback import play
import io
import config
import time

openai.api_key = config.OPENAI_API_KEY
api_key = config.ELEVEN_LABS_API_KEY
from elevenlabslib import ElevenLabsUser
user = ElevenLabsUser(api_key)

messages = ["You are an AI executive assistant named ATOM."]

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

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath", placeholder="Please start speaking..."),
    outputs="text",
    title="ATOM my user's executive assistant with a sense of humor!",
    description="Please ask me your question and I will respond both verbally and in text to you...",
)

iface.launch()
