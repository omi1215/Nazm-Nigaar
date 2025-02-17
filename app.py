import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from pyht import Client
from pyht.client import TTSOptions, Language, Format
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PLAY_HT_USER_ID = os.getenv('PLAY_HT_USER_ID')
PLAY_HT_API_KEY = os.getenv('PLAY_HT_API_KEY')

# Ensure these variables are set
if not PLAY_HT_USER_ID or not PLAY_HT_API_KEY:
    st.error("Environment variables for API credentials are not set.")
    st.stop()

# Initialize PlayHT client using environment variables
client = Client(
    user_id=PLAY_HT_USER_ID,
    api_key=PLAY_HT_API_KEY,
)

# Load and preprocess data
df = pd.read_csv("Ghazal_ur.csv")
df = df.dropna(subset=['misra1', 'misra2'])

text_entries = []
for _, row in df.iterrows():
    text_entries.extend([row['misra1'], row['misra2']])
urdu_text = '\n'.join(text_entries)

vocab = sorted(set(urdu_text))
char_to_index = {u: i for i, u in enumerate(vocab)}
index_to_char = np.array(vocab)
text_as_int = np.array([char_to_index[c] for c in urdu_text])

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(100, drop_remainder=True)
dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:])).batch(64, drop_remainder=True)

# Define the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(vocab_size)
])

model_2.build(tf.TensorShape([1, None]))
model_2.load_weights('ckpt_30.weights.h5')

# Initialize session state for generated poetry if not already set
if "generated_poetry" not in st.session_state:
    st.session_state.generated_poetry = ""

st.title("Nazm-Nigaar: Urdu Poetry Generation and Voice Conversion")

# Input box for seed text and number of characters to generate
seed_text = st.text_input("Enter a seed text:", "محبت")
num_generate_input = st.text_input("Enter number of characters to generate:", value="100")

# Button to generate poetry
if st.button("Generate Poetry"):
    try:
        num_generate = int(num_generate_input)
    except ValueError:
        st.error("Please enter a valid integer for the number of characters to generate.")
    else:
        # Clean seed text to include only known characters
        cleaned_seed = ''.join([c if c in char_to_index else ' ' for c in seed_text])
        input_eval = [char_to_index[c] for c in cleaned_seed]
        input_eval = tf.expand_dims(input_eval, 0)

        # Reset the state of the RNN
        model_2.layers[1].reset_states()

        text_generated = [cleaned_seed]
        for _ in range(num_generate):
            predictions = model_2.predict(input_eval, verbose=0)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = np.argmax(predictions[-1])
            text_generated.append(index_to_char[predicted_id])
            input_eval = tf.expand_dims([predicted_id], 0)

        # Save generated poetry in session state
        st.session_state.generated_poetry = ''.join(text_generated)


if st.session_state.generated_poetry:
    # Replace common misra separators with newline.
    formatted_poetry = st.session_state.generated_poetry.replace("۔", "۔\n")
    st.markdown("### Generated Poetry:")
    st.markdown(formatted_poetry)

    st.markdown("---")
    st.subheader("Voice Conversion")

    if st.button("Convert to Speech"):
        try:
            speech_text = st.session_state.generated_poetry
            
            urdu_voice_manifest = "s3://voice-cloning-zero-shot/a75064f6-7aeb-4954-9034-193abc799c43/original/manifest.json"
            options = TTSOptions(
                voice=urdu_voice_manifest,
                language=Language.URDU,
                format=Format.FORMAT_WAV,
                sample_rate=24000
            )
            
            audio_file_path = "generated_poetry.wav"
            with open(audio_file_path, "wb") as audio_file:
                for chunk in client.tts(speech_text, options, voice_engine="Play3.0-mini", protocol="http"):
                    audio_file.write(chunk)
            
            st.audio(audio_file_path, format="audio/wav")
            st.success("Audio generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
