import os
import requests

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="DevPrompt Lab", layout="wide")

st.title("DevPrompt Lab – Model Comparator")

with st.sidebar:
    st.header("API Keys")
    default_groq = os.getenv("GROQ_API_KEY", "")

    groq_key = st.text_input("Groq API Key", value=default_groq, type="password")

    st.subheader("Groq Models")
    default_groq_small = os.getenv("GROQ_SMALL_MODEL", "llama-3.1-8b-instant")
    default_groq_big = os.getenv("GROQ_BIG_MODEL", "llama-3.3-70b-versatile")
    groq_small_model = st.text_input("Small / fast model", value=default_groq_small)
    groq_big_model = st.text_input("Large / high-quality model", value=default_groq_big)

    st.header("Backend")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="Base URL of the FastAPI backend.",
    )

system_instruction = st.text_area(
    "System Instruction",
    value="You are a helpful AI assistant.",
    height=80,
)

prompt = st.text_area(
    "Prompt",
    value="Explain the difference between FastAPI and Streamlit in simple terms.",
    height=180,
)

if st.button("Compare", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt before comparing.")
    elif not groq_key:
        st.warning("Please provide the Groq API key.")
    else:
        with st.spinner("Contacting models..."):
            try:
                resp = requests.post(
                    f"{backend_url.rstrip('/')}/evaluate",
                    json={
                        "prompt": prompt,
                        "system_instruction": system_instruction,
                        "groq_api_key": groq_key,
                        "groq_small_model": groq_small_model,
                        "groq_big_model": groq_big_model,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Groq – small model")
                    st.markdown(f"**Model:** {data['groq_small']['model']}")
                    st.markdown(f"**Latency:** {data['groq_small']['latency_ms']:.1f} ms")
                    st.markdown(
                        f"**Tokens:** input={data['groq_small'].get('input_tokens')}, "
                        f"output={data['groq_small'].get('output_tokens')}, "
                        f"total={data['groq_small'].get('total_tokens')}"
                    )
                    st.markdown("**Response:**")
                    st.write(data["groq_small"]["text"])

                with col2:
                    st.subheader("Groq – big model")
                    st.markdown(f"**Model:** {data['groq_big']['model']}")
                    st.markdown(f"**Latency:** {data['groq_big']['latency_ms']:.1f} ms")
                    st.markdown(
                        f"**Tokens:** input={data['groq_big'].get('input_tokens')}, "
                        f"output={data['groq_big'].get('output_tokens')}, "
                        f"total={data['groq_big'].get('total_tokens')}"
                    )
                    st.markdown("**Response:**")
                    st.write(data["groq_big"]["text"])

            except requests.HTTPError as e:
                try:
                    err = resp.json()
                    detail = err.get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"Backend error: {detail}")
            except Exception as e:
                st.error(f"Request failed: {e}")

