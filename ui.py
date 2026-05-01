import streamlit as st
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="BEULXSM", page_icon="🚀", layout="wide")

st.title("🚀 BEULXSM Agent")
st.caption("Mode Cepat - Groq + Mistral")

if "agent" not in st.session_state:
    try:
        from agent.core.agent import BeulxsmAgent
        st.session_state.agent = BeulxsmAgent()
        st.success("✅ Agent siap (Groq Priority)")
    except Exception as e:
        st.error(f"❌ Gagal load Agent: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ketikan goal atau perintah..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 BEULXSM sedang berpikir (bisa agak lama di step planning)..."):
            try:
                result = asyncio.run(st.session_state.agent.run(prompt))
                
                response = "**✅ Selesai diproses**\n\n"
                
                if isinstance(result, dict):
                    steps = len(result.get("results", {}))
                    complexity = result.get("plan", {}).get("estimated_complexity", "N/A")
                    response += f"• Step dieksekusi: **{steps}**\n"
                    response += f"• Estimasi kompleksitas: **{complexity}**\n\n"
                    response += "Detail lengkap ada di terminal."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

with st.sidebar:
    st.header("⚡ Status")
    st.write("**Priority Model:** Groq (cepat)")
    st.write("**Fallback:** Mistral")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
