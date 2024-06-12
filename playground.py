import streamlit as st
import torch

if torch.cuda.is_available():
      st.text(f"cuda device: {torch.cuda.get_device_name()}")

text_contents = """This is some text"""
st.download_button("Download some text", text_contents)
