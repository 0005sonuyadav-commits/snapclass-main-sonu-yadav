import streamlit as st

def footer_home():
    logo_url ="https://avatars.githubusercontent.com/u/89276273?v=4"
    st.markdown(f"""
        <div style="margin-top:2rem;display:flex; gap:6px; justify-content:center; item-align:center">
        <p style='font-weight:bold;color:white'>Created with ❤️ by Sonu Yadav</p>
        <img src='{logo_url}'style='max-height:25px'/>
        </div>
    
    """,unsafe_allow_html=True)


def footer_dashboard():
    logo_url ="https://avatars.githubusercontent.com/u/89276273?v=4"
    st.markdown(f"""
        <div style="margin-top:2rem;display:flex; gap:6px; justify-content:center; item-align:center">
        <p style='font-weight:bold;color:black'>Created with ❤️ by Sonu Yadav</p>
        <img src='{logo_url}'style='max-height:25px'/>
        </div>
    
                """,unsafe_allow_html=True)