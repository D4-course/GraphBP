import requests
import streamlit as st

st.title("GraphBP")

file = st.file_uploader("Upload the .types file", type="types")

st.write("Example")
st.code('''1 0 0.367752 5HT1B_HUMAN_31_390_0/4iar_A_rec.pdb 5HT1B_HUMAN_31_390_0/4iar_A_rec_4iaq_2gm_lig_tt_docked_0.sdf.gz #-14.46163''')

if st.button("Predict") and (file is not None):
    r = requests.post(url = "http://localhost:8000/predict", files = {'file': file})
    st.download_button(label = "Download", data = r.content, file_name = "prediction.png")
    
if st.button("Previous Inferences"):
    r = requests.post(url = "http://localhost:8000/pretrained", files = {'file': file})
    st.download_button(label = "Download", data = r.content, file_name = "prediction.zip")
