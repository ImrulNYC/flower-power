import os
import pandas as pd
import requests
import streamlit as st
import re
from io import StringIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Function to load the dataset from the local data folder
@st.cache_data
def load_dataset_from_local():
    dataset_path = "data/language-of-flowers.csv"
    try:
        data = pd.read_csv(dataset_path, quotechar='"', encoding='utf-8-sig', on_bad_lines='skip')
        data.columns = data.columns.str.strip()  # Clean column names
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the 'language-of-flowers.csv' file is present in the 'data' folder.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error during dataset parsing: {e}")
        return None

# Function to generate flower information
def generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline):
    flower_description = flower_info_dict.get(flower_name, "No description available.")
    query = f"Why is the flower {flower_name} associated with the meaning '{flower_description}'? Explain the cultural or historical significance behind the {flower_name}."
    generated_info = gpt2_pipeline(query, max_length=200, truncation=True)[0]["generated_text"]
    sentences = re.split(r'(?<=\w[.!?])\s+', generated_info.strip())
    limited_output = "".join(sentences[:5])
    return flower_name, flower_description, limited_output

# Function to load flower image
def load_flower_image(flower_name):
    formatted_name = flower_name.replace(' ', '_').lower()
    image_path_with_color = f"data/Flower_images/{formatted_name}.jpg"
    image_path_without_color = f"data/Flower_images/{formatted_name.split('_')[-1]}.jpg"
    image_path_alternative = f"data/Flower_images/{formatted_name.split('_')[-1]}_{formatted_name.split('_')[0]}.jpg"
    if os.path.exists(image_path_with_color):
        return image_path_with_color
    elif os.path.exists(image_path_alternative):
        return image_path_alternative
    elif os.path.exists(image_path_without_color):
        return image_path_without_color
    return None



def developer_info():
    st.markdown(
        """
        <style>
            .dev-title {
                font-size: 2.5em;
                color: #4CAF50;
                text-align: center;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
            .dev-content {
                font-size: 1.2em;
                line-height: 1.6em;
                color: #00695c;
                text-align: center;
                margin-bottom: 2em;
            }
        </style>
        <div class='dev-title'>Developer Information</div>
        <div class='dev-content'>
            <ul>
                <li><strong>Developer 1</strong>: <a href='https://www.linkedin.com/in/developer1' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Developer 2</strong>: <a href='https://www.linkedin.com/in/developer2' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Developer 3</strong>: <a href='https://www.linkedin.com/in/developer3' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Developer 4</strong>: <a href='https://www.linkedin.com/in/developer4' target='_blank'>LinkedIn Profile</a></li>
                
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
def streamlit_app():
    
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 3em;
                color: #4CAF50;
                text-align: center;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
            .sub-title {
                font-size: 1.5em;
                color: #666;
                text-align: center;
                margin-bottom: 2em;
            }
            .info-box {
                background-color: #e0f7fa;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #00acc1;
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 1.5em;
            }
            .info-title {
                font-size: 1.7em;
                color: #004d40;
                margin-bottom: 0.5em;
            }
            .info-content {
                font-size: 1.2em;
                line-height: 1.6em;
                color: #00695c;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main-title'>Flower Power </div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Welcome to the Language of Flowers, Flower Recognition App .</div>", unsafe_allow_html=True)

            
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
        """
        <a href="google.com" target="_blank">
            <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; text-decoration: underline;">
                Pre-trained Flower Recognition
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
    with col2:
        st.markdown(
        """
        <a href="google.com" target="_blank">
            <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; text-decoration: underline;">
                Flower Recognition from Scratch
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    
    data = load_dataset_from_local()
    if data is not None:
        
        data['Flower'] = data['Color'].fillna('') + ' ' + data['Flower']
        flower_info_dict = dict(zip(data['Flower'].str.strip().str.lower(), data['Meaning']))
        meaning_info_dict = dict(zip(data['Meaning'].str.strip().str.lower(), data['Flower']))
        

        # Initialize GPT-2
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

        # User input 
        flower_names = list(flower_info_dict.keys())
        flower_name = st.selectbox("Enter a flower name (e.g., 'Red Rose'):", options=["None"] + sorted(flower_names), index=0, key='flower').strip().lower()
        
        # Displaying information option 2
        if flower_name != "none":
            if flower_name in flower_info_dict:
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline)
                image_path = load_flower_image(flower_name)
                if image_path:
                    st.image(image_path, caption=flower_name.title(), use_container_width=True)
                st.markdown(
                    f"<div class='info-box' style='background-color: #fce4ec; border-color: #f06292;'>"
                    f"<div class='info-title'>Information for {flower_name.title()}:</div>"
                    f"<div class='info-content'><strong>Meaning</strong>: {flower_description}</div>"
                    f"<div class='info-content'><strong>Cultural or Historical Significance</strong>: {generated_info}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>Sorry, we don't have information on the flower: {flower_name.title()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>Sorry, we don't have information on the flower: {flower_name.title()}</div>", unsafe_allow_html=True)

        # User input to get meaning with autocomplete suggestion
        meanings = list(meaning_info_dict.keys())
        meaning = st.selectbox("Enter a meaning to find the flower:", options=["None"] + sorted(meanings), index=0, key='meaning').strip().lower()
        
        # Display information for the selected meaning
        if meaning != "none":
            if meaning in meaning_info_dict:
                matching_flower = meaning_info_dict[meaning]
                image_path = load_flower_image(matching_flower)
                if image_path:
                    st.image(image_path, caption=matching_flower.title(), use_container_width=True)
                st.markdown(
                    f"<div class='info-box' style='background-color: #e8f5e9; border-color: #66bb6a;'>"
                    f"<div class='info-title'>Flower associated with '{meaning.title()}':</div>"
                    f"<div class='info-content'><strong>Flower name: </strong>: {matching_flower.title()}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>Sorry,no flower associated with the meaning: {meaning.title()}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>dataste not loading.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the page:", ["Flower Information", "Developer Info"])
    if app_mode == "Flower Information":
        streamlit_app()
    elif app_mode == "Developer Info":
        developer_info()
