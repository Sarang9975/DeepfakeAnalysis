import streamlit as st
import requests
import pickle
import os
from bs4 import BeautifulSoup

# Add 'src' folder to Python path to fix module not found error
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cleaning import process_text  # Import the cleaning function from the src folder
from prediction import get_predictions  # Import the prediction function from the src folder

# Load the model and feature transformer using pickle
model_path = "models/lr_final_model.pkl"
transformer_path = "models/transformer.pkl"

loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))

def scrape_text_from_url(url):
    """
    Scrape text content from the given URL.
    Args:
        url (str): The URL of the web page to scrape.
    Returns:
        str: The scraped text content or an error message.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from <p> tags
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        return f"Error scraping the URL: {e}"

def main():
    st.title("DeepFake Detection Using URL")
    
    # Input URL for analysis
    url = st.text_input("Enter URL to analyze:")

    if st.button("Analyze"):
        if url:
            st.write("Scraping the URL and analyzing...")

            # Step 1: Scrape text content from the URL
            scraped_text = scrape_text_from_url(url)

            # Debug: Preview scraped text
            st.write("Scraped Text Preview:", scraped_text[:500])
            
            # If scraping fails, show error
            if "Error scraping the URL" in scraped_text or len(scraped_text.strip()) < 50:
                st.error("Error: Insufficient content scraped from the URL. Please try a different URL.")
            else:
                # Step 2: Clean the scraped text
                clean_data = process_text(scraped_text)

                # Debug: Preview cleaned text
                st.write("Cleaned Text Preview:", " ".join(clean_data[:50]))

                # Join the cleaned list of words into a single string
                cleaned_text = " ".join(clean_data)

                # Step 3: Transform the text using the model's transformer
                test_features = loaded_transformer.transform([cleaned_text])

                # Debug: Check feature transformation
                st.write("Feature Transformation Shape:", test_features.shape)

                # Step 4: Make prediction
                my_prediction, probabilities = get_predictions(loaded_model, test_features)

                # Debug: Display probabilities and prediction
                st.write("Probabilities:", probabilities)
                st.write("Prediction:", my_prediction)

                # Confidence scores for "real" and "fake"
                real_confidence = float(probabilities[0][0]) * 100
                fake_confidence = float(probabilities[0][1]) * 100

                # Display the prediction result
                if my_prediction[0] == "real":
                    # Apply green background for "Prediction: Real"
                    st.markdown(
                        f"""
                        <style>
                            .real-bg {{
                                background-color: #d4edda;
                                color: #155724;
                                padding: 15px;
                                border-radius: 10px;
                                text-align: center;
                                font-size: 20px;
                                font-weight: bold;
                            }}
                        </style>
                        <div class="real-bg">Prediction: Fake! (Confidence: {real_confidence:.2f}%)</div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"Prediction: Real (Confidence: {fake_confidence:.2f}%)")

        else:
            st.warning("Please enter a valid URL.")

if __name__ == '__main__':
    main()
