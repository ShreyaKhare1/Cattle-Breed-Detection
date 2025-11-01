import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.header("Cattle Breed Detection",divider='green')
st.sidebar.title("🐄 Cattle Breed Detection")
st.sidebar.markdown("### Navigation")


# Navigation options
page = st.sidebar.radio(
    "Go to:",
    ("🏠 Home", "📤 Upload Image", "📊 Model Info", "ℹ️ About")
)
model=tf.keras.models.load_model('trained_model.keras')
def predict_model(pil_image):
    
    #image_path=image
    #img=cv2.imread(image_path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#BGR TO RGB
    """
    inp_image: can be a file path (str) or a BytesIO object (from Streamlit uploader)
    """

    #image=tf.keras.preprocessing.image.load_img(inp_image, target_size=(128, 128))
    image = pil_image.resize((128, 128))
    input_array=tf.keras.preprocessing.image.img_to_array(image)
    input_array=np.array([input_array])
    input_array = preprocess_input(input_array)
    prediction=model.predict(input_array)
    print(prediction,prediction.shape)
    result_index=np.argmax(prediction)
    if(result_index==0):
       info = {
    "Breed Name": "Ayrshire",
    "Origin": {
        "Country": "Scotland",
        "Region": "County of Ayr",
        "Developed": "Early 1800s"
    },
    "Purpose": "Dairy production",
    "Physical Characteristics": {
        "Color": "Red and white (patterns vary from light to dark red patches)",
        "Size": "Medium-sized dairy breed",
        "Weight": {
            "Cows": "450–550 kg",
            "Bulls": "700–900 kg"
        },
        "Body Type": "Strong frame, straight back, well-shaped udder",
        "Temperament": "Hardy, active, and adaptable"
    },
    "Milk Production": {
        "Average Yield per Lactation": "4,500–6,000 liters",
        "Fat Content": "Around 4.0%",
        "Protein Content": "Around 3.3%",
        "Milk Quality": "Suitable for drinking, cheese, and butter"
    },
    "Advantages": [
        "Hardy and adaptable to different climates",
        "Efficient grazers, good on pasture-based systems",
        "Strong legs and udder – longer productive life",
        "Fertile and easy calving",
        "Good fat-to-protein ratio in milk"
    ],
    "Points to Note": [
        "Active and alert temperament – handle calmly",
        "Needs balanced nutrition for high milk yield",
        "Protect from extreme heat in tropical areas",
        "Monitor udder health to prevent mastitis"
    ],
    "Farming Practices": {
        "Feed": "Green fodder, dry roughage, and concentrate mixture",
        "Housing": "Clean, dry, and well-ventilated shelter",
        "Breeding": "First calving around 2.5 years; regular heat detection",
        "Health Care": "Routine deworming, vaccination (FMD, HS, BQ), and tick control"
    },
    "Summary": "Ayrshire cattle are hardy, medium-sized dairy cows known for good milk yield, adaptability, and strong grazing ability — ideal for small and medium dairy farmers."
}
    if(result_index==1):
      info = {
    "Breed Name": "Brown Swiss",
    "Origin": {
        "Country": "Switzerland",
        "Region": "Swiss Alps",
        "Developed": "Around 1500s"
    },
    "Purpose": "Dairy production",
    "Physical Characteristics": {
        "Color": "Light brown to dark brown, sometimes with gray or silver tones",
        "Size": "Large-sized dairy breed",
        "Weight": {
            "Cows": "600–700 kg",
            "Bulls": "900–1,000 kg"
        },
        "Body Type": "Strong and muscular frame, straight back, large well-formed udder",
        "Temperament": "Gentle, docile, and hardy"
    },
    "Milk Production": {
        "Average Yield per Lactation": "6,000–8,000 liters",
        "Fat Content": "Around 4.0%",
        "Protein Content": "Around 3.5%",
        "Milk Quality": "High-quality milk, excellent for cheese and butter production"
    },
    "Advantages": [
        "High milk yield with good fat and protein",
        "Very hardy and adaptable to different climates",
        "Calm and easy to handle",
        "Strong legs and udders – long productive life",
        "Resistant to diseases and stress"
    ],
    "Points to Note": [
        "Require proper feeding to reach high production",
        "Best suited for dairy-focused farms",
        "Monitor udder health regularly to prevent mastitis",
        "Do not overfeed; maintain proper body condition"
    ],
    "Farming Practices": {
        "Feed": "Good quality green fodder, silage, hay, and concentrate mixture during lactation",
        "Housing": "Clean, dry, well-ventilated barn or shed",
        "Breeding": "First calving around 2.5 years; regular heat detection",
        "Health Care": "Routine vaccination (FMD, HS, BQ), deworming, and tick control"
    },
    "Summary": "Brown Swiss cattle are large, hardy dairy cows known for high milk yield, excellent milk quality, calm temperament, and long productive life, making them ideal for both small and commercial dairy farms."}
    if(result_index==2):
       info = {
    "Breed Name": "Holstein Friesian",
    "Origin": {
        "Country": "Netherlands and Northern Germany",
        "Region": "Friesland",
        "Developed": "Around 16th–17th century"
    },
    "Purpose": "Dairy production",
    "Physical Characteristics": {
        "Color": "Black and white patches (sometimes red and white)",
        "Size": "Large-sized dairy breed",
        "Weight": {
            "Cows": "580–680 kg",
            "Bulls": "900–1,100 kg"
        },
        "Body Type": "Long, large frame; well-formed udder; straight back",
        "Temperament": "Active, alert, and generally docile"
    },
    "Milk Production": {
        "Average Yield per Lactation": "7,000–10,000 liters (high-producing herds can exceed 12,000 liters)",
        "Fat Content": "Around 3.5%",
        "Protein Content": "Around 3.2%",
        "Milk Quality": "High-volume milk, widely used for fluid milk and dairy products"
    },
    "Advantages": [
        "Extremely high milk yield",
        "Adaptable to various dairy systems (intensive or pasture-based)",
        "Strong selection for milk traits in modern breeding",
        "Generally docile and easy to handle",
        "Well-developed udders for milking efficiency"
    ],
    "Points to Note": [
        "Require high-quality nutrition to maintain production",
        "Sensitive to heat stress; need proper housing and ventilation",
        "Monitor udder health closely to prevent mastitis",
        "Not as hardy as some other breeds; more management-intensive"
    ],
    "Farming Practices": {
        "Feed": "Balanced diet with green fodder, silage, hay, and concentrates for high milk production",
        "Housing": "Well-ventilated, clean barns with comfortable bedding",
        "Breeding": "First calving around 2–2.5 years; regular heat detection",
        "Health Care": "Routine vaccination (FMD, HS, BQ), deworming, and mastitis prevention"
    },
    "Summary": "Holstein Friesian cattle are large, high-producing dairy cows, famous for their exceptional milk volume and efficiency, making them the preferred choice for commercial dairy farms worldwide."
}
    if(result_index==3):
       info = {
    "Breed Name": "Jersey",
    "Origin": {
        "Country": "Jersey, Channel Islands",
        "Region": "Island of Jersey",
        "Developed": "17th–18th century"
    },
    "Purpose": "Dairy production",
    "Physical Characteristics": {
        "Color": "Light fawn to dark brown, sometimes with white patches",
        "Size": "Small to medium-sized dairy breed",
        "Weight": {
            "Cows": "400–500 kg",
            "Bulls": "600–700 kg"
        },
        "Body Type": "Compact frame, well-developed udder, straight back",
        "Temperament": "Gentle, docile, and easy to handle"
    },
    "Milk Production": {
        "Average Yield per Lactation": "4,000–5,500 liters",
        "Fat Content": "Around 4.8–5.0%",
        "Protein Content": "Around 3.8–4.0%",
        "Milk Quality": "Rich, high-fat milk, ideal for butter and cheese production"
    },
    "Advantages": [
        "High butterfat content in milk",
        "Efficient feed-to-milk conversion",
        "Small, manageable size suitable for small farms",
        "Calm and easy to handle",
        "Adaptable to various climates"
    ],
    "Points to Note": [
        "Lower total milk volume compared to Holstein or Brown Swiss",
        "Require good nutrition to maintain milk quality",
        "Monitor udder health to prevent mastitis",
        "Not as hardy in extreme conditions without proper care"
    ],
    "Farming Practices": {
        "Feed": "Quality green fodder, silage, hay, and concentrates during lactation",
        "Housing": "Clean, dry, and well-ventilated shelter",
        "Breeding": "First calving around 2–2.5 years; regular heat detection",
        "Health Care": "Routine vaccination (FMD, HS, BQ), deworming, and tick control"
    },
    "Summary": "Jersey cattle are small, gentle dairy cows known for rich, high-fat milk, excellent feed efficiency, and ease of handling — ideal for small and medium-scale dairy farmers focusing on milk quality rather than volume."
}
    if(result_index==4):
       info = {
    "Breed Name": "Red Dane",
    "Origin": {
        "Country": "Denmark",
        "Region": "Central and Southern Denmark",
        "Developed": "19th century"
    },
    "Purpose": "Dairy production (dual-purpose in some areas)",
    "Physical Characteristics": {
        "Color": "Reddish-brown, uniform coat",
        "Size": "Large-sized dairy breed",
        "Weight": {
            "Cows": "550–650 kg",
            "Bulls": "900–1,100 kg"
        },
        "Body Type": "Strong, muscular frame with a well-shaped udder",
        "Temperament": "Calm, docile, and hardy"
    },
    "Milk Production": {
        "Average Yield per Lactation": "6,000–8,000 liters",
        "Fat Content": "Around 4.2%",
        "Protein Content": "Around 3.5%",
        "Milk Quality": "High-quality milk suitable for cheese and butter production"
    },
    "Advantages": [
        "High milk yield with good fat and protein content",
        "Hardy and well-suited to colder climates",
        "Calm and easy to manage",
        "Strong legs and udder for long productive life",
        "Good fertility and easy calving"
    ],
    "Points to Note": [
        "Require proper feeding to maintain high milk production",
        "Monitor udder health to prevent mastitis",
        "Best suited for dairy-focused farms",
        "Provide shelter in extremely cold or wet conditions"
    ],
    "Farming Practices": {
        "Feed": "Quality green fodder, silage, hay, and concentrates during lactation",
        "Housing": "Clean, dry, well-ventilated barn or shed",
        "Breeding": "First calving around 2.5 years; regular heat detection",
        "Health Care": "Routine vaccination (FMD, HS, BQ), deworming, and tick control"
    },
    "Summary": "Red Dane cattle are large, hardy dairy cows with excellent milk yield and quality, calm temperament, and long productive life, making them suitable for medium to large-scale dairy farms, especially in colder climates."
}
    return result_index,info

@st.fragment()
def filter_and_file():
    cols = st.columns(1)
    
    uploaded_file=cols[0].file_uploader("Upload image")
    # Only run this part if a file is uploaded
    if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption="Uploaded Image", use_container_width=True)#display image
      result,info = predict_model(image)
      #st.write("Prediction result:", result)
      #st.subheader(info['Breed Name'])
      #st.write(info['color'])
      #st.write(info['size'])
      #st.subheader("Advantages:")
      #for advantage in info['Advantages']:
      #  st.markdown(f"- {advantage}")
      display_dict(info)

    else:
      st.info("Please upload an image file.")
def display_dict(d, level=0):
    for key, value in d.items():
        if isinstance(value, dict):
            st.markdown(f"{'  '*level}**{key}:**")
            display_dict(value, level + 1)
        elif isinstance(value, list):
            st.markdown(f"{'  '*level}**{key}:**")
            for item in value:
                st.markdown(f"{'  '*(level+1)}- {item}")
        else:
            st.markdown(f"{'  '*level}**{key}:** {value}")    

#filter_and_file()
if page == "🏠 Home":
    st.title("🏠 Home")
    st.write("Welcome to the Cattle Breed Detection App!")
    st.write('The user can upload cattle images, view predicted breeds and get basic information like milk production, advantages and farming practices.')
    st.info("Use this page to understand what this app does.")

elif page == "📤 Upload Image":
    st.title("📤 Upload Image")
    st.write("Upload a cattle image to identify its breed.")
    st.warning("Ensure images are clear and well-lit for best results.")
    filter_and_file()

elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.write("""
    - Model: MobileNetV2  
    - Accuracy: 85.3%  
    - Dataset: 5 Cattle Breeds, 1000 Images  
    """)
    st.success("Fine-tuned using transfer learning with regularization.")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("""
    This project is designed to classify cattle breeds using deep learning.
    Developed with TensorFlow and Streamlit for academic and practical use.
    """)