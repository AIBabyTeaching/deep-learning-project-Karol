import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Names of celebrities in the LFW dataset (partial list)
class_names = [
    'Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush', 'Gerhard Schroeder',
    'Hugo Chavez', 'Jacques Chirac', 'Jean Chretien', 'John Ashcroft', 'Junichiro Koizumi',
    'Serena Williams', 'Tony Blair'
    # ... Add more names here based on your specific LFW dataset
]

# Load the pre-trained model
model = tf.keras.models.load_model('baseline_model.h5')

st.title('Celebrity Recognition')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to grayscale
    img = image.convert('L')

    # Resize the image
    img_resized = img.resize((94, 125))  # Swap the height and width

    # Convert the image back to a numpy array and expand dimensions
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Now you can pass `img_array` to `model.predict()`
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    celebrity_name = class_names[predicted_class]

    st.write(f"Predicted celebrity: {celebrity_name}")