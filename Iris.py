import streamlit as st

st.set_page_config(
    page_title="Iris Flower Species Predictor",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
)

import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model (1).pkl')

# Dictionary mapping species to image paths (if needed)
# species_images = {
#     'Setosa': 'setosa.png',
#     'Versicolor': 'versicolor.png',
#     'Virginica': 'virginica.png'
# }

# Define the app with a custom title and background color


# Create a sidebar with information about the app
st.sidebar.image('logoiris.png')
st.sidebar.title("About This App")

st.sidebar.write(
    "This app predicts the species of an iris flower based on the input values for sepal length, sepal width, petal length, and petal width. "
    "It uses a machine learning model trained on the Iris dataset. This app gives the prediction of correct species where the flower has its roots."
)

# Main content of the app
def main():
    # Custom title and background color
    st.title('Iris Flower Species Prediction')
    st.markdown(
        """
        <style>
        .big-font {
            font-size: 24px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input fields for user to enter values
    sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
    sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0, 3.5)
    petal_length = st.slider('Petal Length (cm)', 0.0, 10.0, 1.4)
    petal_width = st.slider('Petal Width (cm)', 0.0, 10.0, 0.2)

    if st.button('Predict'):
        # Convert user input to a NumPy array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Use the trained model to make predictions
        prediction = model.predict(input_data)[0]

        # Map numerical prediction to species
        species_mapping = {
            0: 'Setosa',
            1: 'Versicolor',
            2: 'Virginica'
        }

        predicted_species = species_mapping.get(prediction, prediction)

        # Display the predicted species with a slightly larger font size
        st.markdown(
            f'<p class="big-font"><strong>Predicted Species:</strong> {predicted_species}</p>',
            unsafe_allow_html=True,
        )

        # # Display the corresponding image on the right (if needed)
        # if predicted_species in species_images:
        #     st.image(species_images[predicted_species], caption=predicted_species, use_column_width=True)
        # else:
        #     st.warning(f'Image not found for species: {predicted_species}')

if __name__ == '__main__':
    main()
