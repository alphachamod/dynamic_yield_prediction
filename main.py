import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

## Load your tomato disease recognition model
try:
    tomato_model = load_model('model/1')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    tomato_model = None  # Set to None if loading fails

# Print the model summary if model is loaded
if tomato_model:
    tomato_model.summary()


disease_yield_loss = {
    'tom_bacterial_spot': {'loss': 75, 'type': 'Bacterial'},  # Bacterial spot
    'tom_early_blight': {'loss': 80, 'type': 'Fungal'},  # Early blight
    'tom_healthy': {'loss': 0, 'type': 'Healthy'},  # Healthy plant
    'tom_late_blight': {'loss': 90, 'type': 'Fungal'},  # Late blight
    'tom_leaf_miner': {'loss': 40, 'type': 'Pest'},  # Leaf miner
    'tom_leaf_mold': {'loss': 50, 'type': 'Fungal'},  # Leaf mold
    'tom_magnesium_deficiency': {'loss': 30, 'type': 'Deficiency'},  # Magnesium deficiency
    'tom_mosaic_virus': {'loss': 70, 'type': 'Viral'},  # Mosaic virus
    'tom_nitrogen_deficiency': {'loss': 35, 'type': 'Deficiency'},  # Nitrogen deficiency
    'tom_potassium_deficiency': {'loss': 40, 'type': 'Deficiency'},  # Potassium deficiency
    'tom_septoria_leaf_spot': {'loss': 50, 'type': 'Fungal'},  # Septoria leaf spot
    'tom_spotted_wilt_virus': {'loss': 95, 'type': 'Viral'},  # Spotted wilt virus
    'tom_target_spot': {'loss': 60, 'type': 'Fungal'},  # Target spot
    'tom_two-spotted_spider_mite': {'loss': 45, 'type': 'Pest'},  # Two-spotted spider mite
    'tom_yellow_leaf_curl_virus': {'loss': 100, 'type': 'Viral'},  # Yellow leaf curl virus
}

optimal_conditions = {
    'tom_bacterial_spot': {'temp': 24, 'humidity': 80},  # Bacterial spot
    'tom_early_blight': {'temp': 25, 'humidity': 95},  # Early blight
    'tom_late_blight': {'temp': 20, 'humidity': 90},  # Late blight
    'tom_leaf_miner': {'temp': 25, 'humidity': 70},  # Leaf miner
    'tom_leaf_mold': {'temp': 22, 'humidity': 85},  # Leaf mold
    'tom_mosaic_virus': {'temp': 28, 'humidity': 70},  # Mosaic virus
    'tom_septoria_leaf_spot': {'temp': 20, 'humidity': 80},  # Septoria leaf spot
    'tom_spotted_wilt_virus': {'temp': 28, 'humidity': 70},  # Spotted wilt virus
    'tom_target_spot': {'temp': 25, 'humidity': 80},  # Target spot
    'tom_two-spotted_spider_mite': {'temp': 30, 'humidity': 50},  # Two-spotted spider mite
    'tom_yellow_leaf_curl_virus': {'temp': 28, 'humidity': 70},  # Yellow leaf curl virus
}

# Optimal conditions for tomato growth in greenhouse
optimal_tomato_conditions = {
    'temperature': 25,
    'humidity': 70,
    'soil_pH': 6.5,
    'N': 100,
    'P': 50,
    'K': 150,
    'soil_quality': 8
}


# Function to calculate soil quality based on NPK values
def calculate_soil_quality(N, P, K):
    # Calculate deviations from optimal NPK values
    N_deviation = abs(N - optimal_tomato_conditions['N']) / optimal_tomato_conditions['N']
    P_deviation = abs(P - optimal_tomato_conditions['P']) / optimal_tomato_conditions['P']
    K_deviation = abs(K - optimal_tomato_conditions['K']) / optimal_tomato_conditions['K']

    # Calculate soil quality as a weighted average of deviations
    soil_quality = 10 - (N_deviation + P_deviation + K_deviation) * 3.33  # Scale to 1-10
    return max(1, min(10, soil_quality))  # Ensure soil quality is within 1-10


# Function to preprocess the image for the model
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize to the input size of your model (256x256)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the disease from the uploaded image
def predict_disease(uploaded_image):
    if tomato_model is None:
        return "Model not loaded"
    img = Image.open(uploaded_image)
    img_array = preprocess_image(img)
    prediction = tomato_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    # Map the predicted class index to the disease name
    class_names = list(disease_yield_loss.keys())
    return class_names[predicted_class[0]]


# Page Config
st.set_page_config(
    page_title="Dynamic Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# Sidebar for additional options
with st.sidebar:
    st.title("🌱 Dynamic Yield Prediction")
    st.header("Ravindu Chamod Abeyrathne")
    st.write("University College Birmingham")
    st.markdown("### ")
    auto_populate = st.button("🎲 Auto-Populate")
    reset = st.button("🔄 Reset")

    # Placeholder for calculation breakdown
    st.markdown("---")
    st.markdown("### Calculation Breakdown")
    calculation_placeholder = st.empty()


# Function to reset all inputs
def reset_inputs():
    for key in st.session_state.keys():
        del st.session_state[key]


# Reset inputs if reset button is clicked
if reset:
    reset_inputs()

# Auto-populate inputs with random values within acceptable ranges
if auto_populate:
    st.session_state['soil_pH'] = round(random.uniform(5.5, 7.5), 1)  # Optimal pH range for tomatoes
    st.session_state['temperature'] = round(random.uniform(20, 30), 1)  # Optimal temperature range
    st.session_state['humidity'] = round(random.uniform(60, 80), 1)  # Optimal humidity range
    st.session_state['wind_speed'] = round(random.uniform(0, 10), 1)  # Moderate wind speed
    st.session_state['N'] = round(random.uniform(50, 150), 1)  # Nitrogen range
    st.session_state['P'] = round(random.uniform(30, 100), 1)  # Phosphorus range
    st.session_state['K'] = round(random.uniform(100, 200), 1)  # Potassium range
    st.session_state['soil_type'] = random.choice(["Clay", "Sandy", "Loamy", "Peaty", "Saline"])
    st.session_state['detected_diseases'] = random.sample(list(disease_yield_loss.keys()), random.randint(1, 5))

# Input fields in the main area
st.title("🌱 Dynamic Yield Prediction")
st.markdown("### Input Parameters")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    soil_pH = st.number_input("Soil pH:", min_value=0.0, max_value=14.0, value=st.session_state.get('soil_pH', 6.5),
                              key='soil_pH')
    temperature = st.number_input("Temperature (°C):", min_value=0.0, max_value=50.0,
                                  value=st.session_state.get('temperature', 25.0), key='temperature')
    humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0,
                               value=st.session_state.get('humidity', 70.0), key='humidity')
    wind_speed = st.number_input("Wind Speed (km/h):", min_value=0.0, value=st.session_state.get('wind_speed', 5.0),
                                 key='wind_speed')

with col2:
    N = st.number_input("Nitrogen (N) content:", min_value=0.0, value=st.session_state.get('N', 50.0), key='N')
    P = st.number_input("Phosphorus (P) content:", min_value=0.0, value=st.session_state.get('P', 50.0), key='P')
    K = st.number_input("Potassium (K) content:", min_value=0.0, value=st.session_state.get('K', 50.0), key='K')
    soil_type = st.selectbox("Soil Type:", ["Clay", "Sandy", "Loamy", "Peaty", "Saline"], index=0, key='soil_type')

# Calculate soil quality based on NPK values
soil_quality = calculate_soil_quality(N, P, K)

# # File uploader for disease detection
# uploaded_file = st.file_uploader("Upload an image of a tomato leaf for disease detection:", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Uploaded Image.', use_column_width=True)
#     predicted_disease, confidence = predict_disease(uploaded_file)
#     st.write(f"**Detected Disease:** {predicted_disease} (Confidence: {confidence:.2f}%)")


# File uploader for disease detection
uploaded_file = st.file_uploader("Upload an image of a tomato leaf for disease detection:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Predict the disease from the uploaded image
    predicted_disease = predict_disease(uploaded_file)
    st.write(f"**Detected Disease:** {predicted_disease}")

    # Add the detected disease to the list of detected diseases
    if 'detected_diseases' not in st.session_state:
        st.session_state['detected_diseases'] = []
    if predicted_disease not in st.session_state['detected_diseases']:
        st.session_state['detected_diseases'].append(predicted_disease)

# Disease selection in an expander
with st.expander("🌡️ Select Detected Diseases"):
    detected_diseases = st.multiselect(
        "Select Detected Diseases:",
        list(disease_yield_loss.keys()),
        default=st.session_state.get('detected_diseases', []),
        key='detected_diseases'
    )

# Predict button
if st.button("🚀 Predict Yield"):
    if soil_pH < 0 or soil_pH > 14:
        st.error("Soil pH must be between 0 and 14.")
    elif humidity < 0 or humidity > 100:
        st.error("Humidity must be between 0 and 100.")
    else:
        # Show loading animation
        with st.spinner("Calculating yield... Please wait."):
            time.sleep(2)  # Simulate a delay for calculations

        # Perform calculations
        try:
            # Get today's date
            today = datetime.today()
            year = today.year
            month = today.month
            day_of_year = today.timetuple().tm_yday

            # Prepare input data
            sample_data = {
                'Soil_pH': [soil_pH],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'Wind_Speed': [wind_speed],
                'N': [N],
                'P': [P],
                'K': [K],
                'Soil_Quality': [soil_quality],
                'Year': [year],
                'Month': [month],
                'Day_of_Year': [day_of_year],
                'Soil_Type': [soil_type]
            }

            # Convert to DataFrame
            sample_df = pd.DataFrame(sample_data)

            try:
                # Load and generate predictions using the model
                model = joblib.load('crop_yield_pipeline.pkl')
                predicted_yield = model.predict(sample_df)[0]
            except Exception as e:
                # If model loading fails, use a random prediction for demonstration
                predicted_yield = random.uniform(50, 150)
                st.warning("Model Loading Failed. Using demonstration mode with random predictions.")

            # Calculate adjusted yield based on diseases
            total_yield_loss = 0
            disease_impact_details = []

            if detected_diseases:
                for disease in detected_diseases:
                    if disease in disease_yield_loss:
                        base_loss = disease_yield_loss[disease]['loss']
                        disease_type = disease_yield_loss[disease]['type']

                        # Skip adjustments for deficiencies
                        if disease_type == 'Deficiency':
                            total_yield_loss += base_loss
                            disease_impact_details.append(
                                f"- **{disease}** ({disease_type}):\n"
                                f"  - Base Loss: {base_loss}%\n"
                                f"  - No temperature or humidity adjustments for deficiencies."
                            )
                        else:
                            # Get optimal conditions for the disease
                            optimal_temp = optimal_conditions[disease]['temp']
                            optimal_humidity = optimal_conditions[disease]['humidity']

                            # Compare current conditions with optimal conditions
                            temp_deviation = abs(temperature - optimal_temp)
                            humidity_deviation = humidity - optimal_humidity

                            # Temperature adjustment
                            if temp_deviation > 5:
                                base_loss *= 0.5
                                temp_adjustment = f"Temperature deviation > 5°C: Loss reduced by 50%"
                            else:
                                temp_adjustment = "No temperature adjustment"

                            # Humidity adjustment
                            if humidity_deviation > 10:
                                base_loss *= 1.1
                                humidity_adjustment = f"Humidity > optimal + 10%: Loss increased by 10%"
                            elif humidity_deviation < -10:
                                base_loss *= 0.8
                                humidity_adjustment = f"Humidity < optimal - 10%: Loss reduced by 20%"
                            else:
                                humidity_adjustment = "No humidity adjustment"

                            # Cap individual disease loss
                            capped_loss = min(base_loss, 50)
                            total_yield_loss += capped_loss

                            # Store disease impact details
                            disease_impact_details.append(
                                f"- **{disease}** ({disease_type}):\n"
                                f"  - Base Loss: {disease_yield_loss[disease]['loss']}%\n"
                                f"  - Optimal Temperature: {optimal_temp}°C (Current: {temperature}°C)\n"
                                f"  - Optimal Humidity: {optimal_humidity}% (Current: {humidity}%)\n"
                                f"  - {temp_adjustment}\n"
                                f"  - {humidity_adjustment}\n"
                                f"  - Capped Loss: {capped_loss:.1f}%"
                            )

                # Cap total yield loss
                total_yield_loss = min(total_yield_loss, 100)
                adjusted_yield = predicted_yield * (1 - total_yield_loss / 100)
            else:
                adjusted_yield = predicted_yield
                total_yield_loss = 0

            # Display final adjusted yield prominently
            st.success(f"### Final Adjusted Yield: {adjusted_yield:.2f}")

            # Display calculation breakdown in the sidebar
            with calculation_placeholder.container():
                st.markdown("### Calculation Breakdown")
                st.markdown(f"- **Predicted Yield (Before Adjustment):** {predicted_yield:.2f}")
                if detected_diseases:
                    st.markdown("#### Disease Impact:")
                    for detail in disease_impact_details:
                        st.markdown(detail)
                    st.markdown(f"- **Total Yield Loss:** {total_yield_loss:.1f}%")
                st.markdown(f"- **Adjusted Yield (After Disease Impact):** {adjusted_yield:.2f}")

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Yield Analysis", "🌡️ Disease Analysis", "🌱 Current Conditions",
                "⚖️ Comparison", "📈 Nutrient Analysis", "🌍 Disease Risk Heatmap", "🔍 In-Depth Comparison"
            ])

            with tab1:
                st.subheader("Yield Impact Analysis")

                # Create figure with smaller size
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced from (6, 4)
                yields = [predicted_yield, adjusted_yield]
                labels = ['Predicted Yield', 'Adjusted Yield']

                ax.bar(labels, yields, color=['#4c72b0', '#dd8452'])
                ax.set_ylabel('Yield')

                # Adjust title and label sizes
                ax.set_title('Yield Impact Analysis', fontsize=5)
                ax.tick_params(axis='both', labelsize=5)
                ax.set_ylabel('Yield', fontsize=5)

                decrease = ((predicted_yield - adjusted_yield) / predicted_yield) * 100
                ax.text(0.5, 0.95, f'Yield Decrease: {decrease:.1f}%',
                        horizontalalignment='center', transform=ax.transAxes, fontsize=8)

                # Add tight layout to prevent text cutoff
                plt.tight_layout()

                st.pyplot(fig)

            with tab2:
                st.subheader("Disease Analysis")
                if not detected_diseases:
                    st.write("No diseases detected.")
                else:
                    # Create a line chart for disease activity (temperature)
                    fig1, ax1 = plt.subplots(figsize=(8, 5))

                    # Temperature range for disease activity
                    temp_range = np.linspace(10, 40, 100)  # Temperature range from 10°C to 40°C

                    for disease in detected_diseases:
                        if disease in optimal_conditions:  # Skip deficiencies
                            optimal_temp = optimal_conditions[disease]['temp']
                            activity = np.exp(-0.5 * ((temp_range - optimal_temp) / 5) ** 2)  # Bell curve
                            ax1.plot(temp_range, activity, label=disease)

                    ax1.axvline(x=temperature, color='red', linestyle='--', label='Current Temperature')
                    ax1.set_xlabel('Temperature (°C)')
                    ax1.set_ylabel('Activity Level')
                    ax1.set_title('Disease Activity vs Temperature')
                    ax1.legend()

                    st.pyplot(fig1)

                    # Create a line chart for disease activity (humidity)
                    fig2, ax2 = plt.subplots(figsize=(8, 5))

                    # Humidity range for disease activity
                    humidity_range = np.linspace(30, 100, 100)  # Humidity range from 30% to 100%

                    for disease in detected_diseases:
                        if disease in optimal_conditions:  # Skip deficiencies
                            optimal_humidity = optimal_conditions[disease]['humidity']
                            activity = np.exp(-0.5 * ((humidity_range - optimal_humidity) / 10) ** 2)  # Bell curve
                            ax2.plot(humidity_range, activity, label=disease)

                    ax2.axvline(x=humidity, color='red', linestyle='--', label='Current Humidity')
                    ax2.set_xlabel('Humidity (%)')
                    ax2.set_ylabel('Activity Level')
                    ax2.set_title('Disease Activity vs Humidity')
                    ax2.legend()

                    st.pyplot(fig2)

            with tab3:
                st.subheader("Current Conditions Analysis")
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

                conditions = ['Temperature', 'Humidity', 'Soil pH', 'N', 'P', 'K']
                current_values = [
                    temperature,
                    humidity,
                    soil_pH,
                    N,
                    P,
                    K
                ]
                optimal_values = [
                    optimal_tomato_conditions['temperature'],
                    optimal_tomato_conditions['humidity'],
                    optimal_tomato_conditions['soil_pH'],
                    optimal_tomato_conditions['N'],
                    optimal_tomato_conditions['P'],
                    optimal_tomato_conditions['K']
                ]

                # Normalize values for radar chart
                max_value = max(max(current_values), max(optimal_values))
                current_values_normalized = [v / max_value for v in current_values]
                optimal_values_normalized = [v / max_value for v in optimal_values]

                angles = np.linspace(0, 2 * np.pi, len(conditions), endpoint=False)
                current_values_normalized.append(current_values_normalized[0])
                optimal_values_normalized.append(optimal_values_normalized[0])
                angles = np.concatenate((angles, [angles[0]]))

                ax.plot(angles, current_values_normalized, color='#4c72b0', label='Current Conditions')
                ax.fill(angles, current_values_normalized, alpha=0.25, color='#4c72b0')
                ax.plot(angles, optimal_values_normalized, color='#55a868', label='Optimal Conditions')
                ax.fill(angles, optimal_values_normalized, alpha=0.25, color='#55a868')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(conditions)
                ax.set_title('Current vs Optimal Conditions')
                ax.legend()

                st.pyplot(fig)

            with tab4:
                st.subheader("Comparison Analysis")
                fig, ax = plt.subplots(figsize=(8, 5))

                conditions = ['Temperature', 'Humidity', 'Soil pH', 'N', 'P', 'K']
                current_values = [
                    temperature,
                    humidity,
                    soil_pH,
                    N,
                    P,
                    K
                ]
                optimal_values = [
                    optimal_tomato_conditions['temperature'],
                    optimal_tomato_conditions['humidity'],
                    optimal_tomato_conditions['soil_pH'],
                    optimal_tomato_conditions['N'],
                    optimal_tomato_conditions['P'],
                    optimal_tomato_conditions['K']
                ]

                x = np.arange(len(conditions))
                width = 0.35

                ax.bar(x - width / 2, current_values, width, label='Current Conditions', color='#4c72b0')
                ax.bar(x + width / 2, optimal_values, width, label='Optimal Conditions', color='#55a868')

                ax.set_ylabel('Value')
                ax.set_title('Current vs Optimal Conditions')
                ax.set_xticks(x)
                ax.set_xticklabels(conditions, rotation=45, ha='right')
                ax.legend()

                st.pyplot(fig)

            with tab5:
                st.subheader("Nutrient Analysis")
                fig, ax = plt.subplots(figsize=(8, 5))

                conditions = ['Current', 'Optimal']
                N_values = [N, optimal_tomato_conditions['N']]
                P_values = [P, optimal_tomato_conditions['P']]
                K_values = [K, optimal_tomato_conditions['K']]

                ax.bar(conditions, N_values, label='Nitrogen (N)', color='#4c72b0')
                ax.bar(conditions, P_values, bottom=N_values, label='Phosphorus (P)', color='#dd8452')
                ax.bar(conditions, K_values, bottom=[N_values[i] + P_values[i] for i in range(len(N_values))],
                       label='Potassium (K)', color='#55a868')

                ax.set_ylabel('Nutrient Levels')
                ax.set_title('Current vs Optimal Nutrient Levels')
                ax.legend()

                st.pyplot(fig)

            with tab6:
                st.subheader("Disease Risk Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))

                # Temperature and humidity ranges
                temp_range = np.linspace(10, 40, 10)  # Temperature range from 10°C to 40°C
                humidity_range = np.linspace(30, 100, 10)  # Humidity range from 30% to 100%

                # Initialize activity matrix
                activity = np.zeros((len(temp_range), len(humidity_range)))

                # Calculate disease activity for detected diseases
                for disease in detected_diseases:
                    if disease in optimal_conditions:  # Skip deficiencies
                        optimal_temp = optimal_conditions[disease]['temp']
                        optimal_humidity = optimal_conditions[disease]['humidity']

                        # Calculate activity for each temperature and humidity combination
                        for i, temp in enumerate(temp_range):
                            for j, humidity in enumerate(humidity_range):
                                # Simulate disease activity using a Gaussian distribution
                                temp_activity = np.exp(-0.5 * ((temp - optimal_temp) / 5) ** 2)  # Temperature effect
                                humidity_activity = np.exp(
                                    -0.5 * ((humidity - optimal_humidity) / 10) ** 2)  # Humidity effect
                                activity[i, j] += temp_activity * humidity_activity  # Combine effects

                # Normalize activity to a 0-1 scale
                if np.max(activity) > 0:
                    activity /= np.max(activity)

                # Plot the heatmap with annotations limited to 1 decimal place
                sns.heatmap(
                    activity,
                    xticklabels=[f"{h:.1f}" for h in humidity_range],  # Format humidity labels to 1 decimal place
                    yticklabels=[f"{t:.1f}" for t in temp_range],  # Format temperature labels to 1 decimal place
                    annot=True,
                    fmt=".1f",  # Annotations limited to 1 decimal place
                    cmap="YlOrRd",
                    ax=ax
                )

                # Add markers for current temperature and humidity
                current_temp = temperature  # Current temperature from user input
                current_humidity = humidity  # Current humidity from user input

                # Find the indices of the current temperature and humidity in the ranges
                temp_index = np.abs(temp_range - current_temp).argmin()
                humidity_index = np.abs(humidity_range - current_humidity).argmin()

                # Add a vertical line for current humidity
                ax.axvline(x=humidity_index + 0.5, color='blue', linestyle='--', linewidth=2,
                           label=f'Current Humidity: {current_humidity}%')

                # Add a horizontal line for current temperature
                ax.axhline(y=temp_index + 0.5, color='green', linestyle='--', linewidth=2,
                           label=f'Current Temperature: {current_temp}°C')

                ax.set_xlabel('Humidity (%)')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title('Disease Activity Heatmap (Detected Diseases)')
                ax.legend(loc='upper right')  # Add legend to show the markers

                st.pyplot(fig)

            with tab7:
                st.subheader("In-Depth Calculation")

                # Soil Quality Calculation
                st.write("### Soil Quality Calculation")
                st.write("Soil quality is calculated based on deviations of N, P, and K from their optimal values:")
                st.latex(r"""
                \text{Soil Quality} = 10 - \left( \frac{|\text{N} - \text{N}_{\text{optimal}}|}{\text{N}_{\text{optimal}}} + \frac{|\text{P} - \text{P}_{\text{optimal}}|}{\text{P}_{\text{optimal}}} + \frac{|\text{K} - \text{K}_{\text{optimal}}|}{\text{K}_{\text{optimal}}} \right) \times 3.33
                """)
                st.write(
                    "**Explanation:** Soil quality is adjusted based on how far N, P, and K are from their optimal values (N=100, P=50, K=150). The result is capped between 1 and 10.")

                # Optimal Values
                st.write("Where:")
                st.latex(r"""
                \text{N}_{\text{optimal}} = 100, \quad \text{P}_{\text{optimal}} = 50, \quad \text{K}_{\text{optimal}} = 150
                """)
                st.write("**Explanation:** These are the optimal values for N, P, and K in the soil.")

                # Current Values
                st.write("Current values:")
                st.latex(fr"""
                \text{{N}} = {N}, \quad \text{{P}} = {P}, \quad \text{{K}} = {K}
                """)
                st.write("**Explanation:** These are the current N, P, and K values in the soil.")

                # Soil Quality Calculation with Current Values
                st.write("Soil quality calculation with current values:")
                st.latex(fr"""
                \text{{Soil Quality}} = 10 - \left( \frac{{|{N} - 100|}}{{100}} + \frac{{|{P} - 50|}}{{50}} + \frac{{|{K} - 150|}}{{150}} \right) \times 3.33 = {soil_quality:.2f}
                """)
                st.write(
                    "**Explanation:** This equation calculates the soil quality based on the current N, P, and K values.")

                st.divider()

                # Predicted Yield Calculation
                st.write("### Predicted Yield Calculation")
                st.write(
                    "The predicted yield is calculated using a pre-trained machine learning model with input features:")
                st.latex(r"""
                \text{Predicted Yield} = f(\text{Soil pH}, \text{Temperature}, \text{Humidity}, \text{Wind Speed}, \text{N}, \text{P}, \text{K}, \text{Soil Quality}, \text{Year}, \text{Month}, \text{Day of Year}, \text{Soil Type})
                """)
                st.write(f"**Current Predicted Yield:** \( {predicted_yield:.2f} \)")
                st.write("**Explanation:** The model predicts the yield based on environmental and soil conditions.")

                st.divider()

                # Disease Impact Calculation
                st.write("### Disease Impact Calculation")
                st.write("For each disease, the yield loss is calculated as:")
                st.latex(r"""
                \text{Base Loss} = \text{Disease Base Loss Percentage}
                """)
                st.write("**Explanation:** Each disease has a base percentage of yield loss.")

                st.write("**Temperature Adjustment:**")
                st.latex(r"""
                \text{If } |\text{Temperature} - \text{Optimal Temperature}| > 5°C, \text{Base Loss} \times 0.5
                """)
                st.write(
                    "**Explanation:** If the temperature deviates by more than 5°C from the optimal, the loss is reduced by 50%.")

                st.write("**Humidity Adjustment:**")
                st.latex(r"""
                \text{If } \text{Humidity} > \text{Optimal Humidity} + 10\%, \text{Base Loss} \times 1.1
                """)
                st.latex(r"""
                \text{If } \text{Humidity} < \text{Optimal Humidity} - 10\%, \text{Base Loss} \times 0.8
                """)
                st.write(
                    "**Explanation:** If humidity is more than 10% above optimal, the loss increases by 10%. If it's more than 10% below, the loss decreases by 20%.")

                st.write("**Capped Loss:**")
                st.latex(r"""
                \text{Capped Loss} = \min(\text{Adjusted Base Loss}, 50\%)
                """)
                st.write("**Explanation:** Each disease's loss is capped at 50%.")

                st.write("**Total Yield Loss:**")
                st.latex(r"""
                \text{Total Yield Loss} = \sum \text{Capped Loss for Each Disease}
                """)
                st.latex(r"""
                \text{Total Yield Loss} = \min(\text{Total Yield Loss}, 100\%)
                """)
                st.write(f"**Current Total Yield Loss:** \( {total_yield_loss:.1f}\% \)")
                st.write(
                    "**Explanation:** The total yield loss is the sum of all capped losses, with a maximum of 100%.")

                st.divider()

                # Adjusted Yield Calculation
                st.write("### Adjusted Yield Calculation")
                st.write("The adjusted yield is calculated by applying the total yield loss to the predicted yield:")
                st.latex(r"""
                \text{Adjusted Yield} = \text{Predicted Yield} \times \left(1 - \frac{\text{Total Yield Loss}}{100}\right)
                """)
                st.write("**Current Calculation:**")
                st.latex(fr"""
                \text{{Adjusted Yield}} = {predicted_yield:.2f} \times \left(1 - \frac{{{total_yield_loss:.1f}}}{100}\right) = {adjusted_yield:.2f}
                """)
                st.write(
                    "**Explanation:** The adjusted yield is the predicted yield reduced by the total yield loss percentage.")

                st.divider()

                # Final Adjusted Yield
                st.write("### Final Adjusted Yield")
                st.latex(fr"""
                \boxed{{\text{{Final Adjusted Yield}} = {adjusted_yield:.2f}}}
                """)
                st.write("**Explanation:** This is the final yield after accounting for all adjustments.")


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
