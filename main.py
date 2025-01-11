import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Disease impact dictionary
disease_yield_loss = {
    'Early_Blight': 80,  # Alternaria solani
    'Septoria_Leaf_Spot': 50,  # Septoria lycopersici
    'Grey_Mould': 40,  # Botrytis cinerea
    'Fusarium_Wilt': 70,  # Fusarium oxysporum f. sp. lycopersici
    'Crown_and_Root_Rot': 90,  # Fusarium oxysporum f. sp. radicis-lycopersici
    'Verticillium_Wilt': 50,  # Verticillium dahliae
    'Bacterial_Canker': 84,  # Clavibacter michiganensis subsp. michiganensis
    'Bacterial_Speck': 75,  # Pseudomonas syringae pv. tomato
    'Stolbur': 80,  # Candidatus Phytoplasma solani
    'Spotted_Wilt_Disease': 95,  # Tomato spotted wilt virus
    'Tomato_Fern_Leaf': 100,  # Cucumber mosaic virus
    'Tomato_Yellow_Leaf_Curl_Disease': 100,  # Tomato yellow leaf curl virus
    'Tomato_Brown_Rugose_Fruit_Disease': 100,  # Tomato brown rugose fruit virus
    'Tomato_Mosaic_Disease': 70,  # Tomato mosaic virus
    'Bunchy_Top_of_Tomato': 90,  # Potato spindle tuber viroid
}

# Define optimal conditions for each disease
optimal_conditions = {
    'Early_Blight': {'temp': 25, 'humidity': 95},  # Near-saturated humidity
    'Septoria_Leaf_Spot': {'temp': 20, 'humidity': 80},  # High humidity
    'Grey_Mould': {'temp': 25, 'humidity': 90},
    'Fusarium_Wilt': {'temp': 27, 'humidity': 80},  # Avoid extremes
    'Crown_and_Root_Rot': {'temp': 27, 'humidity': 80},  # High in early stages
    'Verticillium_Wilt': {'temp': 25, 'humidity': 70},  # Medium to high temp
    'Bacterial_Canker': {'temp': 23, 'humidity': 80},
    'Bacterial_Speck': {'temp': 24, 'humidity': 80},
    'Stolbur': {'temp': 25, 'humidity': 80},  # High humidity
    'Spotted_Wilt_Disease': {'temp': 28, 'humidity': 70},  # Medium to high temp
    'Tomato_Fern_Leaf': {'temp': 28, 'humidity': 70},  # Below 32°C
    'Tomato_Yellow_Leaf_Curl_Disease': {'temp': 28, 'humidity': 70},
    'Tomato_Brown_Rugose_Fruit_Disease': {'temp': 28, 'humidity': 70},
    'Tomato_Mosaic_Disease': {'temp': 25, 'humidity': 60},  # Below 70% humidity
    'Bunchy_Top_of_Tomato': {'temp': 31, 'humidity': 35}  # Dry conditions
}


class InsightsWindow:
    def __init__(self, parent, disease_data, current_conditions, predicted_yield, adjusted_yield):
        self.window = tk.Toplevel(parent)
        self.window.title("Prediction Insights")
        self.window.geometry("800x600")

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.disease_tab = ttk.Frame(self.notebook)
        self.yield_tab = ttk.Frame(self.notebook)
        self.conditions_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.disease_tab, text='Disease Analysis')
        self.notebook.add(self.yield_tab, text='Yield Impact')
        self.notebook.add(self.conditions_tab, text='Current Conditions')

        self.create_disease_analysis(disease_data, current_conditions)
        self.create_yield_analysis(predicted_yield, adjusted_yield)
        self.create_conditions_analysis(current_conditions)

    def create_disease_analysis(self, disease_data, current_conditions):
        if not disease_data:  # If no diseases detected
            label = ttk.Label(self.disease_tab, text="No diseases detected")
            label.pack(pady=20)
            return

        # Create a Figure for disease analysis
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        diseases = list(disease_data.keys())
        optimal_temps = [disease_data[d]['temp'] for d in diseases]
        current_temp = current_conditions['temperature']

        # Bar plot of optimal temperatures vs current
        x = np.arange(len(diseases))
        width = 0.35

        ax.bar(x - width / 2, optimal_temps, width, label='Optimal Temperature')
        ax.bar(x + width / 2, [current_temp] * len(diseases), width, label='Current Temperature')

        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Disease Optimal vs Current Temperature')
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, self.disease_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add text information
        text = tk.Text(self.disease_tab, height=10)
        text.pack(fill=tk.X, padx=5, pady=5)
        text.insert(tk.END, "Optimal Conditions for Detected Diseases:\n\n")

        for disease in diseases:
            text.insert(tk.END, f"{disease}:\n")
            text.insert(tk.END, f"  Optimal Temperature: {disease_data[disease]['temp']}°C\n")
            text.insert(tk.END, f"  Optimal Humidity: {disease_data[disease]['humidity']}%\n\n")

    def create_yield_analysis(self, predicted_yield, adjusted_yield):
        # Create yield impact visualization
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        yields = [predicted_yield, adjusted_yield]
        labels = ['Predicted Yield', 'Adjusted Yield']

        ax.bar(labels, yields)
        ax.set_ylabel('Yield')
        ax.set_title('Yield Impact Analysis')

        # Add percentage decrease
        decrease = ((predicted_yield - adjusted_yield) / predicted_yield) * 100
        ax.text(0.5, 0.95, f'Yield Decrease: {decrease:.1f}%',
                horizontalalignment='center', transform=ax.transAxes)

        canvas = FigureCanvasTkAgg(fig, self.yield_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_conditions_analysis(self, current_conditions):
        # Create radar chart of current conditions
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='polar')

        conditions = ['Temperature', 'Humidity', 'Soil pH', 'N', 'P', 'K']
        values = [
            current_conditions['temperature'],
            current_conditions['humidity'],
            current_conditions['soil_pH'],
            current_conditions['N'],
            current_conditions['P'],
            current_conditions['K']
        ]

        # Normalize values for radar chart
        values_normalized = [v / max(values) for v in values]
        angles = np.linspace(0, 2 * np.pi, len(conditions), endpoint=False)

        # Close the plot
        values_normalized.append(values_normalized[0])
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, values_normalized)
        ax.fill(angles, values_normalized, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(conditions)
        ax.set_title('Current Growing Conditions')

        canvas = FigureCanvasTkAgg(fig, self.conditions_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def validate_inputs():
    """Validate all input fields."""
    try:
        soil_pH = float(soil_pH_entry.get())
        temperature = float(temperature_entry.get())
        humidity = float(humidity_entry.get())
        wind_speed = float(wind_speed_entry.get())
        N = float(N_entry.get())
        P = float(P_entry.get())
        K = float(K_entry.get())
        soil_quality = float(soil_quality_entry.get())
        soil_type = soil_type_combobox.get()

        # Basic range checks
        if not (0 <= soil_pH <= 14):
            raise ValueError("Soil pH must be between 0 and 14")
        if not (0 <= humidity <= 100):
            raise ValueError("Humidity must be between 0 and 100")
        if not (0 <= soil_quality <= 10):
            raise ValueError("Soil quality must be between 0 and 10")

        return True
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        return False


def predict_yield():
    if not validate_inputs():
        return

    try:
        # Get today's date
        today = datetime.today()
        year = today.year
        month = today.month
        day_of_year = today.timetuple().tm_yday

        # Get input values from the UI
        soil_pH = float(soil_pH_entry.get())
        temperature = float(temperature_entry.get())
        humidity = float(humidity_entry.get())
        wind_speed = float(wind_speed_entry.get())
        N = float(N_entry.get())
        P = float(P_entry.get())
        K = float(K_entry.get())
        soil_quality = float(soil_quality_entry.get())
        soil_type = soil_type_combobox.get()

        # Get detected diseases from the checkboxes
        detected_diseases = [disease for disease, var in disease_vars.items() if var.get() == 1]

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
            messagebox.showwarning("Model Loading Failed",
                                   "Using demonstration mode with random predictions.")

        # Calculate adjusted yield based on diseases
        if detected_diseases:
            total_yield_loss = 0
            disease_impact_details = []

            for disease in detected_diseases:
                if disease in disease_yield_loss:
                    base_loss = disease_yield_loss[disease]

                    # Adjust for temperature and humidity
                    optimal_temp = optimal_conditions[disease]['temp']
                    optimal_humidity = optimal_conditions[disease]['humidity']

                    if abs(temperature - optimal_temp) > 5:
                        base_loss *= 0.5
                    if humidity > optimal_humidity + 10:
                        base_loss *= 1.1
                    elif humidity < optimal_humidity - 10:
                        base_loss *= 0.8

                    # Cap individual disease loss
                    capped_loss = min(base_loss, 50)
                    total_yield_loss += capped_loss

                    disease_impact_details.append(
                        f"{disease}: Base Loss = {disease_yield_loss[disease]}%, "
                        f"Adjusted Loss = {capped_loss:.1f}%"
                    )

            # Cap total yield loss
            total_yield_loss = min(total_yield_loss, 100)
            adjusted_yield = predicted_yield * (1 - total_yield_loss / 100)
        else:
            adjusted_yield = predicted_yield
            total_yield_loss = 0
            disease_impact_details = []

        # Create current conditions dictionary
        current_conditions = {
            'temperature': temperature,
            'humidity': humidity,
            'soil_pH': soil_pH,
            'N': N,
            'P': P,
            'K': K
        }

        # Get detected diseases and their optimal conditions
        detected_disease_data = {
            disease: optimal_conditions[disease]
            for disease in detected_diseases
            if disease in optimal_conditions
        }

        # Create insights window
        insights_window = InsightsWindow(
            root,
            detected_disease_data,
            current_conditions,
            predicted_yield,
            adjusted_yield
        )

        # Display basic results in message box
        result_message = (
            f"Predicted Yield (Before Adjustment): {predicted_yield:.2f}\n\n"
            f"Detected Diseases and Their Impact:\n"
        )
        if disease_impact_details:
            result_message += "\n".join(disease_impact_details)
        else:
            result_message += "No diseases detected.\n"

        result_message += (
            f"\nTotal Yield Loss: {total_yield_loss:.1f}%\n"
            f"Adjusted Yield (After Disease Impact): {adjusted_yield:.2f}"
        )

        messagebox.showinfo("Prediction Results", result_message)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def show_activity_graphs(detected_diseases, current_temperature, current_humidity):
    """
    Display temperature and humidity activity graphs for detected diseases.
    """
    # Define optimal temperature and humidity for each disease
    optimal_conditions = {
        'Early_Blight': {'temp': 25, 'humidity': 95},  # Near-saturated humidity
        'Septoria_Leaf_Spot': {'temp': 20, 'humidity': 80},  # High humidity
        'Grey_Mould': {'temp': 25, 'humidity': 90},
        'Fusarium_Wilt': {'temp': 27, 'humidity': 80},  # Avoid extremes
        'Crown_and_Root_Rot': {'temp': 27, 'humidity': 80},  # High in early stages
        'Verticillium_Wilt': {'temp': 25, 'humidity': 70},  # Medium to high temp
        'Bacterial_Canker': {'temp': 23, 'humidity': 80},
        'Bacterial_Speck': {'temp': 24, 'humidity': 80},
        'Stolbur': {'temp': 25, 'humidity': 80},  # High humidity
        'Spotted_Wilt_Disease': {'temp': 28, 'humidity': 70},  # Medium to high temp
        'Tomato_Fern_Leaf': {'temp': 28, 'humidity': 70},  # Below 32°C
        'Tomato_Yellow_Leaf_Curl_Disease': {'temp': 28, 'humidity': 70},  # Not available
        'Tomato_Brown_Rugose_Fruit_Disease': {'temp': 28, 'humidity': 70},  # Not available
        'Tomato_Mosaic_Disease': {'temp': 25, 'humidity': 60},  # Below 70% humidity
        'Bunchy_Top_of_Tomato': {'temp': 31, 'humidity': 35}  # Dry conditions
    }

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Disease Activity Graphs", fontsize=16)

    # Plot 1: Temperature Activity
    temp_range = np.linspace(10, 40, 100)  # Temperature range from 10°C to 40°C
    for disease in detected_diseases:
        optimal_temp = optimal_conditions[disease]['temp']
        activity = np.exp(-0.5 * ((temp_range - optimal_temp) / 5) ** 2)  # Bell curve
        ax1.plot(temp_range, activity, label=disease)
    ax1.axvline(x=current_temperature, color='red', linestyle='--', label='Current Temperature')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Activity Level')
    ax1.set_title('Temperature Activity')
    ax1.legend()

    # Plot 2: Humidity Activity
    humidity_range = np.linspace(30, 100, 100)  # Humidity range from 30% to 100%
    for disease in detected_diseases:
        optimal_humidity = optimal_conditions[disease]['humidity']
        activity = np.exp(-0.5 * ((humidity_range - optimal_humidity) / 10) ** 2)  # Bell curve
        ax2.plot(humidity_range, activity, label=disease)
    ax2.axvline(x=current_humidity, color='red', linestyle='--', label='Current Humidity')
    ax2.set_xlabel('Humidity (%)')
    ax2.set_ylabel('Activity Level')
    ax2.set_title('Humidity Activity')
    ax2.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()



def reset_fields():
    """Reset all input fields and checkboxes."""
    soil_pH_entry.delete(0, tk.END)
    temperature_entry.delete(0, tk.END)
    humidity_entry.delete(0, tk.END)
    wind_speed_entry.delete(0, tk.END)
    N_entry.delete(0, tk.END)
    P_entry.delete(0, tk.END)
    K_entry.delete(0, tk.END)
    soil_quality_entry.delete(0, tk.END)
    soil_type_combobox.set("Clay")  # Reset Combobox to default value
    for var in disease_vars.values():
        var.set(0)  # Uncheck all disease checkboxes


def auto_populate():
    """Auto populate input fields with random values."""
    reset_fields()  # Clear all fields first

    # Randomize soil pH (typical range: 4.5 to 8.5)
    soil_pH_entry.insert(0, f"{random.uniform(4.5, 8.5):.2f}")

    # Randomize temperature (typical range: 10°C to 40°C)
    temperature_entry.insert(0, f"{random.uniform(10, 40):.2f}")

    # Randomize humidity (typical range: 30% to 100%)
    humidity_entry.insert(0, f"{random.uniform(30, 100):.2f}")

    # Randomize wind speed (typical range: 0 to 20 km/h)
    wind_speed_entry.insert(0, f"{random.uniform(0, 20):.2f}")

    # Randomize Nitrogen content (typical range: 0 to 100)
    N_entry.insert(0, f"{random.uniform(0, 100):.2f}")

    # Randomize Phosphorus content (typical range: 0 to 100)
    P_entry.insert(0, f"{random.uniform(0, 100):.2f}")

    # Randomize Potassium content (typical range: 0 to 100)
    K_entry.insert(0, f"{random.uniform(0, 100):.2f}")

    # Randomize Soil Quality (typical range: 1 to 10)
    soil_quality_entry.insert(0, f"{random.randint(1, 10)}")

    # Randomly select a soil type
    soil_types = ["Clay", "Sandy", "Loamy", "Peaty", "Saline"]
    soil_type_combobox.set(random.choice(soil_types))

    # Randomly select a subset of diseases
    for disease in disease_vars:
        disease_vars[disease].set(random.choice([0, 1]))  # Randomly set to 0 or 1


# Create the main window
root = tk.Tk()
root.title("Crop Yield Prediction")

# Create input fields
tk.Label(root, text="Soil pH:").grid(row=0, column=0, padx=10, pady=5)
soil_pH_entry = tk.Entry(root)
soil_pH_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Temperature (°C):").grid(row=1, column=0, padx=10, pady=5)
temperature_entry = tk.Entry(root)
temperature_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Humidity (%):").grid(row=2, column=0, padx=10, pady=5)
humidity_entry = tk.Entry(root)
humidity_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Wind Speed:").grid(row=3, column=0, padx=10, pady=5)
wind_speed_entry = tk.Entry(root)
wind_speed_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Nitrogen (N) content:").grid(row=4, column=0, padx=10, pady=5)
N_entry = tk.Entry(root)
N_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Phosphorus (P) content:").grid(row=5, column=0, padx=10, pady=5)
P_entry = tk.Entry(root)
P_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Potassium (K) content:").grid(row=6, column=0, padx=10, pady=5)
K_entry = tk.Entry(root)
K_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Soil Quality (1-10):").grid(row=7, column=0, padx=10, pady=5)
soil_quality_entry = tk.Entry(root)
soil_quality_entry.grid(row=7, column=1, padx=10, pady=5)

# Create a drop-down menu for Soil Type
tk.Label(root, text="Soil Type:").grid(row=8, column=0, padx=10, pady=5)
soil_type_combobox = ttk.Combobox(root, values=["Clay", "Sandy", "Loamy", "Peaty", "Saline"])
soil_type_combobox.grid(row=8, column=1, padx=10, pady=5)
soil_type_combobox.set("Clay")  # Set default value

# Create checkboxes for disease selection
disease_vars = {}
row_offset = 9
for i, disease in enumerate(disease_yield_loss.keys()):
    disease_vars[disease] = tk.IntVar()
    tk.Checkbutton(root, text=disease, variable=disease_vars[disease]).grid(row=row_offset + i, column=0, padx=10, pady=5, sticky="w")

# Create "Predict" button
predict_button = tk.Button(root, text="Predict Yield", command=predict_yield)
predict_button.grid(row=row_offset + len(disease_yield_loss), column=0, columnspan=2, pady=10)


# Create "Reset" button
reset_button = tk.Button(root, text="Reset", command=reset_fields)
reset_button.grid(row=row_offset + len(disease_yield_loss) + 1, column=1, pady=10)

# Create "Auto-Populate" button
auto_populate_button = tk.Button(root, text="Auto-Populate", command=auto_populate)
auto_populate_button.grid(row=row_offset + len(disease_yield_loss) + 2, column=0, columnspan=2, pady=10)

# Global variable to store visualization data
visualization_data = {}

# Run the application
root.mainloop()