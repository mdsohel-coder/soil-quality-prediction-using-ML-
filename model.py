# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create folder for saving plots
os.makedirs("plots", exist_ok=True)

# Define the file path
file_path = "C:/Users/maham/OneDrive/Desktop/soil_qulaity/dataset/Soil Fertility Data.csv"

# --- 1. Data Loading and Cleaning ---
try:
    soil_data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

print("\nInitial Dataset Info:")
print(soil_data.info())

# Convert all columns to numeric
soil_data = soil_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaNs or duplicates
soil_data.dropna(inplace=True)
soil_data.drop_duplicates(inplace=True)

print("\n--- Data Cleaning Complete ---")
print(f"Cleaned dataset shape: {soil_data.shape}")
print("\nCleaned Dataset Info:")
print(soil_data.info())

# --- 2. Preprocessing ---
features = soil_data.drop(columns=["fertility"])
target = soil_data["fertility"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, random_state=42, stratify=target
)
print("\nData split into training and testing sets.")

# --- 3. Model Training ---
print("\n--- Training Random Forest Classifier ---")
model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=None)
model.fit(X_train, y_train)
print("Model training complete.")

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low (0)', 'Medium (1)', 'High (2)'],
            yticklabels=['Low (0)', 'Medium (1)', 'High (2)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Soil Fertility Prediction')
plt.savefig("plots/confusion_matrix.png")  # Save plot
plt.show()

# --- Feature Importance ---
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print("\n--- Feature Importance Analysis ---")
print(importance_df)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance for Soil Fertility')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.savefig("plots/feature_importance.png")  # Save plot
plt.show()

# --- 4. Crop Recommendation ---
def recommend_crops(fertility_level):
    crop_mapping = {
        0: ["Millet", "Sorghum", "Pulses"],
        1: ["Wheat", "Barley", "Maize"],
        2: ["Rice", "Sugarcane", "Vegetables"]
    }
    return crop_mapping.get(fertility_level, ["No recommendation"])



# --- Save Model & Scaler ---
joblib.dump(model, "soil_fertility_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and Scaler saved for deployment.")
print("Plots saved in 'plots/' folder.")
