import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Sample the data (25% of the original dataset)
sampled_df = df.sample(frac=1, random_state=42)

# Separate features and target column from the sampled data
X_sampled = sampled_df.drop(columns=['calss'])
y_sampled = sampled_df['calss']

# Perform train-test split (80:20 ratio) on the sampled data
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Check the shape of the sampled and split data
print(f"Shape of X_train_sampled: {X_train.shape}")
print(f"Shape of X_test_sampled: {X_test.shape}")
print(f"Shape of y_train_sampled: {y_train.shape}")
print(f"Shape of y_test_sampled: {y_test.shape}")
# Load models using pickle
model = [
    'ada_boost',
    'knn',
    'lda',
    'mlp',
    'logistic_regression',
    'naive_bayes',
    'random_forest',
    'decision_tree'
]

models = []
for model_name in model:
    with open(f'{model_name}.pkl', 'rb') as f:
        models.append(pickle.load(f))

# Define function for model prediction


# def predict_from_models(data):
#     # Load data into DataFrame
#     df = pd.read_csv(data.name)
#     df.drop(columns=['calss'], inplace=True)
#     scaled_data = scaler.transform(df)

#     # Make predictions from each model
#     predictions = []
#     for model in models:
#         preds = model.predict(scaled_data)
#         predictions.append(preds)

#     label_map = {label: code for code,
#                  label in enumerate(np.unique(predictions))}
#     inv_label_map = {v: k for k, v in label_map.items()}

#     # Perform majority voting using integer codes
#     predictions_array = np.array(predictions)
#     final_prediction = np.array([])
#     for i in range(predictions_array.shape[1]):
#         labels, counts = np.unique(predictions_array[:, i], return_counts=True)
#         most_common_label = labels[np.argmax(counts)]
#         most_common_code = label_map[most_common_label]
#         final_prediction = np.append(final_prediction, most_common_code)

#     # Map integer codes back to string labels
#     final_prediction_labels = [inv_label_map[code]
#                                for code in final_prediction]

#     return final_prediction_labels

#     # return final_prediction


# # Create Gradio interface
# iface = gr.Interface(fn=predict_from_models, inputs="file", outputs="text")

# # Launch the app
# iface.launch(share='True')
def predict_from_models(data):
    # Load data into DataFrame
    df = pd.read_csv(data.name)
    df.drop(columns=['calss'], inplace=True)
    scaled_data = scaler.transform(df)

    # Make predictions from each model
    predictions = []
    for model in models:
        preds = model.predict(scaled_data)
        predictions.append(preds)

    label_map = {label: code for code,
                 label in enumerate(np.unique(predictions))}
    inv_label_map = {v: k for k, v in label_map.items()}

    # Perform majority voting using integer codes
    predictions_array = np.array(predictions)
    final_prediction = np.array([])
    for i in range(predictions_array.shape[1]):
        labels, counts = np.unique(predictions_array[:, i], return_counts=True)
        most_common_label = labels[np.argmax(counts)]
        most_common_code = label_map[most_common_label]
        final_prediction = np.append(final_prediction, most_common_code)

    # Map integer codes back to string labels
    final_prediction_labels = [inv_label_map[code]
                               for code in final_prediction]

    # Calculate percentages
    categories, counts = np.unique(final_prediction_labels, return_counts=True)
    total_count = np.sum(counts)
    percentages = [count / total_count for count in counts]

    # Rename labels
    renamed_labels = ['BENIGN' if label == 'benign' else 'ADWARE' if label == 'asware' else label.upper()
                      for label in categories]

    # Debugging statements
    print("Renamed Labels:", renamed_labels)
    print("Percentages:", percentages)

    # Check if percentage of General Malware is greater than 20%
    if 'GENERALMALWARE' in renamed_labels:
        general_malware_index = np.where(
            np.array(renamed_labels) == 'GENERALMALWARE')[0][0]
        general_malware_percentage = percentages[general_malware_index]

        # Determine detection result based on percentage of General Malware
        if general_malware_percentage > 0.2:
            detection_result = 'Malicious activity detected(>20%)'
        else:
            detection_result = 'No malicious activity detected'
    else:
        detection_result = 'No General Malware found'

    # Plotting the bar graph
    fig, ax = plt.subplots()
    ax.bar(renamed_labels, counts)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of different categories')

    # Convert the plot to an image
    plt.tight_layout()
    image_path = 'output_plot.png'
    plt.savefig(image_path)  # Save the plot to a file
    plt.close()

    # Return the bar graph image and the detection result
    return image_path, detection_result


# Create Gradio interface
input_component = gr.File(label="Upload CSV file ")
output_component = [
    gr.Image(label="Counts vs Categories"), gr.Textbox(label="Prediction")]
iface = gr.Interface(fn=predict_from_models, inputs=input_component, outputs=output_component,
                     title="Android Malware Detection App")


# Launch the app
iface.launch(share='True')
