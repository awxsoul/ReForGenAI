import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler # Keep if you might switch
from sklearn.preprocessing import MinMaxScaler # <--- ADD THIS LINE
import numpy as np # Ensure numpy is imported
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt

file_name = "trees.csv"
df = pd.read_csv(file_name)

csv_file="trees.csv"
df = pd.read_csv(csv_file)

print("Original DataFrame shape:", df.shape) 

# --- Preprocess the soil_ph column ---
ph_split = df['soil_ph'].astype(str).str.split('-', expand=True)
df['ph_min_str'] = ph_split[0]
df['ph_max_str'] = ph_split[1].fillna(ph_split[0])
df['ph_min'] = pd.to_numeric(df['ph_min_str'], errors='coerce')
df['ph_max'] = pd.to_numeric(df['ph_max_str'], errors='coerce')

# --- Handle potential NaN values created during conversion ---
initial_rows = len(df)
df.dropna(subset=['ph_min', 'ph_max'], inplace=True)
rows_after_dropna = len(df)
print(f"\nDropped {initial_rows - rows_after_dropna} rows due to invalid pH values.")

# --- Drop the original and temporary string columns ---
columns_to_drop = ['soil_ph', 'ph_min_str', 'ph_max_str']
# Check if columns exist before dropping to avoid errors
columns_exist = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_exist, inplace=True)
print(f"Dropped columns: {columns_exist}")


# --- Verify problematic rows are gone ---
problem_rows = df[df['treename'].str.contains('Problematic|Empty', na=False)]
if problem_rows.empty:
    print("\nSuccessfully dropped rows with problematic pH.")
else:
    print("\nWarning: Problematic rows still present:")
    print(problem_rows)

print(df.info())
print(df.head())


# Function to safely parse the string representation of a list
def parse_list_string(x):
    try:
        # Safely evaluate the string as a Python literal (in this case, a list)
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list representation
        # Might return an empty list, or None, or raise an error depending on needs
        print(f"Warning: Could not parse '{x}'. Returning empty list.")
        return []

# Apply this function to the relevant columns
df['climate_list'] = df['climate'].apply(parse_list_string)
df['soil_type_list'] = df['soil_type'].apply(parse_list_string)

# Verify the result (optional)
print("Original climate column sample:", df['climate'].iloc[0])
print("Parsed climate_list column sample:", df['climate_list'].iloc[0])
print("Original soil_type column sample:", df['soil_type'].iloc[0])
print("Parsed soil_type_list column sample:", df['soil_type_list'].iloc[0])


df = df.drop(columns=['climate', 'soil_type'])
print(df.columns)
print(df.head())

allowed_soil_types = set([
    'Acidic', 'Alluvial', 'Clay', 'Clay loam', 'Dry', 'Loamy', 'Moist',
    'Poor', 'Poorly drained', 'Sandy', 'Sandy loam', 'Swampy',
    'Well-drained', 'Wet'
])

def filter_soil_list(soil_list):
  """Keeps only soil types that are in the allowed_soil_types set."""
  # Use a list comprehension for concise filtering
  return [soil for soil in soil_list if soil in allowed_soil_types]

# Apply the function to each list in the 'soil_type_list' column
# This modifies the column in place (or rather, assigns the modified series back)
df['soil_type_list'] = df['soil_type_list'].apply(filter_soil_list)

# --- Display the result ---
print("DataFrame after removing non-allowed soil types from lists:")
print(df[['treename', 'soil_type_list']]) # Display relevant columns

# --- Check for Empty Lists ---

print("\nChecking for rows with empty soil_type_list...")

# Method 1: Check list length using apply(len)
mask_empty_len = df['soil_type_list'].apply(len) == 0

# Method 2: Check equality with an empty list using apply(lambda)
# mask_empty_eq = df['soil_type_list'].apply(lambda x: x == []) # Also works

# Filter the DataFrame using the mask
rows_with_empty_lists = df[mask_empty_len]

# --- Display the result of the check ---
if not rows_with_empty_lists.empty:
    print("\nRows where soil_type_list became empty after filtering:")
    print(rows_with_empty_lists)
else:
    print("\nNo rows with empty soil_type_list found after filtering.")

# --- 1. Identify Unique Categories & Encode ---

# Climate
mlb_climate = MultiLabelBinarizer()
climate_encoded = mlb_climate.fit_transform(df['climate_list'])
# Create a DataFrame with the encoded climate columns
climate_df = pd.DataFrame(climate_encoded, columns=mlb_climate.classes_, index=df.index)

# Soil Type
mlb_soil = MultiLabelBinarizer()
soil_encoded = mlb_soil.fit_transform(df['soil_type_list'])
# Create a DataFrame with the encoded soil type columns
soil_df = pd.DataFrame(soil_encoded, columns=mlb_soil.classes_, index=df.index)


# --- Combine Encoded Features with Original Data ---

# Drop the original list columns
df = df.drop(columns=['climate_list', 'soil_type_list'])

# Concatenate the new encoded columns
df = pd.concat([df, climate_df, soil_df], axis=1)

# --- Display Results ---
print("Unique Climate Classes:", mlb_climate.classes_)
print("Unique Soil Type Classes:", mlb_soil.classes_)
print("\nDataFrame Head after Encoding:")
print(df.head())
print("\nDataFrame Info after Encoding:")
df.info()

# Ensure both columns exist before proceeding (optional but good practice)
if 'Arid' in df.columns and 'Arid (high alt)' in df.columns:
    print("Combining 'Arid' and 'Arid (high alt)'...")

    # Update the 'Arid' column: It will be 1 if either 'Arid' or 'Arid (high alt)' is 1
    # The '|' operator performs an element-wise logical OR
    df['Arid'] = df['Arid'] | df['Arid (high alt)']

    # Drop the 'Arid (high alt)' column as it's now redundant
    df = df.drop(columns=['Arid (high alt)'])

    print("'Arid (high alt)' column dropped. 'Arid' column updated.")

    # Verify the result (optional)
    print("\nColumns after combination:")
    print(df.columns)
    print("\nValue counts for the updated 'Arid' column:")
    print(df['Arid'].value_counts()) # Check how many trees are now marked as Arid

else:
    print("One or both columns ('Arid', 'Arid (high alt)') not found. No changes made.")

import numpy as np

# Store the climate and soil classes for later use
climate_classes = list(mlb_climate.classes_)
soil_classes = list(mlb_soil.classes_)

# Add the combined 'Arid' class if it was created, and remove the specific high-alt one if needed
# (Adjust based on previous step's outcome)
if 'Arid' in df.columns and 'Arid (high alt)' not in df.columns and 'Arid (high alt)' in climate_classes:
     climate_classes.remove('Arid (high alt)')
     if 'Arid' not in climate_classes: # Ensure 'Arid' is present if it represents the combined category
        climate_classes.append('Arid')
        climate_classes.sort() # Optional: keep it sorted


print("Using Climate Classes:", climate_classes)
print("Using Soil Classes:", soil_classes)

def calculate_suitability_score(user_input, tree_row):
    """
    Calculates a suitability score for a tree based on user input conditions.

    Args:
        user_input (dict): A dictionary containing user's conditions:
                           {'temp': float, 'ele': float, 'ph': float,
                            'climate': str, 'soil': str}
        tree_row (pd.Series): A row from the DataFrame representing a single tree.

    Returns:
        float: A suitability score (higher is better).
               Returns -1 if input climate/soil is invalid.
    """
    score = 0
    max_possible_score = 5 # Temp, Ele, pH, Climate, Soil

    # 1. Temperature Check
    if tree_row['min-temp'] <= user_input['temp'] <= tree_row['max-temp']:
        score += 1

    # 2. Elevation Check
    if tree_row['min-ele'] <= user_input['ele'] <= tree_row['max-ele']:
        score += 1

    # 3. pH Check
    if tree_row['ph_min'] <= user_input['ph'] <= tree_row['ph_max']:
        score += 1

    # 4. Climate Check
    input_climate = user_input['climate']
    if input_climate in climate_classes:
        if tree_row[input_climate] == 1:
            score += 1
    else:
        print(f"Warning: Input climate '{input_climate}' not found in known classes.")
        # Decide how to handle invalid input: return error, score 0 for this part, etc.
        # For now, let's penalize significantly by returning -1
        # Alternatively, could just skip this check (score remains unchanged)
        return -1 # Indicate invalid input climate

    # 5. Soil Type Check
    input_soil = user_input['soil']
    if input_soil in soil_classes:
        if tree_row[input_soil] == 1:
            score += 1
    else:
        print(f"Warning: Input soil type '{input_soil}' not found in known classes.")
        # Penalize for invalid soil type
        return -1 # Indicate invalid input soil

    return score



# --- Example Usage ---
# Define hypothetical user input
example_input = {
    'temp': 25,      # degrees C
    'ele': 500,      # meters
    'ph': 6.8,
    'climate': 'Tropical',
    'soil': 'Loamy'
}

# Calculate score for the first tree in the dataframe
first_tree = df.iloc[0]
example_score = calculate_suitability_score(example_input, first_tree)

print(f"\nExample Suitability Score for '{first_tree['treename']}': {example_score}")

# Calculate score for the second tree
second_tree = df.iloc[1]
example_score_2 = calculate_suitability_score(example_input, second_tree)
print(f"Example Suitability Score for '{second_tree['treename']}': {example_score_2}")

# Example with potentially unsuitable input
example_input_unsuitable = {
    'temp': 5,       # Low temp
    'ele': 3000,     # High elevation
    'ph': 4.0,       # Acidic
    'climate': 'Alpine', # Check if 'Alpine' is a class, might be unsuitable for first tree
    'soil': 'Saline'    # Check if 'Saline' is a class, might be unsuitable
}

# Check if 'Alpine' and 'Saline' are valid classes first
if 'Alpine' not in climate_classes: print("Note: 'Alpine' climate not in data.")
if 'Saline' not in soil_classes: print("Note: 'Saline' soil not in data.")

example_score_unsuitable = calculate_suitability_score(example_input_unsuitable, first_tree)
print(f"Example Suitability Score (unsuitable conditions) for '{first_tree['treename']}': {example_score_unsuitable}")

def calculate_normalized_suitability(user_input, tree_row, climate_classes, soil_classes):
    """
    Calculates a normalized suitability score (0-1) for a tree based on user input.
    Handles potential invalid climate/soil inputs more robustly for data generation.
    """
    score = 0
    max_possible_score = 5 # Temp, Ele, pH, Climate, Soil

    # --- Numerical Checks ---
    # Use soft checks (optional, but can help learning) - e.g., slightly outside range still gets some score
    # For simplicity now, let's stick to hard boundaries
    if tree_row['min-temp'] <= user_input['temp'] <= tree_row['max-temp']:
        score += 1
    if tree_row['min-ele'] <= user_input['ele'] <= tree_row['max-ele']:
        score += 1
    if tree_row['ph_min'] <= user_input['ph'] <= tree_row['ph_max']:
        score += 1

    # --- Categorical Checks ---
    input_climate = user_input['climate']
    if input_climate in climate_classes and tree_row[input_climate] == 1:
            score += 1
    # else: score remains unchanged if climate doesn't match or is invalid

    input_soil = user_input['soil']
    if input_soil in soil_classes and tree_row[input_soil] == 1:
            score += 1
    # else: score remains unchanged if soil doesn't match or is invalid

    normalized_score = score / max_possible_score
    return normalized_score

# Example: Re-run with the first tree and previous example input
# Make sure climate_classes and soil_classes are correctly defined from previous steps
# (You might need to re-run the MultiLabelBinarizer fitting if the kernel restarted)
# example_input = {'temp': 25, 'ele': 500, 'ph': 6.8, 'climate': 'Tropical', 'soil': 'Loamy'}
# first_tree = df.iloc[0]
# norm_score = calculate_normalized_suitability(example_input, first_tree, climate_classes, soil_classes)
# print(f"Normalized Score for '{first_tree['treename']}': {norm_score}") # Should be 1.0 if all conditions match

def create_nn_feature_vector(user_input, tree_row, climate_classes, soil_classes):
    """
    Creates a feature vector for the NN, combining user input and tree characteristics.
    """
    features = []

    # --- Temperature Features ---
    temp_in_range = 1.0 if tree_row['min-temp'] <= user_input['temp'] <= tree_row['max-temp'] else 0.0
    # How far below min (0 if above min)
    temp_below_min = max(0, tree_row['min-temp'] - user_input['temp'])
    # How far above max (0 if below max)
    temp_above_max = max(0, user_input['temp'] - tree_row['max-temp'])
    # Width of the tree's tolerance range
    temp_range_width = tree_row['max-temp'] - tree_row['min-temp']

    features.extend([
        user_input['temp'], # User's raw input temp
        temp_in_range,
        temp_below_min,
        temp_above_max,
        temp_range_width
        # Optional: tree_row['min-temp'], tree_row['max-temp'] (can sometimes help)
    ])

    # --- Elevation Features (Similar Logic) ---
    ele_in_range = 1.0 if tree_row['min-ele'] <= user_input['ele'] <= tree_row['max-ele'] else 0.0
    ele_below_min = max(0, tree_row['min-ele'] - user_input['ele'])
    ele_above_max = max(0, user_input['ele'] - tree_row['max-ele'])
    ele_range_width = tree_row['max-ele'] - tree_row['min-ele']

    features.extend([
        user_input['ele'], # User's raw input ele
        ele_in_range,
        ele_below_min,
        ele_above_max,
        ele_range_width
        # Optional: tree_row['min-ele'], tree_row['max-ele']
    ])

    # --- pH Features (Similar Logic) ---
    ph_in_range = 1.0 if tree_row['ph_min'] <= user_input['ph'] <= tree_row['ph_max'] else 0.0
    ph_below_min = max(0, tree_row['ph_min'] - user_input['ph'])
    ph_above_max = max(0, user_input['ph'] - tree_row['ph_max'])
    ph_range_width = tree_row['ph_max'] - tree_row['ph_min']

    features.extend([
        user_input['ph'], # User's raw input pH
        ph_in_range,
        ph_below_min,
        ph_above_max,
        ph_range_width
        # Optional: tree_row['ph_min'], tree_row['ph_max']
    ])

    # --- Climate Features ---
    climate_match = 0.0
    encoded_climate = np.zeros(len(climate_classes)) # Vector representing input climate
    if user_input['climate'] in climate_classes:
        climate_match = 1.0 if tree_row[user_input['climate']] == 1 else 0.0
        climate_index = climate_classes.index(user_input['climate'])
        encoded_climate[climate_index] = 1.0 # One-hot encode the user's input climate

    features.append(climate_match)
    # features.extend(encoded_climate) # Option 1: Add one-hot encoding of input climate
    # Option 2: Include the tree's climate preferences vector directly? Maybe too much?
    # features.extend(tree_row[climate_classes].values)

    # --- Soil Features ---
    soil_match = 0.0
    encoded_soil = np.zeros(len(soil_classes)) # Vector representing input soil
    if user_input['soil'] in soil_classes:
        soil_match = 1.0 if tree_row[user_input['soil']] == 1 else 0.0
        soil_index = soil_classes.index(user_input['soil'])
        encoded_soil[soil_index] = 1.0 # One-hot encode the user's input soil

    features.append(soil_match)
    # features.extend(encoded_soil) # Option 1: Add one-hot encoding of input soil
    # Option 2: Include the tree's soil preferences vector directly?
    # features.extend(tree_row[soil_classes].values)


    return np.array(features)

# Example: Create feature vector for the first tree and example input
# nn_features = create_nn_feature_vector(example_input, first_tree, climate_classes, soil_classes)
# print(f"\nNN Feature Vector for '{first_tree['treename']}' (length {len(nn_features)}):")
# print(nn_features)



def generate_training_data(df, climate_classes, soil_classes, num_samples_per_tree=20):
    """
    Generates training data (X, y) by simulating user inputs for each tree.
    """
    X_train = []
    y_train = []

    all_climates = climate_classes
    all_soils = soil_classes

    for index, tree_row in df.iterrows():
        for _ in range(num_samples_per_tree):
            # Strategy: Generate some inputs likely to be good, some bad, some mixed

            # --- Generate simulated user input ---
            simulated_input = {}

            # Temperature: Sample around the tree's range
            temp_range = tree_row['max-temp'] - tree_row['min-temp']
            # Bias towards sampling within the range, but allow outside values
            simulated_input['temp'] = random.uniform(tree_row['min-temp'] - temp_range * 0.3,
                                                     tree_row['max-temp'] + temp_range * 0.3)

            # Elevation: Sample around the tree's range
            ele_range = tree_row['max-ele'] - tree_row['min-ele']
            # Handle cases where min/max elevation are the same or range is huge
            if ele_range <= 0: ele_range = 100 # Assign a nominal range if min=max
            ele_range = min(ele_range, 4000) # Cap range influence to avoid extreme sampling
            simulated_input['ele'] = random.uniform(max(0, tree_row['min-ele'] - ele_range * 0.3), # Elevation >= 0
                                                     tree_row['max-ele'] + ele_range * 0.3)

            # pH: Sample around the tree's range
            ph_range = tree_row['ph_max'] - tree_row['ph_min']
            simulated_input['ph'] = random.uniform(max(0, tree_row['ph_min'] - ph_range * 0.5),
                                                    min(14, tree_row['ph_max'] + ph_range * 0.5)) # pH between 0 and 14

            # Climate & Soil: Higher chance of picking a suitable one, but allow others
            tree_suitable_climates = [c for c in all_climates if tree_row[c] == 1]
            tree_suitable_soils = [s for s in all_soils if tree_row[s] == 1]

            if random.random() < 0.7 and tree_suitable_climates: # 70% chance to pick a suitable climate (if available)
                simulated_input['climate'] = random.choice(tree_suitable_climates)
            else: # Otherwise pick any climate
                simulated_input['climate'] = random.choice(all_climates)

            if random.random() < 0.7 and tree_suitable_soils: # 70% chance to pick a suitable soil (if available)
                simulated_input['soil'] = random.choice(tree_suitable_soils)
            else: # Otherwise pick any soil
                simulated_input['soil'] = random.choice(all_soils)


            # --- Create feature vector and calculate target score ---
            feature_vector = create_nn_feature_vector(simulated_input, tree_row, climate_classes, soil_classes)
            target_score = calculate_normalized_suitability(simulated_input, tree_row, climate_classes, soil_classes)

            X_train.append(feature_vector)
            y_train.append(target_score)

    return np.array(X_train), np.array(y_train)

# Generate the data (might take a moment)
# Start with fewer samples per tree for faster testing, e.g., 10 or 20
X_data, y_data = generate_training_data(df, climate_classes, soil_classes, num_samples_per_tree=30)

print(f"Generated training data shape: X = {X_data.shape}, y = {y_data.shape}")
# Check distribution of scores (optional)

plt.hist(y_data, bins=10)
plt.title('Distribution of Generated Suitability Scores (y_data)')
plt.xlabel('Normalized Suitability Score')
plt.ylabel('Frequency')
plt.show()


# 1. Split Data
# test_size=0.2 means 20% of the data will be used for testing, 80% for training
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print("Data Split Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# 2. Scale Features
# Initialize the scaler
# scaler = StandardScaler()
scaler = MinMaxScaler() # Now this line should work

# Fit the scaler ONLY on the training data
scaler.fit(X_train)

# Transform both training and testing data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the scaling results (optional)
print("\nScaled Data Example (first 5 rows of X_train_scaled):")
print(X_train_scaled[:5])
print("\nMin values in scaled train data (should be close to 0 for MinMaxScaler):", np.min(X_train_scaled, axis=0))
print("Max values in scaled train data (should be close to 1 for MinMaxScaler):", np.max(X_train_scaled, axis=0))
# Check test set scaling too
print("\nMin values in scaled test data (might not be exactly 0):", np.min(X_test_scaled, axis=0))
print("Max values in scaled test data (might not be exactly 1):", np.max(X_test_scaled, axis=0))


# Optional: for adding dropout regularization
# from tensorflow.keras.layers import Dropout

# --- 1. Define Model Architecture ---

# Get the number of input features from the scaled training data shape
input_dim = X_train_scaled.shape[1]

model = keras.Sequential(
    [
        # Input layer: Specify input shape for the first layer
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        # Optional: Add Dropout for regularization (helps prevent overfitting)
        # layers.Dropout(0.3), # Drop 30% of neurons randomly during training
        layers.Dense(32, activation="relu"),
        # layers.Dropout(0.2), # Optional dropout
        layers.Dense(16, activation="relu"),
        # Output layer: 1 neuron (predicting suitability score)
        # Use 'sigmoid' activation because our target (y_train) is normalized between 0 and 1
        layers.Dense(1, activation="sigmoid"),
    ]
)

# --- 2. Compile the Model ---

# Optimizer: Controls how the model updates its weights (Adam is a common, effective choice)
# Loss function: Measures how wrong the model's predictions are.
#   'mean_squared_error' (MSE) is standard for regression.
#   'mean_absolute_error' (MAE) is another option, less sensitive to outliers.
# Metrics: Used to monitor training and testing steps. We'll track MAE.
model.compile(optimizer='adam',
              loss='mean_squared_error', # or 'mean_absolute_error'
              metrics=['mae']) # Mean Absolute Error

# --- 3. Display Model Summary ---
print("Model Architecture Summary:")
model.summary()

# --- 4. Train the Model ---

print("\nStarting Model Training...")
# epochs: Number of times the model sees the entire training dataset
# batch_size: Number of samples processed before the model's weights are updated
# validation_split: Use a portion of the training data as a validation set during training
#                   to monitor performance on data not used for weight updates in that epoch.
history = model.fit(X_train_scaled,
                    y_train,
                    epochs=50,  # Start with 50-100, can increase if needed
                    batch_size=32,
                    validation_split=0.2, # Use 20% of training data for validation
                    verbose=1) # Set to 1 to see progress per epoch, 0 for silent

print("\nModel Training Finished.")



# --- 5. Evaluate the Model on Test Data ---

print("\nEvaluating Model Performance on Test Set:")
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Set Mean Squared Error (Loss): {loss:.4f}")
print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")

# MAE gives a sense of the average error in our predicted suitability score (0-1 scale)
# e.g., MAE = 0.05 means the predictions are off by +/- 0.05 on average.


# Ensure these are accessible:
# - df: Your original DataFrame with tree data and encoded categories
# - model: Your trained Keras model
# - scaler: Your fitted MinMaxScaler
# - climate_classes: List of climate column names used in feature engineering
# - soil_classes: List of soil column names used in feature engineering
# - create_nn_feature_vector: The function defined earlier

#--- 6. Save the Model (Recommended) ---
model.save('modelv1.keras')
print("\nModel saved as modelv1.keras")

def suggest_trees_nn(user_input, df_trees, model, scaler, climate_classes, soil_classes, top_n=5):
    """
    Suggests top_n trees based on user input using the trained NN model.

    Args:
        user_input (dict): Dictionary with keys 'temp', 'ele', 'ph', 'climate', 'soil'.
        df_trees (pd.DataFrame): The DataFrame containing all tree data.
        model (keras.Model): The trained Keras model.
        scaler (sklearn.preprocessing.Scaler): The fitted scaler.
        climate_classes (list): List of climate category names.
        soil_classes (list): List of soil category names.
        top_n (int): The number of top trees to suggest.

    Returns:
        list: A list of tuples, where each tuple is (treename, predicted_score),
              sorted by score in descending order. Returns empty list on error.
    """
    # --- Input Validation (Basic) ---
    required_keys = {'temp', 'ele', 'ph', 'climate', 'soil'}
    if not required_keys.issubset(user_input.keys()):
        print("Error: User input missing required keys:", required_keys - user_input.keys())
        return []

    input_climate = user_input['climate']
    if input_climate not in climate_classes:
        print(f"Warning: Input climate '{input_climate}' not in known classes. Predictions might be less accurate.")
        # Depending on strictness, you could return [] here

    input_soil = user_input['soil']
    if input_soil not in soil_classes:
        print(f"Warning: Input soil type '{input_soil}' not in known classes. Predictions might be less accurate.")
        # Depending on strictness, you could return [] here

    # --- Prediction Loop ---
    predictions = []
    tree_names = df_trees['treename'].tolist() # Get tree names for pairing later

    # Prepare feature vectors for all trees efficiently
    feature_vectors = []
    valid_tree_indices = []
    for index, tree_row in df_trees.iterrows():
        try:
            # 1. Create feature vector for this tree and user input
            feature_vec = create_nn_feature_vector(user_input, tree_row, climate_classes, soil_classes)
            feature_vectors.append(feature_vec)
            valid_tree_indices.append(index) # Keep track of which trees we generated features for
        except Exception as e:
            print(f"Error generating features for tree {tree_row.get('treename', index)}: {e}")


    if not feature_vectors:
        print("Error: Could not generate feature vectors for any tree.")
        return []

    # Convert list of vectors to a NumPy array for batch processing
    feature_vectors_np = np.array(feature_vectors)

    # 2. Scale the feature vectors using the PRE-FITTED scaler
    feature_vectors_scaled = scaler.transform(feature_vectors_np)

    # 3. Predict scores using the NN model (predict in batch)
    predicted_scores = model.predict(feature_vectors_scaled, verbose=0) # verbose=0 for silent prediction

    # --- Combine results and Rank ---
    results = []
    original_df_indices = df_trees.index[df_trees.index.isin(valid_tree_indices)] # Get original indices corresponding to vectors

    for i, original_index in enumerate(original_df_indices):
        tree_name = df_trees.loc[original_index, 'treename']
        score = predicted_scores[i][0] # Prediction output is often [[score]], so get [i][0]
        results.append((tree_name, score))


    # 4. Sort by score (descending)
    results.sort(key=lambda item: item[1], reverse=True)

    # 5. Return the top N results
    return results[:top_n]

# --- Example Usage ---
print("\n--- Testing the Prediction Function ---")

# Use the same example input from before
example_input = {
    'temp': 25,
    'ele': 500,
    'ph': 6.8,
    'climate': 'Tropical',
    'soil': 'Loamy'
}

recommended_trees = suggest_trees_nn(example_input, df, model, scaler, climate_classes, soil_classes, top_n=20)

print(f"\nTop 5 recommended trees for input {example_input}:")
if recommended_trees:
    for name, score in recommended_trees:
        print(f"- {name} (Predicted Suitability: {score:.4f})")
else:
    print("No recommendations could be made.")

# Example with potentially less suitable conditions
example_input_2 = {
    'temp': 12,
    'ele': 1800,
    'ph': 5.2,
    'climate': 'Temperate', # Ensure 'Temperate' is in your climate_classes
    'soil': 'Sandy'       # Ensure 'Sandy' is in your soil_classes
}

# Check if 'Temperate' exists before running if unsure
if 'Temperate' not in climate_classes: print("Warning: 'Temperate' not in climate classes list!")

recommended_trees_2 = suggest_trees_nn(example_input_2, df, model, scaler, climate_classes, soil_classes, top_n=20)

print(f"\nTop 5 recommended trees for input {example_input_2}:")
if recommended_trees_2:
    for name, score in recommended_trees_2:
        print(f"- {name} (Predicted Suitability: {score:.4f})")
else:
    print("No recommendations could be made.")

print("\n--- Neem Targeted ---")

# Use the same example input from before
example_input = {
    'temp': 26,
    'ele': 500,
    'ph': 6.7,
    'climate': 'Subtropical',
    'soil': 'Alkaine'
}

recommended_trees = suggest_trees_nn(example_input, df, model, scaler, climate_classes, soil_classes, top_n=20)

print(f"\nTop 5 recommended trees for input {example_input}:")
if recommended_trees:
    for name, score in recommended_trees:
        print(f"- {name} (Predicted Suitability: {score:.4f})")
else:
    print("No recommendations could be made.")