import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf 

def format(template_data):    
    input={
        'temp': template_data['temperature_celsius'],
        'ele': template_data['elevation_meters'],
        'ph': template_data['soil_ph'],
        'climate': template_data['climate_classes'][0],
        'soil': template_data['soil_type_classes'][0]
    }
    return input

def fetch(pickle_filename):
    if os.path.exists(pickle_filename):
        try:
            # Open the file in binary read mode ('rb')
            with open(pickle_filename, 'rb') as file:
                # Use pickle.load() to deserialize the object from the file
                loaded_df = pickle.load(file)
            print(f"DataFrame successfully loaded from '{pickle_filename}'")
            return loaded_df
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
    else:
        print(f"Error: File '{pickle_filename}' not found. Cannot load.")

def create_nn_feature_vector(user_input, tree_row, climate_classes, soil_classes):

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

def trees(user_input, top_n=5):

    climate_classes=['Alpine', 'Arid', 'Coastal', 'Cool temperate', 'Mediterranean', 'Semi-arid', 'Subtropical', 'Temperate', 'Tropical']
    soil_classes=['Acidic', 'Alluvial', 'Clay', 'Clay loam', 'Dry', 'Loamy', 'Moist', 'Poor', 'Poorly drained', 'Sandy', 'Sandy loam', 'Swampy', 'Well-drained', 'Wet']

    df_trees=fetch("model/dftopredict.pkl")
    scaler=fetch("model/scaler.pkl")

    model = tf.keras.models.load_model("model/modelv1.keras")

    required_keys = {'temp', 'ele', 'ph', 'climate', 'soil'}
    if not required_keys.issubset(user_input.keys()):
        print("Error: User input missing required keys:", required_keys - user_input.keys())
        return []
    input_climate = user_input['climate']
    if input_climate not in climate_classes:
        print(f"Warning: Input climate '{input_climate}' not in known classes. Predictions might be less accurate.")

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
