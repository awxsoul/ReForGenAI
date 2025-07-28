# Image to Plan
import cv2
import numpy as np
import os
import random

def get():
    # Define file paths
    roadmap_file = 'static/images/userlocationmap/roadmap.png'
    satellite_file = 'static/images/userlocationmap/satellite.png'
    result_file = 'static/images/Results/result.png'
    covered_mask_file = 'static/images/Results/covered_mask.png'
    tree_mask_file = 'static/images/Results/tree.png'
    water_mask_file = 'static/images/Results/water.png'
    building_mask_file = 'static/images/Results/building.png'
    road_mask_file = 'static/images/Results/road.png'

    out_fold="static/images/Results"
    # Create the output folder if it doesn't exist
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
        print(f"Created directory: {out_fold}")
    else:
        print(f"Directory already exists: {out_fold}")

    print("Loading images...")
    roadmap_img = cv2.imread(roadmap_file)
    satellite_img = cv2.imread(satellite_file)

    # Basic error check
    if roadmap_img is None:
        print(f"Error: Could not load roadmap image at '{roadmap_file}'")
        exit()
    if satellite_img is None:
        print(f"Error: Could not load satellite image at '{satellite_file}'")
        exit()

    # Check if images have the same dimensions
    if roadmap_img.shape[:2] != satellite_img.shape[:2]:
        print("Error: Roadmap and Satellite images have different dimensions!")
        exit()

    height, width = roadmap_img.shape[:2]
    total_pixels = height * width
    print(f"Images loaded successfully. Dimensions: {width}x{height}")

    # --- Step 2: Prepare Base Result Image (Grayscale Roadmap) ---
    print("Creating base grayscale result image...")
    gray_roadmap = cv2.cvtColor(roadmap_img, cv2.COLOR_BGR2GRAY)
    result_image = cv2.cvtColor(gray_roadmap, cv2.COLOR_GRAY2BGR) # Keep 3 channels for coloring
    print("Base result image created.")

    # --- Step 3: Initialize Coverage Masks ---
    # 0 Black not covered, 225 White covered 
    covered_mask = np.zeros((height, width), dtype=np.uint8)

    # --- Step 4: Identify Water from Roadmap ---
    print("Identifying water from roadmap image...")
    lower_water_blue = np.array([180, 180, 0])   # Lower bound for BGR
    upper_water_blue = np.array([255, 255, 180]) # Upper bound for BGR

    # Create a mask for water pixels
    water_mask = cv2.inRange(roadmap_img, lower_water_blue, upper_water_blue)
    print("Water mask created.")

    covered_mask = cv2.bitwise_or(covered_mask, water_mask)
    print(f"'covered_mask' updated. Current covered pixels: {cv2.countNonZero(covered_mask)}")

    print("Identifying trees from satellite image...")
    hsv_satellite = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2HSV)
    # Define approximate range for green color in HSV (same as before)
    lower_green = np.array([40, 30, 20])
    upper_green = np.array([160, 100, 90])

    tree_mask = cv2.inRange(hsv_satellite, lower_green, upper_green)
    print("Initial tree mask created.")
    inverse_water_mask = cv2.bitwise_not(water_mask)
    tree_mask = cv2.bitwise_and(tree_mask, inverse_water_mask)

    covered_mask = cv2.bitwise_or(covered_mask, tree_mask)

    print("Identifying buildings from roadmap image...")
    # Buildings in the roadmap are typically light grey blocks.
    lower_building_gray = np.array([230, 230, 230]) # Looser lower bound
    upper_building_gray = np.array([245, 245, 245]) # Upper bound

    lower_yellow = np.array([230, 210, 210])  
    upper_yellow = np.array([240, 250, 255])  

    # Create a mask for yellow pixels
    yellow_mask = cv2.inRange(roadmap_img, lower_yellow, upper_yellow)
    building_mask = cv2.inRange(roadmap_img, lower_building_gray, upper_building_gray)
    building_mask = cv2.bitwise_or(building_mask, yellow_mask)

    print("Building mask created.")

    covered_mask = cv2.bitwise_or(covered_mask, building_mask)

    print("Identifying roads from roadmap image...")
    lower_road = np.array([200, 200, 200])   # Lower bound for BGR
    upper_road = np.array([255, 255, 255]) # Upper bound for BGR

    # Create a mask for roads
    road_mask = cv2.inRange(roadmap_img, lower_road, upper_road)
    road_mask = cv2.bitwise_not(road_mask)
    road_mask = cv2.bitwise_and(road_mask,inverse_water_mask)
    # print("Road mask created.")

    covered_mask = cv2.bitwise_or(covered_mask, road_mask)

    #grouping tree spreads
    kernel_size = 9 # Experiment with values like 5, 7, 9, 11, 15 etc.
    iterations = 1

    _, tree_mask_binary = cv2.threshold(tree_mask, 1, 255, cv2.THRESH_BINARY)
    print("Mask loaded and ensured binary.")
    print(f"Performing morphological closing with kernel size {kernel_size}x{kernel_size}...")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    tree_mask = cv2.morphologyEx(tree_mask_binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    covered_mask = cv2.bitwise_or(covered_mask, tree_mask)
    print("Closing operation complete.")

    # --- Step 6: Calculate Coverage ---
    print("Calculating water coverage...")
    water_pixels = cv2.countNonZero(water_mask)
    water_percentage = (water_pixels / total_pixels) * 100
    print(f"Water Coverage: {water_pixels} pixels")
    print(f"Water Coverage Percentage: {water_percentage:.2f}%")

    # Calculate refined tree coverage
    refined_tree_pixels = cv2.countNonZero(tree_mask)
    refined_tree_percentage = (refined_tree_pixels / total_pixels) * 100
    print(f"Refined Tree Coverage (excluding water overlap): {refined_tree_pixels} pixels")
    print(f"Refined Tree Coverage Percentage: {refined_tree_percentage:.2f}%")

    # Calculate refined building coverage
    building_pixels = cv2.countNonZero(building_mask)
    building_percentage = (building_pixels / total_pixels) * 100
    print(f"Refined Building Coverage (excluding water overlap): {building_pixels} pixels")
    print(f"Refined Building Coverage Percentage: {building_percentage:.2f}%")

    # Calculate refined building coverage
    road_pixels = cv2.countNonZero(road_mask)
    road_percentage = (road_pixels / total_pixels) * 100
    print(f"Road Coverage (excluding water overlap): {road_pixels} pixels")
    print(f"Road Coverage Percentage: {road_percentage:.2f}%")

    print("\nCalculating final coverage percentages...")
    # Coverage including Water (Trees + Buildings + Roads + Water)
    final_covered_pixels = cv2.countNonZero(covered_mask)
    final_covered_percentage = (final_covered_pixels / total_pixels) * 100
    print(f"Total Covered Area (including water): {final_covered_pixels} pixels")
    print(f"Total Coverage Percentage (including water): {final_covered_percentage:.2f}%")

    cv2.imwrite(water_mask_file, water_mask) # Save the water mask for inspection
    print(f"Water mask saved to '{water_mask_file}'")

    cv2.imwrite(tree_mask_file, tree_mask)
    print(f"Tree mask saved to '{tree_mask_file}'")

    cv2.imwrite(building_mask_file, building_mask)
    print(f"Building mask saved to '{building_mask_file}'")

    cv2.imwrite(road_mask_file, road_mask)
    print(f"Road mask saved to '{road_mask_file}'")

    cv2.imwrite(covered_mask_file, covered_mask)
    print(f"Covered mask saved to '{covered_mask_file}'")

    # --- Step 9: Mark Refined Trees in Red on Result Image ---
    print("Marking refined trees in red on the result image...")
    tree_color = [34,139,34] # BGR
    result_image[tree_mask == 255] = tree_color
    print("Refined trees marked.")

    # --- Step 11: Save Intermediate Results ---
    print(f"Saving result image with trees (water excluded) to '{result_file}'...")
    cv2.imwrite(result_file, result_image)

    result_mask = cv2.bitwise_not(covered_mask)
    cv2.imwrite(result_file, result_mask)

    # --- Configuration ---
    # Input mask file (black and white image where white is the target area)
    # Make sure this path is correct
    input_mask_file = 'static/images/Results/result.png' # Use the mask you provided
    # Output file to save the image with points
    output_image_file = 'static/images/Results/check_result.png'
    # Optional: Load an original image (like roadmap.png) to draw points onto
    # Set to None to draw points directly on a BGR version of the mask
    base_image_file = 'static/images/userlocationmap/satellite.png' # Or 'satellite.png', or None

    # Number of random points to generate
    num_points = 35  

    # Point appearance
    point_color_bgr = [0, 0, 255]  # Red (BGR format)
    point_radius = 10          # Size of the points (pixels)
    point_thickness = -1       # -1 fills the circle

    # --- Check if Mask File Exists ---
    if not os.path.exists(input_mask_file):
        print(f"ERROR: Input mask file not found at path: '{input_mask_file}'")
        exit()
    else:
        print(f"Input mask file found at: '{input_mask_file}'")

    # --- Load the Mask ---
    print(f"Loading mask: {input_mask_file}")
    mask = cv2.imread(input_mask_file, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"ERROR: cv2.imread failed to load the mask image.")
        exit()
    else:
        print(f"Mask loaded successfully. Shape: {mask.shape}")

    # --- Ensure mask is binary (0 or 255) ---
    _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    print("Mask ensured binary.")

    # --- Find Coordinates of all White Pixels ---
    # np.nonzero returns a tuple of arrays (rows, columns) for non-zero elements
    white_pixel_indices = np.nonzero(mask_binary)

    # Check if there are any white pixels
    if len(white_pixel_indices[0]) == 0:
        print("ERROR: No white pixels found in the mask. Cannot place points.")
        exit()

    # Convert the indices into a list of (x, y) coordinates
    # Note the order: np.nonzero gives (row, col), we need (col, row) for OpenCV points (x, y)
    white_pixel_coords = list(zip(white_pixel_indices[1], white_pixel_indices[0]))
    num_available_points = len(white_pixel_coords)
    print(f"Found {num_available_points} possible locations (white pixels).")

    # --- Check if requested points exceed available points ---
    if num_points > num_available_points:
        print(f"Warning: Requested {num_points} points, but only {num_available_points} white pixels exist.")
        print(f"         Placing points on all available white pixels instead.")
        num_points = num_available_points
        selected_coords = white_pixel_coords # Use all available points
    else:
        # --- Randomly Select Coordinates ---
        print(f"Randomly selecting {num_points} locations...")
        # Use random.sample to pick unique coordinates without replacement
        selected_coords = random.sample(white_pixel_coords, num_points)
        print(f"Selected {len(selected_coords)} unique coordinates.")


    # --- Prepare Output Image ---
    if base_image_file and os.path.exists(base_image_file):
        print(f"Loading base image: {base_image_file}")
        output_image = cv2.imread(base_image_file)
        if output_image is None:
            print(f"Warning: Failed to load base image '{base_image_file}'. Drawing on mask instead.")
            output_image = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR) # Create BGR mask
        elif output_image.shape[:2] != mask.shape[:2]:
            print(f"Warning: Base image dimensions {output_image.shape[:2]} differ from mask dimensions {mask.shape[:2]}. Drawing on mask instead.")
            output_image = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR) # Create BGR mask
        else:
            print("Using loaded base image.")
    else:
        if base_image_file:
            print(f"Warning: Base image file '{base_image_file}' not found. Drawing on mask instead.")
        else:
            print("No base image specified. Drawing on mask.")
        # Create a BGR version of the mask to draw color points on
        output_image = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)


    # --- Draw Points on Output Image ---
    print("Drawing points on the output image...")
    for point in selected_coords:
        # 'point' is already an (x, y) tuple from our earlier zip
        cv2.circle(output_image, point, point_radius, point_color_bgr, point_thickness)

    print("Drawing complete.")

    # --- Save the Result ---
    print(f"Saving image with points to: {output_image_file}")
    cv2.imwrite(output_image_file, output_image)

    print("Done. Check the output file.")
