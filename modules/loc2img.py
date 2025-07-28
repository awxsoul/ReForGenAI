# modules/loc2img.py
from flask import Flask, render_template, request, redirect, url_for
import ee
import requests  # Library to make HTTP requests
import os 

def activateauth():
   try:
      ee.Initialize(project="reforestationai",opt_url='https://earthengine-highvolume.googleapis.com')
   except Exception as e:
      print(f"Earth Engine initialization failed: {e}")
      print("Attempting authentication...")
      ee.Authenticate()
      ee.Initialize(project="reforestationai",opt_url='https://earthengine-highvolume.googleapis.com')
    
def getinput():
   if request.method == 'POST':
      l1 = request.form.get('latitude')
      l2 = request.form.get('longitude')
      return float(l1),float(l2)
    
def dwdimg(latitude, longitude):
   zoom = 18
   size = "600x600"
   scale = 2
   style = "feature:all|element:labels|visibility:off" 
   base_url = "https://maps.googleapis.com/maps/api/staticmap"
   output_folder = "static/images/userlocationmap"
   GOOGLE_MAPS_API_KEY="AIzaSyDG8qX9u83gOZduHZz3rpKe4B8rz86E69U"

   # Map types and filenames to generate
   map_configs = [
      {"type": "roadmap", "filename": "roadmap.png"},
      {"type": "satellite", "filename": "satellite.png"}
   ]

   # Create the output folder if it doesn't exist
   if not os.path.exists(output_folder):
      os.makedirs(output_folder)
      print(f"Created directory: {output_folder}")
   else:
      print(f"Directory already exists: {output_folder}")

   # Loop through map configurations and fetch images
   for config in map_configs:
      map_type = config["type"]
      output_filename = config["filename"]
      file_path = os.path.join(output_folder, output_filename) # Construct full file path

      # Construct the parameters for the API request
      params = {
         'center': f"{latitude},{longitude}",
         'zoom': zoom,
         'size': size,
         'scale': scale,
         'maptype': map_type,
         'style': style,
         'key': GOOGLE_MAPS_API_KEY
      }

      print(f"\nFetching {map_type} map for {latitude},{longitude}...")

      try:
         # Make the GET request to the Google Maps API
         response = requests.get(base_url, params=params)

         # Check if the request was successful (status code 200)
         response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)

         # Save the image content (binary data) to the file
         with open(file_path, 'wb') as f:
               f.write(response.content)
         print(f"Successfully saved: {file_path}")

      except requests.exceptions.RequestException as e:
         # Handle potential errors during the request (network issues, bad status codes)
         print(f"Error fetching {map_type} map: {e}")
         # Print more details if available (like API error messages)
         if hasattr(e, 'response') and e.response is not None:
               # Check if response content is text (often contains error details)
               content_type = e.response.headers.get('Content-Type', '')
               if 'text' in content_type or 'json' in content_type:
                  print(f"Response Status Code: {e.response.status_code}")
                  print(f"Response Text: {e.response.text}")
               else:
                  print(f"Response Status Code: {e.response.status_code} (Binary or unknown content type)")
         print(f"Failed to download {map_type} map. Check your API key, quota, and parameters.")

      except Exception as e:
         # Handle other potential errors (e.g., file system errors)
         print(f"An unexpected error occurred while processing {map_type} map: {e}")


   print(f"\nFinished. Check the '{output_folder}' folder in the Colab file browser.")
