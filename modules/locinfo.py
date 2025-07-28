# modules/locinfo.py
import ee
import sys

def get(latitude, longitude):

    location_point = ee.Geometry.Point([longitude, latitude])
    print(f"Target Location: Latitude={latitude}, Longitude={longitude}")

    location_details = {
    'process':"Results of provided location",
    'latitude': latitude,
    'longitude': longitude,
    'radius': 173.35, 
    'temperature_celsius': None,
    'elevation_meters': None,
    'soil_ph': None,
    'soil_type_classes': [], 
    'climate_classes': []    
    }

    # ------------------------------------------------------
    print("Fetching Elevation (using NASADEM)...")
    try:
        dem_dataset = ee.Image('NASA/NASADEM_HGT/001') 
        elevation_band = dem_dataset.select('elevation')
        elevation_info = elevation_band.reduceRegion(
            reducer=ee.Reducer.firstNonNull(), 
            geometry=location_point,
            scale=30
        ).get('elevation') 
        elevation_value = elevation_info.getInfo() 
        if elevation_value is not None:
            location_details['elevation_meters'] = float(elevation_value) 
            print(f"  Elevation retrieved: {location_details['elevation_meters']:.2f} meters (using NASADEM)")
        else:
            print("  Elevation data not available for this location (using NASADEM).")
            location_details['elevation_meters'] = None
    except ee.EEException as e:
        print(f"  Error fetching elevation (NASADEM): {e}", file=sys.stderr)
        location_details['elevation_meters'] = None 
    except Exception as e:
        print(f"  An unexpected error occurred during elevation fetching (NASADEM): {e}", file=sys.stderr)
        location_details['elevation_meters'] = None 

    # ------------------------------------------------------
    print("\nFetching Temperature (using ERA5-Land for Jan-Nov 2024)...")
    try:
        temp_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        start_date_str = "2024-01-01T00:00:00" 
        end_date_str = "2024-12-01T00:00:00"  
        print(f"  Using date range: {start_date_str} to {end_date_str}")
        period_temp_images = temp_dataset.filterDate(
                start_date_str,
                end_date_str
            ).select('temperature_2m')
        image_count = period_temp_images.size().getInfo()
        if image_count > 0:
            print(f"  Found {image_count} images in the specified period.")
            mean_temp_image = period_temp_images.mean() 
            temp_info = mean_temp_image.reduceRegion(
                reducer=ee.Reducer.firstNonNull(), 
                geometry=location_point,
                scale=1000,
                tileScale=4 
            ).get('temperature_2m') 
            temp_value_k = temp_info.getInfo() 
            if temp_value_k is not None:
                temp_value_c = float(temp_value_k) - 273.15
                location_details['temperature_celsius'] = temp_value_c
                print(f"  Mean temperature ({start_date_str} to {end_date_str}) retrieved: {location_details['temperature_celsius']:.2f} °C (using ERA5-Land)")
            else:
                print("  Temperature data calculation resulted in null for this location/period (ERA5-Land).")
                location_details['temperature_celsius'] = None
        else:
            print(f"  No ERA5-Land temperature images found for the period {start_date_str} to {end_date_str}.")
            location_details['temperature_celsius'] = None
    except ee.EEException as e:
        print(f"  Error fetching temperature (ERA5-Land): {e}", file=sys.stderr)
        location_details['temperature_celsius'] = None 
    except Exception as e:
        print(f"  An unexpected error occurred during temperature fetching (ERA5-Land): {e}", file=sys.stderr)
        location_details['temperature_celsius'] = None 

    # ------------------------------------------------------
    print("\nFetching Soil pH (using OpenLandMap dataset from GEE)...")
    try:
        openlandmap_dataset = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
        print(f"  Attempting to use GEE Dataset ID: {openlandmap_dataset.id().getInfo()}")
        ph_band_olm = openlandmap_dataset.select('b0')
        ph_info_olm = ph_band_olm.reduceRegion(
            reducer=ee.Reducer.firstNonNull(), 
            geometry=location_point,
            scale=250 
        ).get('b0')
        ph_value_times_10_olm = ph_info_olm.getInfo() 
        if ph_value_times_10_olm is not None:
            actual_ph_value_olm = float(ph_value_times_10_olm) / 10.0
            location_details['soil_ph'] = actual_ph_value_olm
            print(f"  Soil pH (0cm) retrieved: {location_details['soil_ph']:.2f} (from GEE dataset OpenLandMap)")
        else:
            print("  Soil pH data not available for this location (from GEE dataset OpenLandMap).")
            location_details['soil_ph'] = None
    except ee.EEException as e:
        if 'Image.load' in str(e) and 'not found' in str(e):
            print(f"  Error fetching Soil pH: {e}", file=sys.stderr)
            print("  Failed to load OpenLandMap dataset as well. This indicates a persistent issue accessing certain GEE assets.", file=sys.stderr)
            print("  Consider checking your Google Cloud Project settings or trying from a different network/environment if possible.", file=sys.stderr)
        else:
            print(f"  Error fetching Soil pH from GEE dataset (OpenLandMap): {e}", file=sys.stderr)
        location_details['soil_ph'] = None 
    except Exception as e:
        print(f"  An unexpected error occurred during Soil pH fetching using OpenLandMap: {e}", file=sys.stderr)
        location_details['soil_ph'] = None 

    # ------------------------------------------------------------------------------------
    print("\nClassifying Climate (using WorldClim V1 BIO variables)...")
    print("  (Note: This is an approximation based on temperature/precipitation thresholds)")
    try:
        worldclim_bio = ee.Image('WORLDCLIM/V1/BIO')
        print(f"  Using GEE Dataset ID: {worldclim_bio.id().getInfo()}")
        bands_to_sample = ['bio01', 'bio10', 'bio11', 'bio12', 'bio14']
        bio_bands = worldclim_bio.select(bands_to_sample)
        bio_values_info = bio_bands.reduceRegion(
            reducer=ee.Reducer.firstNonNull(),
            geometry=location_point,
            scale=1000 
        )
        bio_values = bio_values_info.getInfo() 
        if not bio_values or any(bio_values.get(band) is None for band in bands_to_sample):
            print("  Could not retrieve all necessary WorldClim BIO variables for this location.")
            location_details['climate_classes'] = []
        else:
            annual_mean_temp = float(bio_values['bio01']) / 10.0
            warmest_q_temp = float(bio_values['bio10']) / 10.0
            coldest_q_temp = float(bio_values['bio11']) / 10.0
            annual_precip = float(bio_values['bio12'])
            driest_month_precip = float(bio_values['bio14'])
            print(f"  Retrieved WorldClim values: AnnMeanT={annual_mean_temp:.1f}C, WarmQ_T={warmest_q_temp:.1f}C, ColdQ_T={coldest_q_temp:.1f}C, AnnP={annual_precip:.0f}mm, DriestM_P={driest_month_precip:.0f}mm")
            assigned_classes = []
            arid_thresh_precip = 250
            semi_arid_thresh_precip = 500
            tropical_thresh_cold_temp = 18.0
            alpine_thresh_warm_temp = 10.0 
            mediterranean_thresh_dry_month = 30 
            cool_temperate_thresh_cold_temp = 0.0 
            if warmest_q_temp < alpine_thresh_warm_temp:
                assigned_classes.append('Alpine')
            else:
                if annual_precip < arid_thresh_precip:
                    assigned_classes.append('Arid')
                elif annual_precip < semi_arid_thresh_precip:
                    assigned_classes.append('Semi-arid')
                if not any(c in assigned_classes for c in ['Arid', 'Semi-arid']):
                    if coldest_q_temp >= tropical_thresh_cold_temp:
                        assigned_classes.append('Tropical')
                    else:
                        is_mediterranean_candidate = (driest_month_precip < mediterranean_thresh_dry_month and
                                                    coldest_q_temp >= cool_temperate_thresh_cold_temp)
                        if is_mediterranean_candidate:
                            if 'Tropical' not in assigned_classes:
                                assigned_classes.append('Mediterranean')
                        if not assigned_classes :
                            if coldest_q_temp < cool_temperate_thresh_cold_temp:
                                assigned_classes.append('Cool temperate')
                            else: 
                                if annual_mean_temp > 16.0: 
                                    assigned_classes.append('Subtropical')
                                else:
                                    assigned_classes.append('Temperate')
            if assigned_classes:
                location_details['climate_classes'] = sorted(list(set(assigned_classes)))
                print(f"  Assigned Climate Class(es) (Approximate): {location_details['climate_classes']}")
            else:
                print("  Could not classify climate based on WorldClim thresholds.")
                location_details['climate_classes'] = []

            print("  Note: 'Coastal' class cannot be reliably determined from this climate data alone.")
    except ee.EEException as e:
        if 'Image.load' in str(e) and 'WORLDCLIM/V1/BIO' in str(e):
            print(f"  CRITICAL Error: Failed to load core WorldClim dataset '{'WORLDCLIM/V1/BIO'}'. Check GEE status or connection.", file=sys.stderr)
        else:
            print(f"  Error fetching or processing WorldClim data: {e}", file=sys.stderr)
        location_details['climate_classes'] = []
    except Exception as e:
        print(f"  An unexpected error occurred during climate classification: {e}", file=sys.stderr)
        location_details['climate_classes'] = []

    # -------------------------------------------------
    print("\nAttempting Soil/Site Classification (Target List Focused)...")
    print("  (Using accessible datasets for the specified classes)")
    TARGET_CLASSES = [
        'Acidic', 'Alluvial', 'Clay', 'Clay loam', 'Dry', 'Loamy', 'Moist',
        'Poor', 'Poorly drained', 'Sandy', 'Sandy loam', 'Swampy',
        'Well-drained', 'Wet'
    ]
    print(f"  Target Classes: {TARGET_CLASSES}")
    location_details['soil_type_classes'] = []
    datasets_to_fetch = {
        # For Texture ('Clay', 'Clay loam', 'Loamy', 'Sandy', 'Sandy loam') -> Drainage
        "olm_texture": {"id": "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02", "band": "b0"},
        # For Fertility ('Poor')
        "olm_soc": {"id": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "band": "b0"},
        # For Alluvial proxy, slope calculation
        "nasadem": {"id": "NASA/NASADEM_HGT/001", "band": "elevation"},
        # For Moisture ('Dry', 'Moist')
        "worldclim_precip": {"id": "WORLDCLIM/V1/BIO", "band": "bio12"},
        # For Water Presence ('Swampy', 'Wet', 'Moist')
        "worldcover": {"id": "ESA/WorldCover/v200", "band": "Map", "type": "ImageCollection"},
        "jrc_water": {"id": "JRC/GSW1_4/GlobalSurfaceWater", "band": "occurrence"},
        # For Alluvial proxy
        "rivers": {"id": "WWF/HydroSHEDS/v1/FreeFlowingRivers", "band": "RIV_ORD", "type": "FeatureCollection"}
    }
    retrieved_values = {}
    fetch_success = {}

    print("\n--- Fetching Data for Classification ---")
    for key, info in datasets_to_fetch.items():
        try:
            asset_id = info["id"]
            band_name = info.get("band")
            asset_type = info.get("type", "Image")
            print(f"  Attempting to fetch {key}: {asset_id} [{band_name or 'N/A'}] (Type: {asset_type})")
            value = None
            if asset_type == "ImageCollection":
                collection = ee.ImageCollection(asset_id)
                if key == "worldcover":
                    image_from_collection = collection.first()
                    if image_from_collection is None: raise ee.EEException(f"Empty collection {asset_id}")
                    image_to_sample = image_from_collection.select(band_name)
                else: raise NotImplementedError(f"ImageCollection handling not defined for {key}")
            elif asset_type == "FeatureCollection":
                if key == "rivers":
                    rivers_collection = ee.FeatureCollection(asset_id)
                    point_buffer = location_point.buffer(50)
                    try:
                        count = rivers_collection.filterBounds(point_buffer).size().getInfo() 
                        value = 1.0 if count > 0 else 0.0
                        print(f"    River intersection check: Count = {count}")
                    except ee.EEException as river_err:
                        print(f"    -> GEE ERROR checking river intersection: {river_err}", file=sys.stderr)
                        fetch_success[key] = False; continue
                else: raise NotImplementedError(f"FeatureCollection handling not defined for {key}")
            else: 
                dataset = ee.Image(asset_id)
                image_to_sample = dataset.select(band_name)

            if value is None:
                scale = 30 if key == "nasadem" else (10 if key == "worldcover" else (30 if key == "jrc_water" else (1000 if key == "worldclim_precip" else 250)))
                reducer = ee.Reducer.mode() if key in ["worldcover"] else ee.Reducer.firstNonNull()
                value_info = image_to_sample.reduceRegion(reducer=reducer, geometry=location_point, scale=scale, tileScale=4).get(band_name)
                value = value_info.getInfo()

            if value is not None:
                retrieved_values[key] = float(value)
                fetch_success[key] = True
                print(f"    -> Success: Retrieved value: {retrieved_values[key]}")
            elif key != 'rivers':
                print(f"    -> Fail: Data null or unavailable for {key} at this location.")
                fetch_success[key] = False
            elif key == 'rivers' and value is None: fetch_success[key] = False
        except ee.EEException as e:
            print(f"    -> GEE ERROR fetching {key} ({asset_id}): {e}", file=sys.stderr)
            fetch_success[key] = False
            if 'Asset is not an Image' in str(e) or 'not an ImageCollection' in str(e): print(f"       HINT: Check asset type.", file=sys.stderr)
            elif 'FeatureCollection.load' in str(e) or 'failed' in str(e): print(f"       HINT: Check FeatureCollection handling.", file=sys.stderr)
            elif ('Image.load' in str(e) or 'ImageCollection.load' in str(e)) and 'not found' in str(e): print(f"       UNEXPECTED FAILURE for supposedly accessible asset: {asset_id}.", file=sys.stderr) 
        except Exception as e:
            print(f"    -> UNEXPECTED ERROR fetching {key}: {e}", file=sys.stderr)
            fetch_success[key] = False
    print("\n--- Evaluating Target Soil/Site Classes ---")
    site_classes = []
    derived_texture_class = None
    slope = None
    print("  Pre-calculating Slope (for Alluvial proxy)...")
    if fetch_success.get("nasadem"):
        try:
            dem = ee.Image(datasets_to_fetch["nasadem"]["id"])
            terrain = ee.Terrain.products(dem)
            slope_info = terrain.select('slope').reduceRegion(reducer=ee.Reducer.firstNonNull(), geometry=location_point, scale=30).get('slope')
            slope_val = slope_info.getInfo()
            if slope_val is not None:
                slope = float(slope_val)
                print(f"    Slope calculated: {slope:.1f} deg")
            else:
                print("    -> Could not retrieve slope value.")
        except Exception as topo_e:
            print(f"    -> ERROR calculating slope: {topo_e}", file=sys.stderr)
    else:
        print("    -> Skipped (NASADEM data unavailable).")

    # 1. Texture Classes ('Clay', 'Clay loam', 'Loamy', 'Sandy', 'Sandy loam') (Requires: olm_texture)
    print("  Evaluating Texture Classes...")
    if fetch_success.get("olm_texture"):
        texture_code = int(retrieved_values["olm_texture"])
        print(f"    Using OLM Texture Code = {texture_code}")
        if texture_code in [1, 2]: derived_texture_class = 'Sandy'
        elif texture_code == 3: derived_texture_class = 'Sandy loam'
        elif texture_code in [4, 5, 6]: derived_texture_class = 'Loamy'
        elif texture_code in [7, 8, 9]: derived_texture_class = 'Clay loam'
        elif texture_code in [10, 11, 12]: derived_texture_class = 'Clay'
        if derived_texture_class: site_classes.append(derived_texture_class); print(f"    -> Assigned: {derived_texture_class}")
    else: print("    -> Skipped (OLM Texture data unavailable).")

    # 2. Fertility Class ('Poor') (Requires: olm_soc)
    print("  Evaluating Fertility Class ('Poor')...")
    if fetch_success.get("olm_soc"):
        soc_g_kg = retrieved_values["olm_soc"]
        print(f"    Using OLM SOC = {soc_g_kg:.1f} g/kg")
        if soc_g_kg < 10: site_classes.append('Poor'); print("    -> Assigned: Poor")
        else: print("    -> Not classified as Poor (SOC >= 1%).")
    else: print("    -> Skipped (OLM SOC data unavailable).")

    # 3. Acidity Class ('Acidic') (Requires: stored soil_ph)
    print("  Evaluating Acidity Class ('Acidic')...")
    if location_details.get('soil_ph') is not None:
        ph_value = location_details['soil_ph']
        print(f"    Using pH = {ph_value:.2f}")
        if ph_value < 5.5: site_classes.append('Acidic'); print("    -> Assigned: Acidic")
        else: print("    -> Not classified as Acidic (pH >= 5.5).")
    else: print("    -> Skipped (pH data unavailable).")

    # 4. Drainage Classes ('Well-drained', 'Poorly drained') (Requires: derived_texture_class)
    print("  Evaluating Drainage Classes...")
    if derived_texture_class:
        print(f"    Based on derived texture '{derived_texture_class}'")
        if derived_texture_class in ['Sandy', 'Sandy loam']: site_classes.append('Well-drained'); print("    -> Assigned: Well-drained")
        elif derived_texture_class in ['Clay', 'Clay loam']: site_classes.append('Poorly drained'); print("    -> Assigned: Poorly drained")
        else: print("    -> Drainage not inferred (texture Loamy).")
    else: print("    -> Skipped (Texture class could not be determined).")

    # 5. Water Presence Classes ('Wet', 'Swampy', 'Moist') (Requires: jrc_water, worldcover)
    print("  Evaluating Water Presence Classes ('Wet', 'Swampy', 'Moist')...")
    is_wet = False
    assigned_moist_jrc = False
    if fetch_success.get("jrc_water"):
        water_occurrence = retrieved_values["jrc_water"]
        print(f"    Using JRC Water Occurrence = {water_occurrence:.1f}%")
        if water_occurrence > 50: site_classes.append('Wet'); is_wet = True; print("    -> Assigned: Wet (High JRC occurrence)")
        elif water_occurrence > 10: site_classes.append('Moist'); assigned_moist_jrc = True; print("    -> Assigned: Moist (Moderate JRC occurrence)")
    else: print("    JRC Water data unavailable.")

    if fetch_success.get("worldcover"):
        wc_code = int(retrieved_values["worldcover"])
        print(f"    Checking WorldCover code {wc_code} for Wetlands/Water...")
        if wc_code == 80: # Permanent Water body
            if not is_wet: site_classes.append('Wet'); print("    -> Assigned: Wet (WorldCover Water Body)")
            is_wet = True
        elif wc_code == 90: # Herbaceous Wetland
            site_classes.append('Swampy')
            if not is_wet: site_classes.append('Wet')
            print("    -> Assigned: Swampy (WorldCover Wetland)")
            if not is_wet: print("    -> Also assigned: Wet (Implied by Swampy)")
            is_wet = True # Swampy implies wet conditions
    else: print("    -> WorldCover data unavailable for water/wetland check.")

    # 6. Moisture Regime Classes ('Dry', 'Moist') (Requires: worldclim_precip, plus water status)
    print("  Evaluating Moisture Regime Classes ('Dry', 'Moist')...")
    if fetch_success.get("worldclim_precip"):
        annual_precip = retrieved_values["worldclim_precip"]
        print(f"    Using Annual Precipitation = {annual_precip:.0f} mm")
        if annual_precip < 300: site_classes.append('Dry'); print("    -> Assigned: Dry")
        # Add 'Moist' only if High Precip AND not already classified as Moist/Wet/Swampy
        elif annual_precip > 1000 and not assigned_moist_jrc and not is_wet and 'Swampy' not in site_classes:
            site_classes.append('Moist'); print("    -> Assigned: Moist (High precipitation)")
        else: print("    -> Not Dry/Moist based on precipitation thresholds or already classified.")
    else: print("    -> Skipped (Precipitation data unavailable).")

    # 7. Alluvial Proxy Class ('Alluvial') (Requires: slope AND rivers)
    print("  Evaluating Alluvial Proxy Class ('Alluvial')...")
    if slope is not None and fetch_success.get("rivers"):
        river_present = retrieved_values["rivers"] > 0
        print(f"    Slope = {slope:.1f} deg, River nearby = {river_present}")
        if slope < 2 and river_present:
            site_classes.append('Alluvial')
            print("    -> Assigned: Alluvial (Proxy: Low slope near river)")
        else: print("    -> Not Alluvial (Proxy conditions not met).")
    elif slope is None: print("    -> Skipped (Slope data unavailable).")
    else: print("    -> Skipped (Rivers data unavailable).")

    print("\n--- Final Derived Site Classes (Target List Focused) ---")
    assigned_set = set(site_classes)
    not_assessed_or_assigned = [cls for cls in TARGET_CLASSES if cls not in assigned_set]
    print(f"  Classes from Target List NOT Assigned: {sorted(not_assessed_or_assigned)}")
    if site_classes:
        final_classes = [cls for cls in site_classes if cls in TARGET_CLASSES]
        location_details['soil_type_classes'] = sorted(list(set(final_classes)))
        print(f"  ASSIGNED TARGET CLASSES: {location_details['soil_type_classes']}")
    else:
        print("  No specific target site classes could be assigned based on available data and rules.")
        location_details['soil_type_classes'] = []

    print("\nTargeted classification attempt complete.")

    print(location_details)

    return location_details
