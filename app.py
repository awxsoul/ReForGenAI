# app.py
from flask import Flask, render_template, request

from modules import loc2img, locinfo, predict, img2plan

ReForGen = Flask(__name__) 
loc2img.activateauth()

# --- Define Routes ---
@ReForGen.route('/')  # Route for the homepage
def index():
    template_data= {
        'process':"Generate to see Results",
        'latitude': "",
        'longitude': "",
        'radius': "", 
        'temperature_celsius': "",
        'elevation_meters': "",
        'soil_ph': "",
        'soil_type_classes': "", 
        'climate_classes': ""   
    }
    return render_template('index.html', **template_data)

@ReForGen.route('/processing', methods=['POST'])  
def fetchinput():

    l1,l2=loc2img.getinput()
    loc2img.dwdimg(l1,l2)

    template_data = locinfo.get(l1,l2)
    data=predict.format(template_data)
    
    trees=predict.trees(data,top_n=25)
    template_data['trees']=trees

    img2plan.get()
    return render_template('index.html', **template_data)

# --- Run the Application ---
if __name__ == '__main__':
    ReForGen.run(debug=True, port=5001) 
