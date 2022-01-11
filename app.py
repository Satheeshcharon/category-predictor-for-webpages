import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
from inscriptis import get_text
import re

cat_dict={
    0:'Bulletins_news_magazine',
    1:'Education',
    2:'Entertainment',
    3:'Finance',
    4:'Fitness',
    5:'Food',
    6:'Fortune_Telling',
    7:'Forums',
    8:'Gamble',
    9:'Governments',
    10:'ISP',
    11:'Job_Portals',
    12:'Online_Shopping',
    13:'Porn',
    14:'Religion',
    15:'Science',
    16:'Sex_Education',
    17:'Social_Media',
    18:'Sports',
    19:'Travel',
    20:'Url_shortner_redirector',
    21:'VPN',
    22:'Weapons',
    23:'WebTV',
    24:'Webmails',
    25:'Webphones',
    26:'alcohol',
    27:'automobiles',
    28:'chatting',
    29:'dating',
    30:'healthcare',
    31:'piracy_movies_softwares'
}

app = Flask(__name__)
model = pickle.load(open('Model.pkl','rb'))


cat_data = load_files(r"database_ml/")
X, y = cat_data.data, cat_data.target

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    url = request.form['url_to_predict']
    print(url)
    regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    validate_url = re.match(regex, url) is not None
    if validate_url == False:
        url = "http://"+url
    try:
        html = urllib.request.urlopen(url).read().decode('utf-8')
        text = get_text(html)
        extracted_data=text.split()
        refined_data=[]
        SYMBOLS = '{}()[].,:;+-*/&|<>=~0123456789' 
        for i in extracted_data:
            if i not in SYMBOLS:
                refined_data.append(i)
        # print("\n","$"*50,"HEYAAA we got arround: ",len(refined_data)," of keywords! Here are they: ","$"*50,"\n")
        predict_this=" ".join(refined_data)
        cat_vect=model[1].transform([predict_this])
        category_predicted = model[0].predict(cat_vect)
    #     output=cat_data.target_names([int(category_predicted)])

        return render_template('index.html', prediction_text='{}: {}'.format(url,cat_dict.get(int(category_predicted))))
    except:
        error="Check url again"
        return render_template('index.html', prediction_text=error)
        

if __name__ == "__main__":
    app.run(debug=True, port=8000)