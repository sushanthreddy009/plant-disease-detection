# 🌱 Plant Disease Detection using Deep Learning 🌿

> 🔥 A complete end-to-end AI web app that detects **Tomato plant diseases** from leaf images using a trained deep learning model — built with 🧠 TensorFlow, Flask, and Streamlit.

---

## 🧠 Project Overview

This project helps farmers, agriculturists, and researchers detect **common tomato plant diseases** from leaf images. Using computer vision and deep learning, we identify and classify leaf diseases into 10 possible categories.

> 🌍 Agriculture meets Artificial Intelligence 💡

---

## 🧩 What It Solves

- 🌾 Helps farmers detect crop diseases early
- 📸 Works by simply uploading an image of the leaf
- 🩺 Returns the type of disease and suggestions (via HTML pages)
- 🧠 Uses a trained Convolutional Neural Network (CNN) model
- 🧪 No manual feature engineering — just pure deep learning magic

---

## 🧰 Tech Stack

| Category       | Tools Used                            |
|----------------|----------------------------------------|
| Programming    | Python 3.x                             |
| Deep Learning  | TensorFlow, Keras                      |
| Frontend       | HTML, CSS                              |
| Backend        | Flask                                  |
| Web App        | Streamlit (optional)                   |
| Deployment     | Localhost / GitHub (manual)            |

---

## 🌾 Diseases Covered

We trained our model to detect **10 tomato leaf conditions**, including:

1. 🍅 Tomato - Bacterial Spot  
2. 🍅 Tomato - Early Blight  
3. 🍅 Tomato - Healthy  
4. 🍅 Tomato - Late Blight  
5. 🍅 Tomato - Leaf Mold  
6. 🍅 Tomato - Septoria Leaf Spot  
7. 🍅 Tomato - Target Spot  
8. 🍅 Tomato - Tomato Yellow Leaf Curl Virus  
9. 🍅 Tomato - Tomato Mosaic Virus  
10. 🍅 Tomato - Two-Spotted Spider Mite

---

## 🧬 Model Architecture (Simplified)

```text
Input Image (128x128x3)
↓
Conv2D + MaxPooling (x3)
↓
Flatten → Dense Layers
↓
Output Layer (10 classes)

## 📁 Project Structure

├── static/
│   ├── upload/               # Uploaded leaf images
│   ├── images/               # Backgrounds, logo
├── templates/
│   ├── index.html            # Upload & UI page
│   ├── Tomato-[Disease].html # Result pages for each disease
├── model.h5                  # Trained deep learning model
├── leaf.py                   # Flask backend app
├── requirements.txt          # All dependencies
├── .gitignore                # To ignore virtualenv etc
├── README.md                 # THIS FILE


## 🚀 How To Run Locally (Windows/Any OS)

Follow these 🔟 legendary steps:

### 1. Clone the repo

git clone https://github.com/sushanthreddy009/plant-disease-detection.git 

cd plant-disease-detection


### 2. Create a Virtual Environment

python -m venv venv


Activate:

- On Windows: venv\Scripts\activate


- On Mac/Linux: source venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


💡 This will install: Flask, TensorFlow, Keras, NumPy, Pandas, etc.

### 4. Run the App

python leaf.py


### 5. Open in Browser

Go to:

http://127.0.0.1:5000


### 6. Upload a Tomato Leaf Image

- Select a **clear photo** of a tomato leaf  
- Wait for it to process  
- Get **instant disease diagnosis** 🔍

---

## 📸 Note About Images

- Image **will be resized** to 128x128 internally  
- You can upload **JPG / PNG** images  
- A small preview will be shown on results page  

---

## 🧠 Sample Output

Predicted Class → Tomato - Bacterial Spot Disease  
HTML Page → Tomato-Bacterial Spot.html  

The app renders a friendly web page for each disease with information and treatment suggestions.

---

## 🔍 Key Code (Prediction Logic)

```python
def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    pred = np.argmax(result, axis=1)
    ...

🧪 Model Training (Optional)
Model was trained using TensorFlow on the PlantVillage dataset

Saved as model.h5 and reused in this web app

To retrain, modify your CNN and retrain with your dataset

🧾 License
MIT License — feel free to use and modify ✨

🤝 Contributing
Pull requests are welcome. For major changes, open an issue first.

🧑‍💻 Maintainer
Made with ❤️ by Sushanth Reddy

🌟 Support
If you found this helpful, please ⭐ star the repo. It motivates me to build more!


