from flask import Flask
from flask import request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Descargar recursos de NLTK
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Definir variables para preprocesamiento
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Tu función de preprocesamiento (copiada de tu notebook)
def data_preprocessing(text):
    # Limpieza de datos
    text = re.sub(r'http\S+|www.\S+', '', text)  # Eliminamos URLs
    text = re.sub(re.compile('<.*?>'), '', text)  # Removemos tags HTML
    text = re.sub('[^A-Za-z]+', ' ', text)  # Tomamos solo las palabras
    text = re.sub('\s+', ' ', text).strip()  # Normalizar espacios
    text = text.lower()  # Convertimos todo a minúsculas

    # Tokenización
    tokens = nltk.word_tokenize(text)

    # Removemos las stopwords
    text = [word for word in tokens if word not in stop_words]

    # Lematización
    text = [lemmatizer.lemmatize(word) for word in text]

    # Unimos las palabras
    text = ' '.join(text)

    return text

# Cargar solo modelo y vectorizador
classifier = joblib.load('classifier.pkl')
vectorizer = joblib.load('sentiment_vectorizer.pkl')

@app.route('/predice', methods=['POST'])
def predict():
    try:
        # Obtener JSON
        json_ = request.json
        text = json_['review']
        
        # Usar función de preprocesamiento
        processed_text = data_preprocessing(text)
        
        # Vectorizar
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predecir
        prediction = classifier.predict(text_vectorized)
        probability = classifier.predict_proba(text_vectorized)
        
        # Mapear resultado
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        result = sentiment_map[prediction[0]]
        confidence = max(probability[0])
        
        return {
            "predicción": result,
            "confianza": float(confidence),
            "texto_review": text
        }
        
    except Exception as e:
        return {"error": str(e)}, 400

@app.route('/health', methods=['GET'])
def health():
    return {"status": "API funcionando correctamente"}, 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)