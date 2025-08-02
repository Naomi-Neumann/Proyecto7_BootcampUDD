# Proyecto de Análisis de Sentimientos con reviews de Google Playstore

API REST desarrollada con Flask para análisis de sentimientos en reseñas de texto. El proyecto utiliza técnicas de procesamiento de lenguaje natural (NLP) y machine learning para clasificar reseñas de Googleplaystore como positivas, neutrales o negativas.

## Características
- **Clasificación de Sentimientos**: Clasifica texto en tres categorías (Negativo, Neutral, Positivo)
- **Preprocesamiento Automático**: Limpieza y normalización de texto
- **API REST**: Endpoints fáciles de usar
- **Confianza de Predicción**: Retorna el nivel de confianza de cada predicción

## Tecnologías Utilizadas
- **Flask**: Framework web para Python
- **joblib**: Serialización de modelos

## Instalación
### Prerrequisitos
- Python 3.12.6
- pip (gestor de paquetes de Python)
### Pasos de instalación
1. **Clona el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/Proyecto7_BootcampUDD.git
   cd Proyecto7_BootcampUDD
   ```
2. **Crea un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
3. **Instala las dependencias**
   ```bash
   pip install flask nltk scikit-learn joblib
   ```

## Uso de la API
### Ejecutar la aplicación
```bash
python app.py
```
La API estará disponible en `http://localhost:8000`
### Endpoints Disponibles
#### 1. Predicción de Sentimientos
- **URL**: `/predice`
- **Método**: `POST`
- **Content-Type**: `application/json`
**Cuerpo de la petición:**
```json
{
    "review": "This movie was absolutely amazing! I loved every minute of it."
}
```
**Respuesta exitosa:**
```json
{
    "predicción": "Positive",
    "confianza": 0.87,
    "texto_review": "This movie was absolutely amazing! I loved every minute of it."
}
```
### Ejemplos de uso
**Con cURL:**
```bash
curl -X POST http://localhost:8000/predice \
  -H "Content-Type: application/json" \
  -d '{"review": "The product quality is terrible, worst purchase ever!"}'
```
**Con Python (requests):**
```python
import requests
url = "http://localhost:8000/predice"
data = {
    "review": "Great service, highly recommend!"
}
response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Estructura del Proyecto
```
Proyecto7_BootcampUDD/
├── app.py                          # Aplicación principal Flask
├── classifier.pkl                  # Modelo entrenado
├── sentiment_vectorizer.pkl        # Vectorizador TF-IDF
├── Naomi_Neumann_Proyecto_M7.ipynb # Notebook de entrenamiento
└── README.md                       # Este archivo
```

## Notas Importantes
- El modelo fue entrenado en inglés, funciona mejor con texto en este idioma
- La API requiere que los archivos `classifier.pkl` y `sentiment_vectorizer.pkl` estén en el directorio raíz

## Autor
**Naomi Neumann**
- Proyecto desarrollado para el Bootcamp UDD
