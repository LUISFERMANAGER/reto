from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import string

# Crear instancia de FastAPI
app = FastAPI()
app.title = "Clasificador FastApi con Naive Bayes  → (c) Grupo 4 - Anderson - Javier - Jorge - Luis - Nelson - Neiver - Valentina"
app.version = "[ Ver → 1.0 ]"

# Datos iniciales
dataset = {
    "General": ["Hola", "¿Cómo están?", "Tengo un problema técnico"],
    "Productos": ["¿Qué productos ofrecen?", "¿Tienen productos nuevos?"],
}

# Modelo y vectorizador
vectorizer = CountVectorizer()
model = MultinomialNB()

# Esquemas para FastAPI
class Phrase(BaseModel):
    category: str
    phrase: str

class Category(BaseModel):
    name: str
    phrases: List[str]

# Función para calcular frecuencia de palabras
def calcular_frecuencia(texto: str) -> Dict[str, int]:
    texto = texto.lower()  # Convertir a minúsculas
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    palabras = texto.split()  # Dividir en palabras
    return dict(Counter(palabras))  # Calcular frecuencia

# Función para expandir el dataset
def expand_dataset(original_dataset):
    new_data = {
        "Llamenos al 1234567890": [
            "¿Cómo puedo contactarlos?",
            "¿Cuál es el número de teléfono?",
            "¿Tienen correo electrónico?",
            "¿Dónde están ubicados?",
        ],
        "Horario de atención lunes a viernes de 8:am a 5:pm": [
            "¿Cuál es el horario de atención?",
            "¿En qué días están abiertos?",
            "¿Abren los fines de semana?",
            "¿Están disponibles en días festivos?",
        ],
        "Los Precios varian de acuerdo al artículo": [
            "¿Cuánto cuesta el producto?",
            "¿Cuál es el precio del servicio?",
            "¿Tienen descuentos disponibles?",
            "¿Puedo obtener una cotización?",
        ],
        "Política de devolución de acuerdo al CSC": [
            "¿Cuál es la política de devoluciones?",
            "¿Puedo devolver el producto?",
            "¿Qué hago si el producto está defectuoso?",
            "¿Ofrecen reembolsos?",
        ],
        "Soporte técnico dirijase a nuestras oficinas o al portar www:tecnicosgrupo4.com o a la linea 123456789": [
            "Tengo un problema técnico",
            "¿Cómo puedo solucionar un error?",
            "¿Pueden ayudarme con una instalación?",
            "Mi dispositivo no funciona, ¿qué hago?",
        ],
    }
    for category, phrases in new_data.items():
        if category in original_dataset:
            original_dataset[category].extend(phrases)
        else:
            original_dataset[category] = phrases
    return original_dataset

# Ampliar dataset inicial
dataset = expand_dataset(dataset)

# Rutas de FastAPI
@app.get("/")
def read_root():
    return {"message": "Bienvenido a Clasificador FastApi con Naive Bayes → (c) Grupo 4 ► Anderson - Javier - Jorge - Luis - Nelson - Neiver - Valentina"}

@app.get("/categories")
def get_categories():
    return {"categories": list(dataset.keys())}

@app.post("/add-category")
def add_category(category: Category):
    if category.name in dataset:
        dataset[category.name].extend(category.phrases)
    else:
        dataset[category.name] = category.phrases
    return {"message": f"Categoría '{category.name}' añadida o actualizada exitosamente."}

@app.post("/add-phrase")
def add_phrase(phrase: Phrase):
    if phrase.category not in dataset:
        raise HTTPException(status_code=404, detail="Categoría no encontrada.")
    dataset[phrase.category].append(phrase.phrase)
    return {"message": f"Frase añadida a la categoría '{phrase.category}'."}

@app.post("/train")
def train_model():
    all_phrases = []
    all_labels = []
    for category, phrases in dataset.items():
        all_phrases.extend(phrases)
        all_labels.extend([category] * len(phrases))
    X = vectorizer.fit_transform(all_phrases)
    model.fit(X, all_labels)
    return {"message": "Modelo entrenado exitosamente."}

@app.post("/predict")
def predict(phrase: str):
    if not vectorizer.vocabulary_:
        raise HTTPException(status_code=400, detail="El modelo no ha sido entrenado.")
    X = vectorizer.transform([phrase])
    prediction = model.predict(X)
    return {"category": prediction[0]}

@app.get("/dataset")
def get_dataset():
    return dataset

@app.post("/frecuencia")
def frecuencia_palabras(texto: str):  
    frecuencias = calcular_frecuencia(texto)
    return {"frecuencia": frecuencias}

# Prueba fuera de FastAPI
if __name__ == "__main__":
    # Mostrar dataset ampliado
    print("Dataset ampliado:")
    for category, phrases in dataset.items():
        print(f"{category}: {phrases}")

    # Prueba de frecuencia de palabras
    texto_ejemplo = """
    La programación en Python es muy divertida. Python es un lenguaje poderoso y fácil de aprender.
    """
    frecuencias = calcular_frecuencia(texto_ejemplo)
    print("\nFrecuencia de palabras:")
    for palabra, frecuencia in frecuencias.items():
        print(f"'{palabra}': {frecuencia}")
        
