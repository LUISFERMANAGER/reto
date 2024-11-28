from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Inicialización de FastAPI
app = FastAPI()

# Conjunto de datos inicial
dataset = {
    "General": ["Hola", "¿Cómo están?", "Gracias por su atención"],
    "Productos": ["¿Qué productos ofrecen?", "¿Tienen productos nuevos?"],
}

# Modelo y vectorizador
vectorizer = CountVectorizer()
model = MultinomialNB()

# Ampliar el conjunto de datos con nuevas categorías y frases
def expand_dataset(original_dataset):
    # Nuevas categorías y frases
    new_data = {
        "Información de contacto": [
            "¿Cómo puedo contactarlos?",
            "¿Cuál es el número de teléfono?",
            "¿Tienen correo electrónico?",
            "¿Dónde están ubicados?",
        ],
        "Horario de atención": [
            "¿Cuál es el horario de atención?",
            "¿En qué días están abiertos?",
            "¿Abren los fines de semana?",
            "¿Están disponibles en días festivos?",
        ],
        "Precios": [
            "¿Cuánto cuesta el producto?",
            "¿Cuál es el precio del servicio?",
            "¿Tienen descuentos disponibles?",
            "¿Puedo obtener una cotización?",
        ],
        "Política de devolución": [
            "¿Cuál es la política de devoluciones?",
            "¿Puedo devolver el producto?",
            "¿Qué hago si el producto está defectuoso?",
            "¿Ofrecen reembolsos?",
        ],
        "Soporte técnico": [
            "Tengo un problema técnico",
            "¿Cómo puedo solucionar un error?",
            "¿Pueden ayudarme con una instalación?",
            "Mi dispositivo no funciona, ¿qué hago?",
        ],
    }
    
    # Fusionar los datos originales con los nuevos
    for category, phrases in new_data.items():
        if category in original_dataset:
            original_dataset[category].extend(phrases)
        else:
            original_dataset[category] = phrases
    
    return original_dataset

# Ampliar el conjunto de datos
dataset = expand_dataset(dataset)

# Esquemas para FastAPI
class Phrase(BaseModel):
    category: str
    phrase: str

class Category(BaseModel):
    name: str
    phrases: List[str]

# Crea una instancia de FastAPI
app = FastAPI()
app.title = "Misión 3 con FastAPI » Reto Chatobot virtual primario « - (C) GRUPO 4 → Anderson - Nelson - LuisFer - Valentina - Jorge - Neiver"
app.version = "[ Ver → 1.0] "

@app.get("/")
def read_root():
    return {"message": "Bienvenido a clasificador de FastAPI y Naive Bayes (@) Anderson - Nelson - LuisFer - Valentina - Jorge - Neiver"}

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
    # Preparar datos para el modelo
    all_phrases = []
    all_labels = []
    for category, phrases in dataset.items():
        all_phrases.extend(phrases)
        all_labels.extend([category] * len(phrases))
    
    # Vectorizar frases y entrenar modelo
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
