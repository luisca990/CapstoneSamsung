from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Crear la app
app = FastAPI()

# Ruta al modelo guardado en formato SavedModel
MODEL_PATH = "modelo_VGGNet16F"

# Cargar el modelo como TFSMLayer (modo inferencia)
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

# Nombres de clases
class_names = ['NORMAL', 'PNEUMONIA']

# Preprocesamiento


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))  # tamaño usado en entrenamiento
    image_array = np.array(image) / 255.0  # normalizar
    return np.expand_dims(image_array, axis=0)


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Realiza predicción
        pred = model(image)
        output_tensor = pred['output_0']
        class_index = int(np.argmax(output_tensor, axis=1)[0])
        confidence = float(np.max(output_tensor))

        result = {
            "filename": file.filename,
            "prediction": class_names[class_index],
            "confidence": round(confidence, 4)
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
