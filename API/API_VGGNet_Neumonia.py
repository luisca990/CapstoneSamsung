from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Rutas modelos preentrenados
MODEL_PATH_VGG = "modelo_VGGNet16F"
MODEL_PATH_RESNET = "modelo_ResNet50F_2"

# Rutas modelos CNN desde cero
MODEL_PATH_CNN1 = "modelo_CNNF"
MODEL_PATH_CNN2 = "modelo_CNN2F"

# Cargar modelos preentrenados
model_vgg = tf.keras.layers.TFSMLayer(
    MODEL_PATH_VGG, call_endpoint='serving_default')
model_resnet = tf.keras.layers.TFSMLayer(
    MODEL_PATH_RESNET, call_endpoint='serving_default')

# Cargar modelos entrenados desde cero
model_cnn1 = tf.keras.layers.TFSMLayer(
    MODEL_PATH_CNN1, call_endpoint='serving_default')
model_cnn2 = tf.keras.layers.TFSMLayer(
    MODEL_PATH_CNN2, call_endpoint='serving_default')

# Clases
class_names = ['NORMAL', 'PNEUMONIA']


# Preprocesamiento para RGB (preentrenados)
def preprocess_image_rgb(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


# Preprocesamiento para grises (CNN normales)
def preprocess_image_grayscale(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert(
        "L")  # Escala de grises
    image = image.resize((256, 256))
    image_array = np.expand_dims(np.array(image), axis=-1) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image_rgb(image_bytes)

        pred_vgg = model_vgg(image)
        output_vgg = pred_vgg['output_0']
        index_vgg = int(np.argmax(output_vgg, axis=1)[0])
        conf_vgg = float(np.max(output_vgg))

        pred_resnet = model_resnet(image)
        output_resnet = pred_resnet['output_0']
        index_resnet = int(np.argmax(output_resnet, axis=1)[0])
        conf_resnet = float(np.max(output_resnet))

        result = {
            "vggnet16": {
                "filename": file.filename,
                "prediction": class_names[index_vgg],
                "confidence": round(conf_vgg, 4),
                "probabilities": output_vgg.numpy().tolist()
            },
            "resnet50": {
                "filename": file.filename,
                "prediction": class_names[index_resnet],
                "confidence": round(conf_resnet, 4),
                "probabilities": output_resnet.numpy().tolist()
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/cnn_baseline/")
async def predict_cnn_baseline(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image_grayscale(image_bytes)

        # Modelo CNN 1
        output_cnn1 = model_cnn1(image)['output_0']
        conf_cnn1 = float(output_cnn1.numpy()[0][0])
        index_cnn1 = int(conf_cnn1 >= 0.5)

        # Modelo CNN 2
        output_cnn2 = model_cnn2(image)['output_0']
        conf_cnn2 = float(output_cnn2.numpy()[0][0])
        index_cnn2 = int(conf_cnn2 >= 0.5)

        result = {
            "cnn_model_1": {
                "filename": file.filename,
                "prediction": class_names[index_cnn1],
                "confidence": round(conf_cnn1, 4),
                "probability": conf_cnn1
            },
            "cnn_model_2": {
                "filename": file.filename,
                "prediction": class_names[index_cnn2],
                "confidence": round(conf_cnn2, 4),
                "probability": conf_cnn2
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
