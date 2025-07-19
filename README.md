
# 🧠 Clasificación de Neumonía en Radiografías de Tórax Pediátricas

Este proyecto implementa y compara múltiples modelos de redes neuronales convolucionales (CNN) para la **clasificación automática de radiografías de tórax en pacientes pediátricos**, identificando la presencia o ausencia de neumonía.

Se utilizan arquitecturas preentrenadas (`ResNet50`, `VGG16`, `DenseNet`) y modelos personalizados, evaluando su **precisión**, **capacidad de generalización** y **sensibilidad clínica**.

---

## 🗂️ Estructura del Proyecto

A continuación, se ilustran los elementos que componen la estructura del repositorio:

### 📁 Estructura General del Repositorio

```plaintext
.
├── .ipynb_checkpoints/          → Checkpoints del notebook principal
├── API/                         → Contiene la API y modelos exportados
│   ├── modelo_CNN2F/            → Modelo profundo personalizado exportado
│   ├── modelo_CNNF/             → Modelo CNN personalizado exportado
│   ├── modelo_ResNet50F/        → Modelo preentrenado ResNet50 exportado
│   ├── API_VGGNet_Neumonia.py   → API para servir los modelos
│   └── requirements.txt         → Dependencias específicas para la API
├── Datos/chest_xray/           → Radiografías utilizadas para entrenamiento y validación
│   ├── tren/                    → Carpeta de entrenamiento
│   ├── Val/                     → Carpeta de validación
│   ├── prueba/                  → Carpeta de testeo final
├── AIRE_LCAS.ipynb             → Notebook principal del proyecto
├── LICENCIA                    → Licencia de uso
├── README.md                   → Este archivo
├── modelo_VGGNet16F.h5         → Modelo preentrenado VGG16 (h5)
├── modelo_ResNet50F.h5         → Modelo preentrenado ResNet50 (h5)
└── requirements.txt            → Requisitos para instalar dependencias generales
```

---

## 1. 📚 Conjunto de Datos

- Fuente: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Composición:
  - **Total de imágenes**: 5856 radiografías
  - **Clases**:
    - Normal
    - Neumonía (viral o bacteriana)
  - **Distribución**:
    - Entrenamiento
    - Validación
    - Prueba

---

## 2. 🧪 Modelos Evaluados

### 🔹 ResNet50
- Precisión: **86.06 %**
- Recall (NEUMONÍA): **1.00**
- F1 Score: **0.94**

### 🔹 VGG16 (congelado)
- Precisión: **99 %**
- Validación: **100 %** (⚠️ *sobreajuste detectado*)

### 🔹 DenseNet (congelado)
- Precisión: **62.5 %**
- Alta pérdida, pobre generalización

### 🔹 ResNet50 con Data Augmentation (congelado)
- Precisión de prueba: **38 %**

### 🔹 CNN Personalizado
- Precisión de prueba: **87 %**
- Recall: **92.8 %**
- AUC: **0.85**

### 🔹 CNN Profundo Personalizado
- Precisión de prueba: **90 %**
- AUC: **0.85**
- F1 Score:
  - NORMAL: **0.87**
  - NEUMONÍA: **0.92**

---

## 3. 🏆 Logros

- ✅ Identificación precisa de neumonía con modelos personalizados
- ✅ Alto *recall* clínicamente relevante
- ✅ Comparativa entre modelos preentrenados y entrenados desde cero

---

## 4. 🚀 Mejoras Futuras

- 🔧 Reducción de *overfitting* mediante:
  - Regularización
  - Aumento de datos
- 🧬 Evaluación en cohortes más diversas
- 🖥️ Desarrollo de una interfaz para uso clínico (**API**)

---

## 🏗️ Comparación de Arquitecturas

- Las **arquitecturas preentrenadas** (`ResNet50`, `VGG16`, `DenseNet`) fueron evaluadas usando:
  - `SparseCategoricalCrossentropy`
- Los **modelos personalizados** se evaluaron con:
  - Métricas para clasificación binaria (`AUC`, `F1`, `Recall`, etc.)

---

## 👨‍💻 Autores del Proyecto

A continuación se listan los autores y sus respectivos roles en el desarrollo del proyecto:

| Nombre                         | Rol                      | Contacto / GitHub                                   |
|--------------------------------|---------------------------|-----------------------------------------------------|
| 🧑‍💻 Luis Carlos Romero Cardenas|  Especialista en Modelado y Arquitectura de Redes Neuronales | ✉️ luisca990@gmail.com<br>🔗 [@luisca990](https://github.com/luisca990) |
| 👩‍🔬 María Camila Plazas Gómez      | Líder de Proyecto y Responsable de Documentación Técnica | ✉️ kmi.pg18@gmail.com<br>🔗 [@MariaCamilaPlazasG](https://github.com/MariaCamilaPlazasG) |
| 👨‍🎓 Nicoll Alejandra Arrieta Gómez      | Analista de Datos y Preprocesamiento          | ✉️ alejagomez072006@gmail.com<br>🔗 [unu0987654](https://github.com/unu0987654) |
| 👩‍💻 Efraín Santiago Mayorga Chaves  | Responsable de Evaluación de Desempeño y Validación Clínica     | ✉️ santiago.mayorga@outlook.com<br>
(https://github.com/santiagoma14) |

## 👨‍💻 Anexos
Link del video:<br>🔗 [Video Final] (https://youtu.be/-v-f2ah-3zg?si=KW-dD6bLu2mLs8IT) |
