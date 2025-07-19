
# 🧠 Clasificación de Neumonía en Radiografías de Tórax Pediátricas

Este proyecto implementa y compara múltiples modelos de redes neuronales convolucionales (CNN) para la **clasificación automática de radiografías de tórax en pacientes pediátricos**, identificando la presencia o ausencia de neumonía.

Se utilizan arquitecturas preentrenadas (`ResNet50`, `VGG16`, `DenseNet`) y modelos personalizados, evaluando su **precisión**, **capacidad de generalización** y **sensibilidad clínica**.

---

## 🗂️ Estructura del Proyecto

A continuación, se ilustran los elementos que componen la estructura del repositorio:

### 📁 Estructura General del Repositorio

![Estructura 1](./imgs/estructura_general.png)

### 🧪 Contenido del Folder `models`

![Estructura 2](./imgs/estructura_models.png)

### 📊 Flujo del Modelo

![Flujo de datos](./imgs/flujo_modelo.png)

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

## 🖼️ Diagrama General del Modelo

![Diagrama completo](./imgs/diagrama_modelo.png)
