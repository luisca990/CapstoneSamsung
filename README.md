Este proyecto implementa y compara múltiples modelos de redes neuronales convolucionales (CNN) para la clasificación automática de radiografías de tórax en pacientes pediátricos, identificando la presencia o ausencia de neumonía. Se utilizan arquitecturas preentrenadas (ResNet50, VGG16, DenseNet) y modelos personalizados, evaluando su precisión, capacidad de generalización y sensibilidad clínica.

Dataset
Se utilizó un conjunto de datos disponible en Kaggle, compuesto por 5856 radiografías divididas en imágenes normales y con neumonía (viral o bacteriana), organizadas en carpetas de entrenamiento, prueba y validación.

Modelos Evaluados
ResNet50
Precisión: 86.06 %
Recall (PNEUMONIA): 1.00
F1-score: 0.94

VGG16 (congelado)
Precisión: 99 %
Validación: 100 % (overfitting)

DenseNet (congelado)
Precisión: 62.5 %
Alta pérdida, pobre generalización

ResNet50 con data augmentation (congelado)
Precisión test: 38 %

CNN Personalizado
Precisión test: 87 %
Recall: 92.8 %
AUC: 0.85

CNN Profundo Personalizado
Precisión test: 90 %
AUC: 0.85
F1-score: 0.87 (NORMAL), 0.92 (PNEUMONIA)

Logros
Identificación precisa de neumonía con modelos personalizados

Alto recall clínicamente relevante

Comparativa entre modelos preentrenados y entrenados desde cero

Mejoras Futuras
Reducción de overfitting mediante regularización y aumento de datos

Evaluación en cohortes más diversas

Desarrollo de una interfaz para uso clínico (API)
