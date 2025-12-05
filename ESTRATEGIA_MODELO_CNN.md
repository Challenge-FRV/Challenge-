# Estrategia del Modelo CNN - Challenge de Clasificación de Enfermedades Oculares

## 📊 Análisis del Challenge

- **Objetivo**: Clasificar imágenes de fondo de ojo en 5 categorías (Normal, Diabetes, Cataract, Myopia, Glaucoma)
- **Dataset**: 5,078 imágenes de entrenamiento, 1,088 validación, 1,089 test
- **Baseline a superar**: F1-score = 0.56 (SVM con descriptores clásicos)
- **Meta competitiva**: F1-score > 0.80 para asegurar posición ganadora

## 🎯 Estrategia Ganadora

### 1. Transfer Learning con EfficientNetB3
**¿Por qué EfficientNet?**
- Estado del arte en eficiencia: mejor precisión/costo computacional
- 12M de parámetros pre-entrenados en ImageNet
- Arquitectura optimizada con compound scaling
- Excelente rendimiento en imágenes médicas

**Implementación:**
- Modelo base pre-entrenado (sin top layer)
- Fine-tuning en 2 fases:
  - Fase 1: Base congelado, entrenar solo capas superiores (30 épocas)
  - Fase 2: Descongelar todo, fine-tuning completo (40 épocas)

### 2. Preprocesamiento Especializado para Imágenes Médicas

**Técnicas aplicadas:**
- **Detección automática del círculo del fondo de ojo**: Elimina bordes negros irrelevantes
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Mejora contraste en estructuras vasculares
- **Normalización**: Escala a [0,1] para estabilidad del entrenamiento
- **Redimensionamiento inteligente**: 224x224 con interpolación Lanczos

**Impacto esperado:** +10-15% en F1-score vs. sin preprocesamiento

### 3. Data Augmentation Agresivo

**Transformaciones aplicadas (con Albumentations):**
- Rotaciones: ±180° (prob=0.7) - crucial para imágenes médicas
- Flips horizontal/vertical: (prob=0.5)
- Ajustes de brillo/contraste: ±30% (prob=0.7)
- Gaussian Blur: kernel 3-5 (prob=0.3)
- Gaussian Noise: (prob=0.3)
- Transformaciones afines: shift, scale (prob=0.5)
- Ajustes HSV: hue/sat/val (prob=0.5)

**Beneficio:** Reduce overfitting, aumenta generalización, simula variabilidad real

### 4. Class Balancing

**Problema detectado:** Dataset probablemente desbalanceado
**Solución:** Pesos de clase calculados con `sklearn.compute_class_weight`
**Efecto:** El modelo presta más atención a clases minoritarias

### 5. Arquitectura Personalizada

```
Input (224, 224, 3)
    ↓
EfficientNetB3 Pre-entrenado (base)
    ↓
GlobalAveragePooling
    ↓
BatchNormalization + Dropout(0.3)
    ↓
Dense(512, ReLU) + L2 Regularization
    ↓
BatchNormalization + Dropout(0.3)
    ↓
Dense(256, ReLU) + L2 Regularization
    ↓
Dropout(0.15)
    ↓
Dense(5, Softmax)
```

**Regularización agresiva:**
- Dropout: 0.3, 0.3, 0.15
- L2 regularization: 0.001
- Batch Normalization en cada etapa

### 6. Test-Time Augmentation (TTA)

**Concepto:** Predecir múltiples versiones augmentadas de cada imagen de test
**Implementación:** 8 predicciones por imagen con diferentes augmentations
**Resultado:** Promedio de probabilidades para predicción final más robusta
**Mejora esperada:** +3-5% en F1-score

### 7. Ensemble de Modelos (Opcional)

**Arquitecturas:**
1. EfficientNetB3 (modelo principal)
2. ResNet50V2 (modelo secundario)

**Combinación:** Promedio ponderado (60% EfficientNet, 40% ResNet)
**Ventaja:** Diferentes arquitecturas capturan diferentes patrones
**Mejora esperada:** +2-4% adicional en F1-score

## 🚀 Ventajas Competitivas

### vs. Baseline (SVM)
- **Transfer Learning**: Aprovecha 12M parámetros pre-entrenados vs. entrenar desde cero
- **Representaciones profundas**: CNNs aprenden características jerárquicas vs. descriptores fijos
- **End-to-end learning**: Optimización conjunta de features y clasificador

### vs. Otros Competidores
- **Preprocesamiento médico especializado**: No solo usar imágenes raw
- **TTA**: Mayoría no lo implementa por costo computacional
- **Ensemble**: Duplica tiempo de entrenamiento pero vale la pena
- **Fine-tuning en 2 fases**: Mejor convergencia que fine-tuning directo
- **Class balancing**: Crucial para F1-score macro

## 📈 Rendimiento Esperado

| Métrica | Baseline | Nuestro Modelo | Mejora |
|---------|----------|----------------|--------|
| F1-score (test) | 0.56 | **0.82-0.88** | +46-57% |
| Accuracy | ~0.60 | **0.85-0.90** | +42-50% |
| AUC | ~0.70 | **0.92-0.95** | +31-36% |

## 🔧 Optimizaciones Implementadas

1. **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
2. **Early Stopping**: Patience=15 épocas, restore best weights
3. **Model Checkpointing**: Guarda mejor modelo según val_loss
4. **Batch Size**: 16 (balance entre memoria y convergencia)
5. **Optimizador**: Adam con LR adaptativo (1e-4 → 1e-5)

## 🎓 Fundamentos del Curso Aplicados

### Conceptos utilizados:
- ✅ **CNNs**: Arquitectura principal
- ✅ **Transfer Learning**: EfficientNet/ResNet pre-entrenados
- ✅ **Data Augmentation**: Rotaciones, flips, transformaciones
- ✅ **Filtros**: Convoluciones en la CNN
- ✅ **Operaciones morfológicas**: Detección de contornos para preprocesamiento
- ✅ **Aprendizaje supervisado**: Clasificación con etiquetas
- ✅ **SVM**: Baseline comparativo
- ✅ **Random Forest**: Alternativa descartada (CNNs superiores)
- ✅ **Descriptores de color/textura**: Implícitos en CNNs
- ✅ **HOG/SIFT**: Conceptos base de features aprendidas por CNNs

### Técnicas avanzadas opcionales:
- 🔥 **Focal Loss**: Para clases muy desbalanceadas
- 🔀 **Mixup**: Regularización mediante mezcla de imágenes
- 🎯 **Attention Mechanisms**: Enfoque en regiones relevantes

## 💡 Recomendaciones para Ejecución

1. **Hardware recomendado**: GPU con ≥6GB VRAM (Google Colab funciona)
2. **Tiempo de entrenamiento**:
   - Con GPU: 2-4 horas (EfficientNet completo)
   - Con CPU: 12-20 horas
3. **Orden de implementación**:
   - Primero: Solo EfficientNet + TTA (más rápido, ya muy bueno)
   - Luego: Agregar ResNet para ensemble (si hay tiempo)
4. **Debugging**: Probar con subset pequeño primero (100 imágenes)

## 🏆 Claves para Ganar la Competición

1. ✅ **Calidad > Cantidad**: Preprocesamiento especializado es crucial
2. ✅ **TTA es oro**: Muchos lo omiten, tú no
3. ✅ **Fine-tuning en 2 fases**: Evita catastrophic forgetting
4. ✅ **Validación cuidadosa**: No hacer overfitting en validación
5. ✅ **Ensemble si es posible**: Vale el esfuerzo extra
6. ✅ **Experimentación**: Probar diferentes learning rates, batch sizes

## 📦 Entregables

- ✅ `TestPredictions.csv`: Predicciones finales
- ✅ `MetodoFinalGrupo3.ipynb`: Notebook completo y ejecutable
- ✅ Modelos guardados: `efficientnet_fundus_final.h5`, opcionalmente `resnet_fundus_final.h5`

---

**¡Buena suerte! Con este enfoque, deberías estar en el top 3 de la competición.** 🎯
