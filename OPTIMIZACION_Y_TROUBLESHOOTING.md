# Guía de Optimización y Troubleshooting

## 🚀 Cómo Maximizar el Rendimiento

### 1. Hiperparámetros Críticos para Ajustar

#### Learning Rate
```python
# Experimentar con:
lr_inicial = [1e-3, 5e-4, 1e-4, 5e-5]  # Fase 1
lr_finetuning = [1e-5, 5e-6, 1e-6]      # Fase 2
```

**Regla de oro:**
- Si val_loss oscila mucho: Reducir LR
- Si el entrenamiento es muy lento: Aumentar LR
- Óptimo típico: 1e-4 (Fase 1), 1e-5 (Fase 2)

#### Batch Size
```python
# Probar:
BATCH_SIZE = [8, 16, 32]
```

**Trade-offs:**
- Batch pequeño (8): Más ruidoso pero mejor generalización
- Batch grande (32): Más estable pero puede overfittear
- **Recomendado: 16** (buen balance)

#### Dropout Rate
```python
# Ajustar según overfitting:
dropout = [0.2, 0.3, 0.4, 0.5]
```

**Señales:**
- Train acc >> Val acc → Aumentar dropout
- Train acc ≈ Val acc → Está bien
- Train acc < Val acc → Reducir dropout

#### Data Augmentation Intensity
```python
# Moderado (conservador)
A.Rotate(limit=90, p=0.5)
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)

# Agresivo (mejor generalización)
A.Rotate(limit=180, p=0.7)
A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
```

### 2. Estrategias de Entrenamiento

#### Opción A: Fast & Good (2-3 horas)
```python
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 25
BATCH_SIZE = 16
USE_TTA = True
N_TTA = 5
TRAIN_RESNET = False  # Solo EfficientNet
```
**F1-score esperado:** 0.78-0.82

#### Opción B: Best Performance (5-6 horas)
```python
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 40
BATCH_SIZE = 16
USE_TTA = True
N_TTA = 10
TRAIN_RESNET = True  # Ensemble
```
**F1-score esperado:** 0.82-0.88

#### Opción C: Ultra Performance (8-10 horas)
```python
# Entrenar 3 modelos:
# - EfficientNetB3
# - EfficientNetB4 (más grande)
# - ResNet50V2
# Ensemble con promedio ponderado
```
**F1-score esperado:** 0.85-0.90

### 3. Preprocesamiento Alternativo

#### Variante 1: Ben Graham Method (especializado para retinas)
```python
def ben_graham_preprocessing(image, target_size=(224, 224)):
    """
    Método usado en competiciones de Kaggle para retinas
    """
    # Escala a un radio fijo
    image = cv2.resize(image, (512, 512))
    
    # Filtro gaussiano para reducir ruido
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Sustracción local promedio
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    local_avg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.addWeighted(image, 4, local_avg, -4, 128)
    
    # Resize final
    image = cv2.resize(image, target_size)
    return image / 255.0
```

#### Variante 2: Green Channel Emphasis
```python
def green_channel_preprocessing(image, target_size=(224, 224)):
    """
    El canal verde tiene más información vascular
    """
    # Extraer canal verde
    green_channel = image[:, :, 1]
    
    # CLAHE en canal verde
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    green_enhanced = clahe.apply(green_channel)
    
    # Reconstruir imagen RGB
    image[:, :, 1] = green_enhanced
    
    image = cv2.resize(image, target_size)
    return image / 255.0
```

### 4. Augmentation Especializado para Retinas

```python
# Augmentation médicamente válido
medical_transform = A.Compose([
    # Rotaciones: los ojos pueden estar en cualquier orientación
    A.Rotate(limit=180, p=0.8),
    
    # Flips: válidos para retinas
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    
    # Zoom: simula diferentes distancias de captura
    A.RandomScale(scale_limit=0.2, p=0.5),
    
    # Brillo/contraste: simula diferentes configuraciones de cámara
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    
    # Desenfoque: simula imágenes de menor calidad
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    
    # Ruido: simula artefactos de sensor
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # Ajustes de color
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.5
    ),
    
    # Compresión JPEG (simula artefactos de compresión)
    A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
])
```

## 🐛 Troubleshooting Común

### Problema 1: Out of Memory (OOM)

**Síntomas:** Error "ResourceExhaustedError" o kernel crash

**Soluciones:**
```python
# 1. Reducir batch size
BATCH_SIZE = 8  # o incluso 4

# 2. Reducir tamaño de imagen
TARGET_SIZE = (192, 192)  # en vez de (224, 224)

# 3. Usar modelo más pequeño
# Cambiar EfficientNetB3 por EfficientNetB0
base_model = EfficientNetB0(...)

# 4. Liberar memoria después de entrenar
import gc
import tensorflow.keras.backend as K

K.clear_session()
gc.collect()

# 5. Reducir número de augmentations en TTA
N_TTA = 3  # en vez de 10
```

### Problema 2: Overfitting Severo

**Síntomas:** Train acc = 0.95, Val acc = 0.60

**Soluciones:**
```python
# 1. Aumentar dropout
dropout_rate = 0.5  # en vez de 0.3

# 2. Más data augmentation
# Ver sección de augmentation especializado arriba

# 3. Más regularización L2
kernel_regularizer=tf.keras.regularizers.l2(0.01)  # en vez de 0.001

# 4. Congelar más capas
for layer in base_model.layers[:-20]:  # solo descongelar últimas 20
    layer.trainable = False

# 5. Early stopping más agresivo
early_stop = EarlyStopping(patience=10)  # en vez de 15
```

### Problema 3: Underfitting (No Aprende)

**Síntomas:** Train acc y Val acc ambos bajos (<0.50)

**Soluciones:**
```python
# 1. Aumentar learning rate
optimizer = Adam(learning_rate=5e-4)  # en vez de 1e-4

# 2. Reducir dropout
dropout_rate = 0.2  # en vez de 0.3

# 3. Descongelar más capas más temprano
for layer in base_model.layers[:-50]:  # descongelar más capas
    layer.trainable = False

# 4. Entrenar más épocas
EPOCHS_PHASE1 = 40  # en vez de 30

# 5. Verificar preprocesamiento
# Asegurarse de que las imágenes se ven correctas
plt.imshow(preprocessed_image)
plt.show()
```

### Problema 4: Convergencia Lenta

**Síntomas:** Después de 20 épocas, acc solo 0.40

**Soluciones:**
```python
# 1. Learning rate schedule diferente
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 1e-4
    else:
        return 1e-5

lr_scheduler = LearningRateScheduler(lr_schedule)

# 2. Usar optimizer diferente
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

# 3. Inicialización de pesos
# Asegurarse de cargar pesos de ImageNet
base_model = EfficientNetB3(weights='imagenet', ...)
```

### Problema 5: Predicciones Sesgadas (Todas una Clase)

**Síntomas:** Test predictions todas "Normal"

**Soluciones:**
```python
# 1. Verificar class weights
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(class_weight_dict)  # Deben ser diferentes

# 2. Usar Focal Loss en vez de categorical crossentropy
model.compile(loss=FocalLoss(gamma=2.0), ...)

# 3. Oversampling de clases minoritarias
from imblearn.over_sampling import RandomOverSampler

# 4. Threshold tuning en predicciones
# En vez de argmax directo, ajustar thresholds por clase
```

### Problema 6: Val Loss Oscila Mucho

**Síntomas:** Val loss sube y baja drásticamente

**Soluciones:**
```python
# 1. Reducir learning rate
optimizer = Adam(learning_rate=5e-5)

# 2. Reducir ReduceLROnPlateau factor
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3)

# 3. Aumentar batch size
BATCH_SIZE = 32

# 4. Reducir augmentation
# Usar augmentation más suave
```

## 📊 Métricas de Monitoreo

### Durante el Entrenamiento

**Señales de buen entrenamiento:**
```
Epoch 1: loss=1.2, val_loss=1.0, acc=0.45, val_acc=0.50 ✅ (val mejor que train)
Epoch 10: loss=0.6, val_loss=0.5, acc=0.75, val_acc=0.72 ✅ (convergiendo)
Epoch 20: loss=0.3, val_loss=0.4, acc=0.88, val_acc=0.82 ✅ (val_acc alto)
```

**Señales de advertencia:**
```
Epoch 20: loss=0.1, val_loss=1.5, acc=0.99, val_acc=0.55 ❌ (overfitting severo)
Epoch 30: loss=1.2, val_loss=1.2, acc=0.45, val_acc=0.45 ❌ (no aprende)
Epoch 15: loss=0.5, val_loss=0.3, acc=0.60, val_acc=0.85 ⚠️  (raro, verificar datos)
```

### Métricas Objetivo por Época

| Época | Train Acc | Val Acc | Val Loss | Comentario |
|-------|-----------|---------|----------|------------|
| 5 | 0.60-0.70 | 0.55-0.65 | 0.8-1.0 | Inicio razonable |
| 10 | 0.75-0.85 | 0.70-0.78 | 0.5-0.7 | Buen progreso |
| 20 | 0.85-0.92 | 0.78-0.85 | 0.3-0.5 | Excelente |
| 30+ | 0.90-0.95 | 0.82-0.88 | 0.2-0.4 | Objetivo final |

## 🎯 Optimizaciones Finales Antes de Entregar

### 1. Calibración de Predicciones
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar probabilidades en conjunto de validación
# Mejora confiabilidad de predicciones
```

### 2. Votación Mayoritaria en TTA
```python
# En vez de promedio de probabilidades
# Usar votación de clases predichas
def predict_with_voting_tta(model, image, n_aug=10):
    predictions = []
    for _ in range(n_aug):
        # ... augmentation ...
        pred_class = np.argmax(model.predict(img))
        predictions.append(pred_class)
    
    # Votación mayoritaria
    from scipy.stats import mode
    final_pred = mode(predictions)[0][0]
    return final_pred
```

### 3. Análisis de Errores
```python
# Analizar imágenes mal clasificadas en validación
errors = val_true != val_pred
error_indices = np.where(errors)[0]

# Visualizar errores
for idx in error_indices[:10]:
    img_path = valid_df.iloc[idx]['ID']
    true_class = category_names[val_true[idx]]
    pred_class = category_names[val_pred[idx]]
    
    # Mostrar imagen
    # Identificar patrones en errores
```

## 💎 Tips Avanzados

### 1. Progressive Resizing
```python
# Fase 1: entrenar con imágenes pequeñas (rápido)
TARGET_SIZE_P1 = (128, 128)
# ... entrenar ...

# Fase 2: fine-tune con imágenes grandes (mejor calidad)
TARGET_SIZE_P2 = (256, 256)
# ... fine-tune ...
```

### 2. Snapshot Ensembling
```python
# Guardar modelos en diferentes épocas
# Promediar predicciones de múltiples snapshots
checkpoints = ['model_epoch20.h5', 'model_epoch25.h5', 'model_epoch30.h5']
```

### 3. Pseudo-Labeling
```python
# Usar predicciones confiables del test set
# para re-entrenar (avanzado, riesgoso)
confident_predictions = test_probs.max(axis=1) > 0.95
# Agregar a training set y re-entrenar
```

---

**Recuerda:** El mejor modelo es el que balances complejidad vs. tiempo de entrenamiento. No necesitas implementar TODO, solo lo que te dé más ganancia por tiempo invertido.

**Prioridad:**
1. ✅ Preprocesamiento especializado (alto impacto)
2. ✅ EfficientNet con fine-tuning (base sólida)
3. ✅ TTA (fácil, alto retorno)
4. ⚡ Ensemble (si tienes tiempo)
5. 🔬 Técnicas avanzadas (solo si ya superaste 0.85)
