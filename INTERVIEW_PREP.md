# Guía de Preparación para Entrevista — Electrical Symbol Recognition

## 1. Arquitectura General

```
┌────────────┐      ┌────────────────────┐      ┌──────────────┐
│  main.py   │─────▶│  InferenceEngine   │─────▶│  detector.py │
│  (CLI)     │      │  (inference_       │      │  SAM3Model   │
├────────────┤      │   engine.py)       │      │  MatrixGen   │
│  app.py    │─────▶│                    │      │  infer_all   │
│  (Gradio)  │      └────────┬───────────┘      └──────────────┘
└────────────┘               │
                    ┌────────▼───────────┐
                    │  utils.py          │
                    │  NMS, visualize,   │
                    │  COCO helpers      │
                    ├────────────────────┤
                    │  metrics.py        │
                    │  P/R/F1, CM        │
                    ├────────────────────┤
                    │  config.py         │
                    │  Parámetros        │
                    └────────────────────┘
```

**Flujo de datos:**
1. `config.py` → Define clases, umbrales, rutas
2. `detector.py` → `MatrixGenerator` extrae crops COCO → genera strip de referencia por clase
3. `detector.py` → `infer_image_all_classes` concatena `[strip | imagen]` → SAM3 predice boxes
4. `utils.py` → NMS global por cobertura, filtra, dibuja boxes, guarda imágenes
5. `metrics.py` → Compara GT vs predicciones con IoU → P/R/F1/CM

---

## 2. Componente por Componente

### `config.py` — Centro de configuración
**Qué contiene:** Todas las constantes del proyecto.
**Puntos clave para la entrevista:**
- `CLASSES` define los 5 tipos de componentes eléctricos
- `CONFIDENCE_THRESHOLDS` → umbral de detección por clase en la inferencia
- `VISUALIZATION_THRESHOLDS` → umbral más alto para filtrar qué se dibuja/muestra
- `VISUALIZATION_DISABLED_CLASSES` → clases desactivadas (FA DC tiene threshold 1.10 = imposible)
- `CROP_MARGINS` → margen de expansión al extraer crops de referencia
- `SCALE_RANGES` → rango de escalado para variaciones multi-escala en el strip

**Si te piden cambiar un umbral:** modifica únicamente `config.py`.

---

### `detector.py` — Núcleo de inferencia
**3 componentes principales:**

#### `SAM3Model` (líneas 43-125)
- Wrapper de HuggingFace Transformers para SAM3
- `initialize()` → carga modelo y procesador
- `predict_with_boxes()` → inferencia con prompts de bounding boxes
- **Clave:** El modelo recibe una imagen con boxes de referencia y devuelve boxes detectados

#### `MatrixGenerator` (líneas 132-408)
- Extrae crops de COCO annotations → `extract_best_crops()`
- Genera strip visual por clase → `generate_matrix()`
- El strip tiene 2 columnas: columna 1 = variaciones de escala, columna 2 = crops originales
- **Si te piden añadir una clase:** necesitas anotaciones COCO de esa clase

#### `infer_image_all_classes()` (líneas 415+)
- Procesa UNA imagen con TODAS las clases
- Para cada clase: escala strip → construye canvas `[strip | target]` → SAM3 infiere
- Filtra por tamaño (`SIZE_FILTER_MULTIPLIER`) y lado del canvas (solo derecho)
- **Si te piden mejorar rendimiento:** se puede paralelizar por clase con threading

---

### `inference_engine.py` — Reutilización CLI ↔ UI
**Patrón clave:** Lazy loading del modelo SAM3.

```python
class InferenceEngine:
    def _ensure_loaded(self)    # Carga modelo en primer uso
    def process_single_image()  # Tab 1: 1 imagen → (imagen, json, log)
    def process_batch()         # Tab 2/CLI: todas las imágenes
```

- El modelo se carga UNA sola vez y se reutiliza
- `process_single_image()` aplica los mismos filtros de visualización que el batch
- **Si te piden separar CLI y UI:** ya están separados, comparten InferenceEngine

---

### `utils.py` — Funciones de utilidad
**4 funciones públicas:**

| Función | Propósito |
|---------|-----------|
| `load_coco_data()` | Carga JSON COCO → dicts por imagen/anotación |
| `class_agnostic_nms()` | NMS por **cobertura** (no IoU estándar) |
| `draw_final_boxes()` | Dibuja boxes con `ClassID : Score` |
| `visualize_coco_results()` | Carga JSON → aplica filtros → guarda imágenes |

**Punto clave sobre NMS:**
- Se usa **coverage** = intersection / min(area_a, area_b)
- Esto captura cajas anidadas que IoU estándar no detecta
- **Si te preguntan por qué no IoU:** porque una caja pequeña dentro de una grande tiene IoU bajo pero coverage alto

---

### `metrics.py` — Evaluación
**Flujo:**
1. `_filter_predictions()` → Filtra por umbrales de visualización
2. `_match_boxes()` → Greedy matching GT ↔ Pred con IoU (torchvision)
3. `_build_classification_data()` → Genera y_true/y_pred con "Background" para FN/FP
4. `compute_metrics()` → CLI: guarda CSV + PNG
5. `compute_metrics_for_ui()` → UI: devuelve DataFrame + Figure

**Si te preguntan sobre la Confusion Matrix:**
- Filas = GT, Columnas = Pred
- "Background" = detecciones sin match (FP) o GT sin detectar (FN)
- IoU threshold = 0.4 (configurable)

---

### `app.py` — Interfaz Gradio
**2 pestañas:**

**Tab 1 — Single Image Debug:**
- Dropdown → selecciona imagen → botón → imagen anotada + JSON + Debug Log
- Layout: 70% imagen / 30% datos

**Tab 2 — Batch Analysis:**
- Checkbox "Skip Inference" → carga results.json existente
- Layout 2 columnas: izquierda (métricas + CM + logs) / derecha (galería + referencias)
- **Si te piden añadir una funcionalidad:** todo está en `predict_single_image_logic()` o `run_batch_logic()`

---

## 3. Preguntas Probables y Dónde Tocar

| Pregunta | Respuesta / Acción |
|----------|-------------------|
| "Añade una nueva clase" | 1) Añadir a `CLASSES` en `config.py` 2) Añadir umbrales 3) Anotar crops en COCO |
| "Cambia el umbral de confianza" | `config.py` → `CONFIDENCE_THRESHOLDS` |
| "Cambia qué clases se visualizan" | `config.py` → `VISUALIZATION_THRESHOLDS` / `VISUALIZATION_DISABLED_CLASSES` |
| "¿Cómo funciona el NMS?" | Coverage-based, no IoU. Ver `utils.py` → `class_agnostic_nms()` |
| "¿Por qué visual prompting?" | SAM3 Few-shot: se le muestra ejemplo visual y detecta similares |
| "¿Por qué lazy loading?" | El modelo tarda ~5s en cargar. Solo se carga 1 vez para ambas pestañas |
| "¿Cómo medir rendimiento?" | `metrics.py` → IoU matching → P/R/F1 por sklearn |
| "¿Cómo haría esto en producción?" | Servir con FastAPI, cachear modelo, batch async, pre-calentar |
| "¿Qué mejoraría?" | Multi-GPU, data augmentation en strips, fine-tuning SAM3, caching de features |
| "¿Cómo dockerizaste?" | CUDA 12.1 base, PyTorch separado, volumes para data/models/output |

---

## 4. Flujo Técnico Detallado (Inference)

```
1. Cargar annotations.json (COCO)
   └─ load_coco_data() → images_by_id, anns_by_image, filename_to_id

2. Generar strips de referencia por clase
   └─ MatrixGenerator.extract_best_crops()
   └─ MatrixGenerator.generate_matrix(cls_id) → (strip_img, grid_info)

3. Para CADA imagen target:
   └─ Para CADA clase:
       ├─ Escalar strip al alto de la imagen
       ├─ Construir canvas: [strip_escalado | imagen_target]
       ├─ SAM3.predict_with_boxes(canvas, prompts) → boxes detectados
       ├─ Filtrar: solo lado derecho, size_filter
       └─ Añadir a raw_annotations

4. NMS global (coverage-based)
   └─ class_agnostic_nms(raw_annotations) → final_annotations

5. Guardar
   └─ results.json (COCO format)
   └─ Imágenes anotadas (con filtro de visualización)
   └─ metrics.csv + confusion_matrix.png
```

---

## 5. Decisiones de Diseño Que Debes Poder Justificar

1. **Visual Prompting vs Fine-tuning:** No se necesita entrenamiento adicional. SAM3 generaliza con ejemplos visuales.

2. **Coverage NMS vs IoU NMS:** Coverage = inter/min(area). Detecta cajas completamente contenidas dentro de otras, que IoU estándar ignora (IoU = inter/union penaliza diferencias de tamaño).

3. **Per-image-all-classes vs Per-class-all-images:** Procesamos por imagen para cargar cada target solo una vez en GPU. Más eficiente en memoria.

4. **Lazy Loading del modelo:** No cargar hasta que se necesite. Permite que la UI arranque rápido y que el dropdown se muestre sin esperar 5s.

5. **Separación CLI/UI con InferenceEngine:** Un solo punto de lógica, dos interfaces. Cambios en la inferencia se reflejan en ambos.

6. **Dos umbrales (detection vs visualization):** `CONFIDENCE_THRESHOLDS` controla qué acepta el modelo. `VISUALIZATION_THRESHOLDS` controla qué se dibuja. Permite ajustar la visualización sin re-ejecutar inferencia.
