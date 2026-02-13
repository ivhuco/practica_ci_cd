# 🚢 Titanic ML Project

Proyecto completo de Machine Learning usando el dataset del Titanic con procesamiento de datos, entrenamiento, evaluación y CI/CD con GitHub Actions.

## 📋 Descripción

Este proyecto implementa un pipeline de ML para predecir la supervivencia de pasajeros del Titanic. Incluye:

- **Procesamiento de datos**: Limpieza, feature engineering, y transformaciones
- **Entrenamiento**: Random Forest con validación cruzada
- **Evaluación**: Métricas detalladas y reportes visuales
- **CI/CD**: GitHub Actions para testing, entrenamiento y evaluación automática

## 📂 Estructura del Proyecto

```
titanic-ml-project/
├── data/
│   ├── raw/              # Datos originales
│   └── processed/        # Datos procesados
├── src/
│   ├── data_loader.py    # Carga de datos
│   ├── preprocessing.py  # Preprocesamiento
│   ├── model.py          # Definición del modelo
│   ├── train.py          # Script de entrenamiento
│   └── evaluate.py       # Evaluación del modelo
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── scripts/
│   ├── download_data.py  # Descargar dataset
│   └── run_pipeline.py   # Ejecutar pipeline completo
├── models/               # Modelos entrenados
├── reports/              # Reportes de evaluación
└── .github/workflows/    # GitHub Actions
    ├── ci.yml
    ├── train-model.yml
    └── evaluate-model.yml
```

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone <your-repo-url>
cd titanic-ml-project

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### 1. Descargar datos

```bash
python scripts/download_data.py
```

### 2. Ejecutar pipeline completo

```bash
python scripts/run_pipeline.py
```

Este comando ejecutará:

- Preprocesamiento de datos
- Entrenamiento del modelo
- Evaluación y generación de reportes

### 3. Entrenar modelo individualmente

```bash
python src/train.py
```

### 4. Evaluar modelo

```bash
python src/evaluate.py
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=src --cov-report=html

# Linting
flake8 src/ tests/ --max-line-length=100
```

## 🔄 GitHub Actions Workflows

### CI Testing (`ci.yml`)

- **Trigger**: Push y Pull Requests
- **Acciones**:
  - Setup de Python
  - Instalación de dependencias
  - Linting con flake8
  - Ejecución de tests con pytest
  - Reporte de cobertura

### Model Training (`train-model.yml`)

- **Trigger**: Manual o programado
- **Acciones**:
  - Descarga de datos
  - Preprocesamiento
  - Entrenamiento del modelo
  - Guardado del modelo como artifact
  - Publicación de métricas

### Model Evaluation (`evaluate-model.yml`)

- **Trigger**: Después del entrenamiento
- **Acciones**:
  - Carga del modelo entrenado
  - Evaluación en conjunto de test
  - Generación de reportes
  - Publicación de resultados

## 📊 Características del Modelo

- **Algoritmo**: Random Forest Classifier
- **Features**:
  - Pclass, Sex, Age, SibSp, Parch
  - FamilySize (engineered)
  - IsAlone (engineered)
  - Title extraído del nombre (engineered)
  - Fare, Embarked

- **Métricas**:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC

## 🛠️ Tecnologías

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- pytest
- GitHub Actions

## 📝 Licencia

MIT License

## 👤 Autor

Tu nombre aquí
