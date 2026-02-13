# 🚢 Proyecto Titanic ML - Predicción de Supervivencia

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)](https://flake8.pycqa.org/)

Proyecto completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic, implementando mejores prácticas de ingeniería de software, testing automatizado y CI/CD con GitHub Actions.

> [!NOTE]
> **🎯 Proyecto académico de práctica**: Este repositorio fue creado como ejercicio práctico para aprender CI/CD, testing y mejores prácticas en proyectos de Machine Learning.

## 📑 Tabla de Contenidos

- [Características](#-características)
- [Inicio Rápido](#-inicio-rápido)
  - [Docker 🐳](#opción-1-usando-docker--recomendado)
  - [Instalación Local](#opción-2-instalación-local)
- [Documentación](#-documentación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso Detallado](#-uso-detallado)
- [Workflows de CI/CD](#-workflows-de-cicd)
- [Testing](#-testing)
- [Modelos Soportados](#-modelos-soportados)
- [Resultados](#-resultados)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## ✨ Características

### Pipeline de ML Completo

- ✅ **Descarga automática de datos** desde fuentes confiables
- ✅ **Preprocesamiento robusto** con feature engineering
- ✅ **Múltiples modelos** ML (Random Forest, Logistic Regression, Gradient Boosting)
- ✅ **Validación cruzada** para estimación robusta del rendimiento
- ✅ **Evaluación completa** con métricas y visualizaciones

### Ingeniería de Software

- ✅ **Código modular** y reutilizable
- ✅ **Tests unitarios** con pytest (cobertura > 80%)
- ✅ **Linting** con flake8
- ✅ **Type hints** y documentación
- ✅ **Git-friendly** con .gitignore configurado

### CI/CD Automatizado

- ✅ **Testing automático** en cada push/PR
- ✅ **Entrenamiento programado** (semanal)
- ✅ **Evaluación automática** después del entrenamiento
- ✅ **Artifacts** versionados (modelos, métricas, reportes)
- ✅ **Docker** para deployment reproducible

## 🚀 Inicio Rápido

### Opción 1: Usando Docker 🐳 (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/ivhuco/practica_ci_cd.git
cd practica_ci_cd

# Construir y ejecutar con Docker Compose
docker-compose up train evaluate

# O usar Docker directamente
docker build -t titanic-ml:dev .
docker run -it --rm titanic-ml:dev
```

Ver [docs/DOCKER.md](docs/DOCKER.md) para documentación completa de Docker.

### Opción 2: Instalación Local

#### Prerequisitos

- Python 3.9 o superior
- pip
- Git

#### Instalación en 3 Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/ivhuco/practica_ci_cd.git
cd practica_ci_cd

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar pipeline completo
python scripts/download_data.py  # Descargar datos
python scripts/run_pipeline.py   # Entrenar y evaluar
```

### Verificación

Si todo funcionó correctamente, deberías ver:

```
✓ Trained model saved in 'models/' directory
✓ Evaluation metrics in 'reports/evaluation_results.json'
✓ Visualizations in 'reports/*.png'
```

## 📚 Documentación

Este proyecto cuenta con documentación completa en español:

### Documentación Principal

| Documento | Descripción |
|-----------|-------------|
| **[📖 README.md](README.md)** | Este archivo - Visión general del proyecto |
| **[🏗️ ARQUITECTURA.md](ARQUITECTURA.md)** | Arquitectura técnica detallada con diagramas |
| **[📚 GUIA_USO.md](GUIA_USO.md)** | Guía paso a paso de instalación y uso |

### Documentación de Módulos

Documentación detallada de cada componente:

- [📦 Data Loader](docs/modulos/DATA_LOADER.md) - Carga y gestión de datos
- [🔧 Preprocessing](docs/modulos/PREPROCESSING.md) - Pipeline de preprocesamiento
- [🤖 Model](docs/modulos/MODEL.md) - Definición y gestión de modelos
- [🎓 Train](docs/modulos/TRAIN.md) - Proceso de entrenamiento
- [📊 Evaluate](docs/modulos/EVALUATE.md) - Evaluación y métricas

### Documentación de Scripts

- [📜 Download Data](docs/scripts/DOWNLOAD_DATA.md) - Descarga del dataset
- [📜 Run Pipeline](docs/scripts/RUN_PIPELINE.md) - Pipeline end-to-end

### Documentación de CI/CD

- [🔄 CI Workflow](docs/ci-cd/WORKFLOW_CI.md) - Testing y linting automático
- [🚂 Train Workflow](docs/ci-cd/WORKFLOW_TRAIN.md) - Entrenamiento automático
- [📈 Evaluate Workflow](docs/ci-cd/WORKFLOW_EVALUATE.md) - Evaluación automática

### Otros

- [🧪 Testing](docs/tests/TESTING.md) - Guía de testing

## 📂 Estructura del Proyecto

```
practica_ci_cd/
├── 📄 README.md                 # Este archivo
├── 📄 ARQUITECTURA.md           # Documentación de arquitectura
├── 📄 GUIA_USO.md              # Guía de uso detallada
├── 📄 requirements.txt          # Dependencias Python
├── 📄 .gitignore               # Archivos ignorados por Git
│
├── 📁 .github/workflows/        # GitHub Actions
│   ├── ci.yml                  # Testing y linting automático
│   ├── train-model.yml         # Entrenamiento automático
│   └── evaluate-model.yml      # Evaluación automática
│
├── 📁 src/                      # Código fuente principal
│   ├── data_loader.py          # Carga de datos
│   ├── preprocessing.py        # Preprocesamiento y feature engineering
│   ├── model.py                # Definición de modelos
│   ├── train.py                # Script de entrenamiento
│   └── evaluate.py             # Script de evaluación
│
├── 📁 scripts/                  # Scripts de utilidad
│   ├── download_data.py        # Descarga del dataset
│   └── run_pipeline.py         # Pipeline end-to-end
│
├── 📁 tests/                    # Tests unitarios
│   ├── test_preprocessing.py   # Tests del preprocessor
│   └── test_model.py           # Tests del modelo
│
├── 📁 data/                     # Datos (git-ignored)
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos procesados
│
├── 📁 models/                   # Modelos entrenados (git-ignored)
├── 📁 reports/                  # Reportes y métricas
└── 📁 docs/                     # Documentación detallada
    ├── modulos/                # Docs de módulos Python
    ├── scripts/                # Docs de scripts
    ├── ci-cd/                  # Docs de workflows
    └── tests/                  # Docs de testing
```

## 💻 Uso Detallado

### Opción 1: Pipeline Completo (Recomendado)

```bash
# Descargar datos y ejecutar todo el pipeline
python scripts/download_data.py
python scripts/run_pipeline.py
```

### Opción 2: Ejecución Paso a Paso

```bash
# 1. Descargar datos
python scripts/download_data.py

# 2. Entrenar modelo (Random Forest por defecto)
python src/train.py

# 3. Evaluar modelo
python src/evaluate.py --use-test

# 4. Ver resultados
cat reports/evaluation_results.json
open reports/confusion_matrix.png  # macOS
```

### Entrenar Diferentes Modelos

```bash
# Random Forest (por defecto)
python src/train.py --model random_forest --cv-folds 5

# Logistic Regression
python src/train.py --model logistic_regression --cv-folds 5

# Gradient Boosting
python src/train.py --model gradient_boosting --cv-folds 5
```

> [!TIP]
> Para uso avanzado y personalización, consulta [GUIA_USO.md](GUIA_USO.md)

## 🔄 Workflows de CI/CD

### CI - Testing y Linting

**Trigger**: Push o Pull Request a `main` o `develop`

```yaml
Matriz de Python: 3.9, 3.10, 3.11
├── Linting con flake8
├── Tests con pytest
├── Reporte de cobertura
└── Upload a Codecov
```

[📖 Documentación detallada del workflow CI](docs/ci-cd/WORKFLOW_CI.md)

### Train Model - Entrenamiento Automático

**Trigger**: Manual o programado (domingos a las 00:00 UTC)

```yaml
├── Descarga de datos
├── Entrenamiento del modelo
├── Guardado de artifacts (modelo + métricas)
└── Publicación de métricas
```

[📖 Documentación detallada del workflow Train](docs/ci-cd/WORKFLOW_TRAIN.md)

### Evaluate Model - Evaluación Automática

**Trigger**: Después de entrenamiento exitoso o manual

```yaml
├── Carga del modelo entrenado
├── Evaluación en conjunto de test
├── Generación de visualizaciones
└── Comentario automático en PR (si aplica)
```

[📖 Documentación detallada del workflow Evaluate](docs/ci-cd/WORKFLOW_EVALUATE.md)

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=src --cov-report=html

# Ver reporte HTML
open htmlcov/index.html
```

### Linting

```bash
# Verificar estilo de código
flake8 src/ tests/ --max-line-length=100

# Solo errores críticos
flake8 src/ tests/ --select=E9,F63,F7,F82 --show-source
```

[📖 Guía completa de testing](docs/tests/TESTING.md)

## 🤖 Modelos Soportados

| Modelo | Ventajas | Accuracy típica | Tiempo de entrenamiento |
|--------|----------|-----------------|------------------------|
| **Random Forest** | Alta precisión, interpretable | ~82-84% | Medio (~10s) |
| **Logistic Regression** | Rápido, baseline sólido | ~78-80% | Rápido (~1s) |
| **Gradient Boosting** | Máxima precisión | ~83-85% | Lento (~30s) |

### Features Utilizados

El modelo utiliza las siguientes características:

**Features originales:**

- `Pclass` - Clase del pasajero
- `Sex` - Género
- `Age` - Edad
- `SibSp` - Número de hermanos/cónyuge a bordo
- `Parch` - Número de padres/hijos a bordo
- `Fare` - Tarifa pagada
- `Embarked` - Puerto de embarque

**Features engineered:**

- `FamilySize` - Tamaño total de la familia
- `IsAlone` - Indicador de viaje solo
- `Title` - Título extraído del nombre (Mr, Mrs, Miss, Master, etc.)

[📖 Detalles del preprocesamiento](docs/modulos/PREPROCESSING.md)

## 📊 Resultados

### Métricas de Rendimiento

Resultados típicos del modelo Random Forest:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 82.68% |
| **Precision** | 80.23% |
| **Recall** | 76.92% |
| **F1-Score** | 78.54% |
| **ROC-AUC** | 87.45% |

### Importancia de Features

Top 5 features más importantes:

1. `Sex` (género) - 25.4%
2. `Title` (título) - 18.7%
3. `Fare` (tarifa) - 15.3%
4. `Age` (edad) - 12.8%
5. `Pclass` (clase) - 11.2%

### Visualizaciones

El proyecto genera automáticamente:

- 📊 **Matriz de Confusión** - Clasificación detallada
- 📈 **Curva ROC** - Rendimiento del clasificador
- 📉 **Importancia de Features** - Peso de cada característica

## 🛠️ Tecnologías Utilizadas

### Core

- **Python 3.9+** - Lenguaje de programación
- **scikit-learn** - Algoritmos de ML
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas

### Visualización

- **matplotlib** - Gráficas
- **seaborn** - Visualizaciones estadísticas

### Testing & Quality

- **pytest** - Framework de testing
- **pytest-cov** - Cobertura de código
- **flake8** - Linting

### CI/CD

- **GitHub Actions** - Automatización

### Otros

- **joblib** - Serialización de modelos

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Antes de contribuir

- Ejecuta los tests: `pytest tests/ -v`
- Verifica el linting: `flake8 src/ tests/ --max-line-length=100`
- Actualiza la documentación si es necesario

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Ivan Hurtado**

- GitHub: [@ivhuco](https://github.com/ivhuco)
- Repositorio: [practica_ci_cd](https://github.com/ivhuco/practica_ci_cd)

## 🙏 Agradecimientos

- Dataset del Titanic de [Kaggle](https://www.kaggle.com/c/titanic)
- Comunidad de scikit-learn
- GitHub Actions por la infraestructura de CI/CD

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!

**[📖 Ver documentación completa](ARQUITECTURA.md)** | **[🚀 Guía de uso](GUIA_USO.md)** | **[🐛 Reportar un bug](https://github.com/ivhuco/practica_ci_cd/issues)**
