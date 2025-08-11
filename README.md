# Telecom X – Sistema Predictivo de Cancelación de Clientes (Churn Prediction)
1. Resumen Ejecutivo
Este proyecto implementa un flujo de trabajo integral de machine learning supervisado para predecir la cancelación de clientes en una empresa de telecomunicaciones. El proceso abarca desde la ingeniería de características hasta la interpretación de modelos usando SHAP, con el objetivo de identificar patrones de abandono y proponer estrategias de retención basadas en evidencia

2. Estructura del Proyecto

data                                # Datos limpios y procesados
notebooks                           # Jupyter Notebooks de análisis y modelado
models                              # Modelos entrenados (opcional)
outputs                             # Visualizaciones, reportes y métricas
informe_cancelacion_telecom_x.md    # Informe técnico detallado
requirements.txt                    # Dependencias necesarias
README.md   

3. Objetivo y Alcance
Objetivo principal: Predecir la probabilidad de cancelación (abandono de servicios) de cada cliente utilizando variables contractuales, demográficas y de consumo.

Alcance: Implementar modelos con alta capacidad predictiva y fácil interpretabilidad para su uso en entornos de negocio.

4. Arquitectura y Flujo de Trabajo

  4.1. Adquisición y Preprocesamiento de Datos

    4.1.1. Eliminación de identificadores irrelevantes.
    4.2.2. Codificación de variables categóricas con pd.get_dummies().
    4.2.3. Balanceo de clases con SMOTE.
    4.2.4. Escalado de variables mediante StandardScaler.

  4.2. Análisis Exploratorio de Datos (EDA)

    4.2.1. Visualización de distribuciones y detección de outliers.
    4.2.2. Correlaciones y análisis segmentado por subgrupos.
    4.2.3. Identificación de variables con mayor impacto en churn.

  4.3. Selección de Características

    4.3.1. Método SelectKBest (f_classif).
    4.3.2. K óptimo: 29 variables seleccionadas.

  4.4. Entrenamiento y Optimización de Modelos

    4.4.1. Modelos evaluados:

      4.4.1.1. Logistic Regression
      4.4.1.2. Random Forest
      4.4.1.3. Gradient Boosting
      4.4.1.4. K-Nearest Neighbors
      4.4.1.5. Ensemble Voting Classifier (LR + GB)

    4.4.2. Búsqueda de hiperparámetros con GridSearchCV y validación cruzada estratificada.

  4.5. Evaluación y Validación

      4.5.1. Métricas: Accuracy, Precision, Recall, F1, AUC.
      4.5.2. Matrices de confusión y curvas ROC.
      4.5.3 Interpretabilidad con SHAP values y coeficientes de regresión.

5. Resultados Principales

  Modelo	              Accuracy	Precision	  Recall	  F1 Score	
  Gradient Boosting	    0.781043	0.585789	  0.602496	0.594025
  Regresión Logística	  0.786730	0.594228	  0.623886	0.608696
  Ensemble (LR + GB)	  0.797156	0.629126	  0.577540	0.602230

Conclusión técnica: 
El modelo Ensemble (LR + GB) logra el mejor rendimiento global, combinando interpretabilidad y capacidad predictiva.

6. Variables con Mayor Impacto en Churn
Contratos month-to-month.

  6.1. Método de pago Electronic Check.
  6.2. Ausencia de servicios adicionales (soporte técnico, seguridad, respaldo).
  6.3. Baja permanencia (meses como cliente).
  6.4 Altos cargos mensuales sin beneficios adicionales.

7. Recomendaciones Estratégicas
Migrar clientes a contratos de largo plazo con incentivos.

  7.1. Promocionar paquetes de servicios adicionales.
  7.2. Incentivar pagos automáticos con descuentos o beneficios.
  7.3. Monitoreo proactivo de clientes con altos cargos.
  7.4. Programas de fidelización para nuevos clientes.

8. Requisitos Técnicos
Lenguaje: Python 3.8+

  8.1. Librerías:

    8.1.1. pandas, numpy, matplotlib, seaborn
    8.1.2. scikit-learn, imblearn
    8.1.3. shap

