# Telecom X – Sistema Predictivo de Cancelación de Clientes (Churn Prediction)

## 1. Resumen Ejecutivo 

Este proyecto implementa un flujo de trabajo integral de machine learning supervisado para predecir la cancelación de clientes en una empresa de telecomunicaciones. El proceso abarca desde la ingeniería de características hasta la interpretación de modelos usando SHAP, con el objetivo de identificar patrones de abandono y proponer estrategias de retención basadas en evidencia

## 2. Objetivo y Alcance

> Objetivo principal: Predecir la probabilidad de cancelación (abandono de servicios) de cada cliente utilizando variables contractuales, demográficas y de consumo.

> Alcance: Implementar modelos con alta capacidad predictiva y fácil interpretabilidad para su uso en entornos de negocio.

## 3. Arquitectura y Flujo de Trabajo**

### 3.1. Adquisición y Preprocesamiento de Datos
> 3.1.1. Eliminación de identificadores irrelevantes.

> 3.2.2. Codificación de variables categóricas con pd.get_dummies().
>
> 3.2.3. Balanceo de clases con SMOTE.
>
> 3.2.4. Escalado de variables mediante StandardScaler.


### 3.2. Análisis Exploratorio de Datos (EDA)

> 3.2.1. Visualización de distribuciones y detección de outliers.
>
> 3.2.2. Correlaciones y análisis segmentado por subgrupos.
>
> 3.2.3. Identificación de variables con mayor impacto en churn.

### 3.3. Selección de Características

> 4.3.1. Método SelectKBest (f_classif).
> 
> 4.3.2. K óptimo: 29 variables seleccionadas.

### 3.4. Entrenamiento y Optimización de Modelos

> 3.4.1. Modelos evaluados:

    3.4.1.1. Logistic Regression
    3.4.1.2. Random Forest
    3.4.1.3. Gradient Boosting
    3.4.1.4. K-Nearest Neighbors
    3.4.1.5. Ensemble Voting Classifier (LR + GB)

> 3.4.2. Búsqueda de hiperparámetros con GridSearchCV y validación cruzada estratificada.

### 3.5. Evaluación y Validación

    3.5.1. Métricas: Accuracy, Precision, Recall, F1, AUC.
    3.5.2. Matrices de confusión y curvas ROC.
    3.5.3 Interpretabilidad con SHAP values y coeficientes de regresión.

## 4. Resultados Principales

  Modelo	              Accuracy	Precision	  Recall	  F1 Score	
  Gradient Boosting	    0.781043	0.585789	  0.602496	0.594025
  Regresión Logística	  0.786730	0.594228	  0.623886	0.608696
  Ensemble (LR + GB)	  0.797156	0.629126	  0.577540	0.602230

Conclusión técnica: 
El modelo Ensemble (LR + GB) logra el mejor rendimiento global, combinando interpretabilidad y capacidad predictiva.

## 5. Variables con Mayor Impacto en Churn
Contratos mes a mes.

> 5.1. Método de pago Electronic Check.
> 
> 5.2. Ausencia de servicios adicionales (soporte técnico, seguridad, respaldo).
> 
> 5.3. Baja permanencia (meses como cliente).
> 5.4 Altos cargos mensuales sin beneficios adicionales.

## 6. Recomendaciones Estratégicas
Migrar clientes a contratos de largo plazo con incentivos.

> 6.1. Promocionar paquetes de servicios adicionales.
> 
> 6.2. Incentivar pagos automáticos con descuentos o beneficios.
> 
> 6.3. Monitoreo proactivo de clientes con altos cargos.
> 
> 6.4. Programas de fidelización para nuevos clientes.

## 7. Requisitos Técnicos
Lenguaje: Python 3.8+

> 7.1. Librerías:

      7.1.1. pandas, numpy, matplotlib, seaborn
      7.1.2. scikit-learn, imblearn
      7.1.3. shap

