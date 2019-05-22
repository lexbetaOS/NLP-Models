# NLP-Models

### Overview
El presente notebook tiene como objetivo analizar un dataset de correos en español mediante funcionalidades de NLP y el uso de modelos supervizados en machine learning.

### Installation Dependencies:
- Python 3+
- Pandas
- Numpy 1+
- tensorflow
- gensim 3+

### Usage

- main:
    - Cargamos la data y exploramos el contenido:
        ```python
             df1 = pd.read_csv('/data/out_AVP.csv')
             print("Total de datos",len(df1))
             df1.head(4)
        ```
    - Revisamos la cantidad de datos por cada clase que contiene el dataset para ver
- md_utils : 
    - graph_error_models : Esta función genera una gráfica por cada tópico que se encuentra en la data de prueba. Cada grafica nos muestra los verdaderos positvos y los falsos positivos.
    - category_to_target : Esta función se encarga de generar las clases Y, Y1 que seran usadas en los modelos segun el tipo de variable que requiera.
    - corpus_to_input : Esta función se encarga de generar los inputs necesarios para los modelos utilizados.
