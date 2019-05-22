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
    - Cargamos la data y exploramos el contenido.
        ```python
             df1 = pd.read_csv('/data/out_AVP.csv')
             print("Total de datos",len(df1))
             df1.head(4)
        ``` 
    - Revisamos la cantidad de datos por cada clase que existe en el dataset.
        ```python
            c1=sum(df1['Clasificacion']=='Consulta')
            c2=sum(df1['Clasificacion']=='Cambio de plan')
            c3=sum(df1['Clasificacion']=='Creditos')
            c4=sum(df1['Clasificacion']=='Excepcion')

            print("Total de la clase \'Consulta\':",c1)
            print("Total de la clase \'Cambio de plan\':",c2)
            print("Total de la clase \'Creditos\':",c3)
            print("Total de la clase \'Excepcion\':",c4)
            lstctd=[('Consulta',c1),('Cambio de plan',c2),('Creditos',c3),('Excepcion',c4)]

            labels, ys = zip(*lstctd)
            xs = np.arange(len(labels)) 
            width = 1
            plt.bar(xs, ys, width, align='center')
            plt.xticks(xs, labels) 
            plt.yticks(ys)
            plt.show()
        ```
    - Realizamos la clasificación de la data por medio del campo Asunto. Debido a esto realizamos un cleaning del campo Asunto mediante el uso de expresiones regulares.
        ```python
             data1 = df1.Asunto.values.tolist()
             data1 = [re.sub('\S*@\S*\s?', '', sent) for sent in data1]
             data1 = [re.sub(r"http\S+", "", sent) for sent in data1]
             data1 = [re.sub(r"(<[^>]*>)", " ", sent) for sent in data1]
             data1 = [re.sub(r"({[^}]*})", " ", sent) for sent in data1]
             data1 = [re.sub('\s+', ' ', sent) for sent in data1]
             data1 = [re.sub("\'", "", sent) for sent in data1]
             data1 = [re.sub("\*", "", sent) for sent in data1]
             data1 = [re.sub("\/", "", sent) for sent in data1]
             data1 = [re.sub("\(", "", sent) for sent in data1]
             data1 = [re.sub("\)", "", sent) for sent in data1]
             data1 = [re.sub("\!", "", sent) for sent in data1]
             data1 = [re.sub("\-", "", sent) for sent in data1]
             data1 = [re.sub(r"(\[.*?\])", " ", sent) for sent in data1]
             data1 = [re.sub(r"(RE:[^\w])", "", sent) for sent in data1]
             data1 = [re.sub(r"(RV:[^\w])", "", sent) for sent in data1]
             data1 = [re.sub(r"([\d\.])", "", sent) for sent in data1]
             data1 = [re.sub("\:", "", sent) for sent in data1]
         ```

    - Creamos la lista que contendra los stopwords que seran eliminados de los Asuntos.
        ```python
            stopwords_es = set(stopwords.words('spanish'))
            stopwords_es_sw = set(get_stop_words('spanish'))
            stopSpanish = set(stopwords_es.union(stopwords_es_sw))
            stopSpanish = list(stopSpanish)
        ```
    - Realizamos la tokenizacion del campo Asunto mediante la funcion sent_to_words que esta definida en md_utils.
        ```python
            data_words = list(sent_to_words(data1))
        ```
    - Construimos los bigramas y trigramas de la lista tokenizada anteriormente.
        ```python
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
        ```
        
- md_utils : 
    - graph_error_models : Esta función genera una gráfica por cada tópico que se encuentra en la data de prueba. Cada grafica nos muestra los verdaderos positvos y los falsos positivos.
    - category_to_target : Esta función se encarga de generar las clases Y, Y1 que seran usadas en los modelos segun el tipo de variable que requiera.
    - corpus_to_input : Esta función se encarga de generar los inputs necesarios para los modelos utilizados.
