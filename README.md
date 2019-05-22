# NLP-Models

### Overview
El presente notebook tiene como objetivo analizar un dataset de correos en español mediante funcionalidades de NLP y el uso de modelos supervizados en machine learning.

### Installation Dependencies:
- Python 3+
- Pandas
- Numpy 1+
- tensorflow 1+
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
            c1=sum(df1['Clasificacion'] == 'Consulta')
            c2=sum(df1['Clasificacion'] == 'Cambio de plan')
            c3=sum(df1['Clasificacion'] == 'Creditos')
            c4=sum(df1['Clasificacion'] == 'Excepcion')

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
    - Los bigramas son dos palabras que ocurren juntas frecuentemente. Trigramas son tres palabras que ocurren frecuentemente. Construimos los modelos bigramas y trigramas de la lista tokenizada anteriormente.
        ```python
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
        ```
     - Una vez creado el modelo bigram y trigram, llamamos a las funciones definidas en md_utils para eliminar los stopwords, aplicar trigram y lemmatizacion manteniendo los sustantivos, adjetivos, verbos y advervios. 
        ```python
            data_words_nostops = remove_stopwords(data_words,stopSpanish)
            
            #data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
            data_words_trigrams = make_trigrams(data_words_nostops,bigram_mod,trigram_mod)

            nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])

            #data_lemmatized = lemmatization(data_words_bigrams,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            data_lemmatized = lemmatization(data_words_trigrams,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        ```
    - Creamos el diccionario de toda la data lematizada, creamos el bag of words de la data en corpus. 
        ```python
            # Create Dictionary
            id2word = corpora.Dictionary(data_lemmatized)

            # Create Corpus
            texts = data_lemmatized

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

        ```
    - Aplicamos TF-IDF.
        ```python
            tfidf = models.TfidfModel(corpus)
            corpus = tfidf[corpus]
        ```
    - Para aplicar los modelos de clasificación supervisada necesitamos preparar las variables segun el tipo de variable que el modelo solicite. Llamamos a las funciones definidas en md_utils.
        ```python
            X = corpus_to_input(corpus,id2word)
            Y = df1.Clasificacion.values.tolist()
            Y1,Y2=category_to_target(Y)
        ```
    - Con la data preparada obtenemos el training-set y el test-set de forma aleatoria para aplicar los modelos.
        ```python
            indices = np.random.permutation(X.shape[0])
            training_idx, test_idx = indices[:4748], indices[4748:]
            training_x, test_x = X[training_idx,:], X[test_idx,:]
            training_y1, test_y1 = Y1[training_idx], Y1[test_idx]
            training_y2, test_y2 = Y2[training_idx], Y2[test_idx]graph_error_models(2,2,test_y1,y_pred)
        ```
    - El primer modelo que aplicamos es una red neuronal. Iniciamos los valores necesarios para el modelo.
        ```python
            # Parameters
            total_words=len(id2word)
            learning_rate = 0.01
            training_epochs = 10
            batch_size = 150
            display_step = 1

            # Network Parameters
            n_hidden_1 = 100      # 1st layer number of features
            n_hidden_2 = 100       # 2nd layer number of features
            n_input = total_words # Words in vocab
            n_classes = 4         # Categories: graphics, sci.space and baseball

            input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
            output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output") 

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

        ```
        
    - Construimos el modelo.
        ```python
            prediction = multilayer_perceptron_nn(input_tensor, weights, biases)

            # Define loss and optimizer
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            # Initializing the variables
            init = tf.global_variables_initializer()

            # [NEW] Add ops to save and restore all the variables
            saver = tf.train.Saver()
        ```
    - Entrenamos el modelo.
        ```python
            # Launch the graph
            with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(training_x)/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = training_x[i*batch_size:i*batch_size+batch_size]
                    batch_y = training_y2[i*batch_size:i*batch_size+batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                 # Display logs per epoch step
                 if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "loss=","{:.9f}".format(avg_cost))
             print("Optimization Finished!")

             # Test model
             correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
             # Calculate accuracy
             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
             total_test_data = len(test_x)
             batch_x_test = test_x
             batch_y_test = test_y2
             print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
    
             # [NEW] Save the variables to disk
             save_path = saver.save(sess, "/content/gdrive/My Drive/Colab Notebooks/lda2vec/out3/model.ckpt")
             print("Model saved in path: %s" % save_path)
        ```
    - El segundo modelo que aplicamos es SVM.
        ```python
             clf_top=svm.SVC(kernel='linear',C=0.1,decision_function_shape='ovr')
             clf_top.fit(training_x,training_y1)

             y_predsvm = clf_top.predict(test_x)

             train_acc=clf_top.score(training_x,training_y1)
             test_acc=clf_top.score(test_x,test_y1)

             print(train_acc*100)
             print(test_acc*100)
        ```
    - El tercer modelo que aplicamos es Naive Bayes.
        ```python
             #Create a Gaussian Classifier
             gnb = GaussianNB()

             #Train the model using the training sets
             gnb.fit(training, training_y1)

             #Predict the response for test dataset
             y_pred = gnb.predict(test)
             
             # Model Accuracy, how often is the classifier correct?
             print("Accuracy:",metrics.accuracy_score(test_y1, y_pred))
        ```
        
    - Graficamos los verdaderos positivos y falsos positivos de cada modelo.
        ```python
        
             graph_error_models(test_y1,classification) ## Modelo red neuronal
             graph_error_models(test_y1,y_predsvm)  ## Modelo SVM
             graph_error_models(test_y1,y_pred)  ## Modelo NB
        ```
    
- md_utils : 
    - graph_error_models : Esta función genera una gráfica por cada tópico que se encuentra en la data de prueba. Cada grafica nos muestra los verdaderos positvos y los falsos positivos.
    - category_to_target : Esta función se encarga de generar las clases Y, Y1 que seran usadas en los modelos segun el tipo de variable que requiera.
    - corpus_to_input : Esta función se encarga de generar los inputs necesarios para los modelos utilizados.
