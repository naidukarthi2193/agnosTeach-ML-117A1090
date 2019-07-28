# Consumer Review Prediction System

Word2Vec and ANN based Supervised Sentiment Analysis and added Flask REST APIs for Prediction Outputs 

### Requirements : 
- Numpy
- Matplotlib
- NLTK 
- Gensim
- Sklearn
- Keras
- Spacy
- Flask

### Preprocessing PipeLine
1. Strip Punctuations
2. Tokenization 
3. Creating Bi-Grams
4. Removing Stop Words
5. Lemmatization

### Classification

 ![Image](https://drive.google.com/uc?id=19rNJB9rpVnca0IkuAsqaCjWjZrRIohEe)
 
 From the Above Distributions to reduce Data Unbalance **Ratings Below 5 == 0 and Ratings 5==1**


### Metrics 

![Image](https://drive.google.com/uc?id=1kW_O2WhehiOCy3hXm7TXych5zt_WSE6C)

After Training for 60 Epocchs we can See the Validation Accuracy flattens At ~85% after 35 Epocchs

**Confusion Matrix**
![Image](https://drive.google.com/uc?id=18iciY3MdkxdnGkrjXi41oEohMfDIvGx4)

We can See from the Above Metrics that model performs Well and Can be Improved further later

### Flask REST API
We can Send Responses from POSTMAN in JSON format inside **"input"** key and get Output as **"review"** key

#### Positive PostMan Request
![Image](https://drive.google.com/uc?id=1LR1JVll_ZpAGPha83xue5RzPGIj1RZ7O)
#### Negative PostMan Request 
 ![Image](https://drive.google.com/uc?id=13cpHZ0qD0I9QldBqrLUmaEwiDgDm1HsJ)


## Final Max Validation Accuracy On Validation Set Was **72.495%**
