# #preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# #Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier

# #Analysis/Optimization
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import classification_report

# #Visualisation
# import seaborn as sns
# import matplotlib.pyplot as plt

#Other
import numpy as np
import pandas as pd
import pickle
from stop_words import get_stop_words
import re
import snowballstemmer
import time






class Classifier_pipe():
    
    VINTED_LANGUAGES = ["lithuanian", "polish", "czech", "french", "dutch", "german", "english", "spanish"]
    
    
    def __init__(self, dataframe, x_col_name, model_path):
        """
        Input:
            dataframe
            
            
        """
        
        #save init data
        self.df = dataframe
        self.x_col_name = x_col_name
        self.model_path = model_path
        
        #setup classifier
        self._load_model()
        #define stop_words
        self._stop_word_generator()
        
        
        
    def predict_top2(self):
        
        t1 = time.time()
        #Run preprocess
        self._preprocess()
        print(f'preprocess : {time.time()-t1:.3f} s')
        
        
        t2 = time.time()
        #Get top2 predictions
        probs = self.model.predict_proba(self.X_test)
        y_pred_top2 = np.argsort(-probs, axis=1)[:,:2]
        self.y_pred1 = self.model.classes_[y_pred_top2][:,0]
        self.y_pred2 = self.model.classes_[y_pred_top2][:,1]
        
        print(f'inference  : {time.time()-t2:.3f} s')
        
        t3 = time.time()
        #Save to .parquet
        self._save_as_parquet()
        
        print(f'saving     : {time.time()-t3:.3f} s')
        
        
    def _load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
    
    def _preprocess(self):
        
        self.X_test = self.df[self.x_col_name].apply(self._clean_text)
#         print(self.X_test)
        

    def _stop_word_generator(self):
        
        all_stop_words = []
        for language in self.VINTED_LANGUAGES:
            try:
                 stop_words = get_stop_words(language)
            except:
                pass
            else:
                all_stop_words += stop_words

        self.all_stop_words = set(all_stop_words)
        
        
    #TO DO: clean_text kaip isorine
        
    def _clean_text(self,text):

        stemmer = snowballstemmer.stemmer('english')
        text = text.lower()
        text = re.sub(r"\d", " ", text)
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"\b\w{1,3}\b", " ", text)
        text = " ".join(stemmer.stemWords((word for word in text.split() if word not in self.all_stop_words)))
       
        return text
        
    
    def _save_as_parquet(self):
        
        self.df['category1'] = self.y_pred1
        self.df['category2'] = self.y_pred2 
        self.df.to_parquet('predictions.parquet')
    
        