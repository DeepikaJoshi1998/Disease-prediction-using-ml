# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:40:06 2019

@author: achin
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 

training = pd.read_csv('Training.csv')
testing  = pd.read_csv('Testing.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
#print(clf.score(x_train,y_train))
#print ("cross result========")
#scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())
#print(clf.score(testx,testy))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

#feature_importances
#for f in range(10):
#    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

print(" Hey there!! I m  CORTONA. \n I m here to Predict Disease from symptoms .")
print("If you want to exit, TYPE THANKS!");

flag=True
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' or user_response=='bye' ):
            flag=False
            print("CORTONA: You are welcome  & consult your doctor once....")
            break;
        else:
            if(greeting(user_response)!=None):
                print("CORTONA: "+greeting(user_response))
                print("How are you ?? ........");
                user_response = input()
                if(user_response=='not_well ' or user_response=='bad'):
                  print("Please reply Yes or No for the following symptoms") 
                  def print_disease(node):
                          #print(node)
                          node = node[0]
                          #print(len(node))
                          val  = node.nonzero() 
                          #print(val)
                          disease = le.inverse_transform(val[0])
                          return disease
                  def tree_to_code(tree, feature_names):
                     tree_ = tree.tree_
                     #print(tree_)
                     feature_name = [
                             feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                             for i in tree_.feature
                             ]
                     #print("def tree({}):".format(", ".join(feature_names)))
                     symptoms_present = []
                     def recurse(node, depth):
                         if tree_.feature[node] != _tree.TREE_UNDEFINED:
                             name = feature_name[node]
                             threshold =tree_.threshold[node]
                             print("Are you facing " +name + " Symptom ?")
                             ans = input()
                             ans = ans.lower()
                             if ans == 'yes':
                                 val = 1
                                 present_disease = print_disease(tree_.value[node])
                                 print( "You might be suffering from  " +  present_disease )
                                 red_cols = reduced_data.columns 
                                 symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                                 print("symptoms present above \n  " + str(list(symptoms_present)))
                                 print("symptoms that might be faced by you \n  "  +  str(list(symptoms_given)) )  
                                 confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
                                 print("\n  Bye! take care & don't forget to consult your doctor once....\n \n  !! THANK YOU !!....")
                                 flag=False;
                                 
                             else:
                                     val = 0
                                     if val <= threshold:
                                         recurse(tree_.children_left[node], depth + 1)
                                     else:
                                          symptoms_present.append(name)
                                          recurse(tree_.children_right[node], depth + 1)
                                                   
                     recurse(0, 1)          

        tree_to_code(clf,cols)
        
    else:
                print("CORTONA: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
else:
        flag=False
        print(" Bye! take care & consult your doctor once....")

  


     

            
   



  
        
    
  
        
    
    

