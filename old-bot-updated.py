# Classify complaints and send default msg

# Importing the libraries
import pandas as pd
import json
# Importing the dataset
dataset = pd.read_csv('Consumer_Complaints.csv')

# class distribution
print(dataset.groupby('Product').size())

X = dataset[["Product", "Consumer complaint narrative"]]

#removing Payday loan,   Payday loan, title loan, or personal loan,   Prepaid card, 
#Student loan ,   Vehicle loan or lease,  Virtual currency  due to too little observations 
dt=X.dropna()
dt=dt[dt.Product != 'Payday loan']
dt=dt[dt.Product != 'Payday loan, title loan, or personal loan']
dt=dt[dt.Product != 'Prepaid card']
dt=dt[dt.Product != 'Student loan']
dt=dt[dt.Product != 'Vehicle loan or lease']
dt=dt[dt.Product != 'Virtual currency']

print(dt.groupby('Product').size())     

#resetting index to make consecutive
dt=dt.reset_index(drop=True)
    
df=dt[:1000]            #taking subset of original dataframe for ease of computation
print(df.groupby('Product').size())

#label encoding the Product column                                                                                                                                                                                                                                                                                     
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df["Product"] = lb_make.fit_transform(df["Product"])
df[["Product"]].head(11)


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

def format_complaint(complaint):

    complaint = re.sub('[^a-zA-Z]', ' ', complaint)
    complaint = complaint.lower()
    complaint = complaint.split()
    ps = PorterStemmer()
    complaint = [ps.stem(word) for word in complaint if not word in set(stopwords.words('english'))]
    complaint = ' '.join(complaint)
    return complaint
corpus = []

for i in range (0, len(df.values)):    
    review = format_complaint(df['Consumer complaint narrative'][i])
    corpus.append(review)      

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[0:len(df.values), 0].values              #change df to dt for full dataset

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

####################### SVM ######################## 72% accuracy
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

################### SEND DEFAULT MSG TO NEW COMPLAINTS #####################

def complaint(new_complaint):
    new_complaint=format_complaint(new_complaint) 
     
    new_corpus=[]    
    new_corpus.append(new_complaint)       
    new_comp = cv.transform(new_corpus).toarray()  
    new_prediction = classifier.predict(new_comp)
    
    dept=lb_make.inverse_transform(new_prediction)

    return (dept[0])
    
##################### CHATBOT CODE #####################     
import luis

# Use the URL LUIS gives you when you publish your app.
l = luis.Luis(url='https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/235f96f0-c3a8-4dda-99a7-d78cfd3f198a?subscription-key=83bdfb8db88a4b1d950097fa842529a5&verbose=true&timezoneOffset=330&q= ')

# Send text to be analyzed:
ticket_no = 0
while(1):
    r = input("Enter text to bot: ")
    if r=='exit':
        break
    res = l.analyze(r)

    if res.best_intent().intent == "Intro":
       response = 'Hey! How\'s it going? :)'
        
    elif res.best_intent().intent == "Goodbye":
       response = 'Happy to help :)'
        
    elif res.best_intent().intent == "Complaint":  
        ticket_no=ticket_no+1
        d=complaint(r)
        response = '\n\nYour request has been forwarded to the ' +d+ ' department for review.\nYour ticket number is %d'% (ticket_no)
# creating JSON object
        complaint_data = {ticket_no:r}

        #with open('Complaint_data.json', 'a') as f:
        #    json.dump(complaint_data, f)
    elif res.best_intent().intent == "None":
       response = 'Sorry, I can\'t understand what you\'re saying.' 
       
    #print (response)
    Response = {r:response}

    from bottle import route, run
    @route('/complaint')
    def comp():
        return complaint_data
      
    @route('/response')
    def res():
        return Response
    run(host='localhost', port=8080, debug=True)
