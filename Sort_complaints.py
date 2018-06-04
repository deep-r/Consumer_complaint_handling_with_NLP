# Classify complaints and send default msg

# Importing the libraries
import pandas as pd

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
vocab=[] 
for i in range(0,1000):     #taking only first 1000 rows since system is crashing due to too large data
    comp = re.sub('[^a-zA-Z]', ' ', df['Consumer complaint narrative'][i])
    comp = comp.lower()
    comp = comp.split()
    ps = PorterStemmer()
    comp = [ps.stem(word) for word in comp if not word in set(stopwords.words('english'))]
    comp = ' '.join(comp)
    corpus.append(comp)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[0:1000, 0].values              #change df to dt for full dataset
vocab=cv.get_feature_names()            #saves vocabulary into vocab so it can fit new complaints into it

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


####################### NB ######################### 58% accuracy

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    


####################### SVM ######################## 72% accuracy
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 


################### SEND DEFAULT MSG TO NEW COMPLAINTS #####################

new_complaint = input("Enter complaint: ")
    
corpus2 = []
comp2 = re.sub('[^a-zA-Z]', ' ',new_complaint)

comp2 = comp2.lower()
comp2 = comp2.split()
ps = PorterStemmer()
comp2 = [ps.stem(word) for word in comp2 if not word in set(stopwords.words('english'))]
comp2 = ' '.join(comp2)
corpus2.append(comp2)   

        
cv2 = CountVectorizer(vocabulary = vocab)
new_pred = cv2.fit_transform(corpus2).toarray()  


dept=(classifier.predict(new_pred))
print("Thankyou for reaching out to us. Your request has been forwarded to the below department for review.")

dept=lb_make.inverse_transform(dept)

print (dept)


