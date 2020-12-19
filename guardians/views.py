import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from guardians.serializers import IndexSerializer
from services.status import check
from services.summarizer import generate_summary


class IndexApi(APIView):
    def post(self, request, *args, **kwargs):
        serializer = IndexSerializer(data=request.data)
        if serializer.is_valid():
            summary = generate_summary(request.data['content'])
            imp = check(request.data['content'])
            ans = {}
            ans["status"] = imp[0]
            ans["summary"] = summary
            return Response(ans, status=status.HTTP_200_OK)
        return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)



def check(a):
    df = pd.read_csv("news.csv")

    x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    #DataFlair - Fit and transform train set, transform test set
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_test=tfidf_vectorizer.transform(x_test)


    #DataFlair - Initialize a PassiveAggressiveClassifier
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train,y_train)
    #DataFlair - Predict on the test set and calculate accuracy
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print(f'Accuracy: {round(score*100,2)}%')


    input_data = [a]
    vectorized_input_data = tfidf_vectorizer.transform(input_data)
    prediction = pac.predict(vectorized_input_data)
    print(prediction)
    return prediction[0]
    