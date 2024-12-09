import numpy as np
import pandas as pd

# ok lets reset life back to normal

from flask import Flask, send_from_directory

import pywebio
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from pywebio.exceptions import SessionClosedException

import pickle
import warnings
import argparse
import locale


# Load the model pipeline from the file
with open('nlp_pipeline.pkl', 'rb') as f:
    loaded_pipe = pickle.load(f)


import string
from nltk.corpus import stopwords

# function to remove punctuation and stopwords
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

app = Flask(__name__)

## PYWEBIO CODE

pywebio.config(theme='sketchy')
# Function to predict the insurance charges
def prediction(prediction_df):
    pred_out = loaded_pipe.predict(prediction_df)
    final_result = pred_out[0]

    return final_result

# Function to get user input and display the prediction results
def main():
    put_markdown(
        '''
        # Application for predicting if a clients response is a Spam or Genuine message
        '''
        , lstrip=True
    )

    model_inputs = input_group(
        "Enter client's message below:",
        [
            textarea("Enter your text here", name='message'),
        ]
    )

    # save user input in a dataframe and prepare it for prediction
    prediction_df = pd.DataFrame(data = [[model_inputs[i] for i in ['message']]], 
                           columns = ['message'])
    
    # use text_process function to remove punctuations and stopwords
    prediction_df['cleaned_message'] = prediction_df.message.apply(text_process)

    user_input_message = prediction_df.cleaned_message
    message_input = prediction(user_input_message)
    put_markdown("### This response has been marked as: {} ".format(message_input))

#ERASE PYWEBIO IF MAIN AND REPLACE WITH FLASK
# function to Start the PyWebIO web application
# if __name__ == "__main__":
#     try:
#         main()
#     except SessionClosedException:
#         print("The session was closed unexpectedly")

app.add_url_rule('/tool', 'webio_view', main) #MIGHT NEED TO CHANGE THIS TO webio_view(main) if it doesn't work
methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS']

# Start Flask app
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port to use')
    args = parser.parse_args()

    # Start the PyWebIO application
    pywebio.start_server(main, port=args.port)
