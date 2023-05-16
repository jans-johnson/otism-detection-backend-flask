import pickle
import pandas as pd
from flask import Flask, request, jsonify

logistic = pickle.load(open('logistic.sav', 'rb'))
adaboost = pickle.load(open('adaboost.sav', 'rb'))
lgbm = pickle.load(open('lgbm.pkl', 'rb'))
mlp = pickle.load(open('mlp.sav', 'rb'))
grad= pickle.load(open('grad.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the data from the POST request
    data = request.json

    # Perform any necessary data processing or transformations
    # ...

    # Make predictions using your model
    predictions = make_predictions(data)

    # Prepare the response
    response = {
        'predictions': predictions
    }

    # Return the response as JSON
    return jsonify(response)

def make_predictions(data):
    columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test']
    data=[[1,1,0,0,0,1,1,0,0,0,36,1,'White European',1,0,'family member']]
    # Create the DataFrame
    dataset = pd.DataFrame(data, columns=columns)

    if(dataset['Who_completed_the_test'].all()=='Health Care Professional'):
        dataset['Health Care Professional']=1
    else:
        dataset['Health Care Professional']=0


    if(dataset['Who_completed_the_test'].all()=='Others'):
        dataset['Others']=1
    else:
        dataset['Others']=0

    if(dataset['Who_completed_the_test'].all()=='Self'):
        dataset['Self']=1
    else:
        dataset['Self']=0

    if(dataset['Who_completed_the_test'].all()=='family member'):
        dataset['family member']=1
    else:
        dataset['family member']=0

    dataset.drop('Who_completed_the_test', axis=1, inplace=True)

    if(dataset['Ethnicity'].all()=='Hispanic'):
        dataset['Hispanic']=1
    else:
        dataset['Hispanic']=0


    if(dataset['Ethnicity'].all()=='Latino'):
        dataset['Latino']=1
    else:
        dataset['Latino']=0


    if(dataset['Ethnicity'].all()=='Native Indian'):
        dataset['Native Indian']=1
    else:
        dataset['Native Indian']=0

    if(dataset['Ethnicity'].all()=='Pacifica'):
        dataset['Pacifica']=1
    else:
        dataset['Pacifica']=0


    if(dataset['Ethnicity'].all()=='White European'):
        dataset['White European']=1
    else:
        dataset['White European']=0


    if(dataset['Ethnicity'].all()=='asian'):
        dataset['asian']=1
    else:
        dataset['asian']=0


    if(dataset['Ethnicity'].all()=='black'):
        dataset['black']=1
    else:
        dataset['black']=0


    if(dataset['Ethnicity'].all()=='middle eastern'):
        dataset['middle eastern']=1
    else:
        dataset['middle eastern']=0


    if(dataset['Ethnicity'].all()=='mixed'):
        dataset['mixed']=1
    else:
        dataset['mixed']=0


    if(dataset['Ethnicity'].all()=='others'):
        dataset['others']=1
    else:
        dataset['others']=0


    if(dataset['Ethnicity'].all()=='south asian'):
        dataset['south asian']=1
    else:
        dataset['south asian']=0


    dataset.drop('Ethnicity', axis=1, inplace=True)
    dataset['months12_24'] = dataset['Age_Mons'].apply(lambda x: 0 if x > 24 else 1)
    dataset['months24_36'] = dataset['Age_Mons'].apply(lambda x: 0 if x < 24 else 1)
    dataset.drop(['Age_Mons'], axis=1, inplace=True)
    return [logistic.predict(dataset),adaboost.predict(dataset),lgbm.predict(dataset),mlp.predict(dataset),grad.predict(dataset)]

if __name__ == '__main__':
    app.run(debug=True)


