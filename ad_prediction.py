from sklearn.linear_model import LogisticRegression
import joblib

def predict(biographics, biomarkers, features):
    bio = [1 if biographics['gender'] == 'M' else 0, biographics['age'], biographics['mmse']]

    x_parts = [bio, biomarkers, features]
    x = [[item for sublist in x_parts for item in sublist]]
    
    lr = joblib.load("logreg_model.pkl")
    prediction = int(lr.predict(x))
    confidences = lr.predict_proba(x)
    confidence = confidences[0][prediction]
    
    return prediction, confidence