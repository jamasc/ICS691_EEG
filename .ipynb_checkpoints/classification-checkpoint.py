### classification

from joblib import load
import numpy as np

emb_rf = load("eeg_classifier_rf_embeddings_withAge.joblib")
bio_rf = load("eeg_classifier_rf_biomarkers_withAge.joblib")
emb_rf_no_age = load("eeg_classifier_rf_embeddings.joblib")
bio_rf_no_age = load("eeg_classifier_rf_biomarkers.joblib")

def get_diagnosis(x, age, mode=''):
    if age:
        x = np.concatenate([x, [age]])
        if mode == 'embeddings':
            rf = emb_rf
        elif mode == 'biomarkers':
            rf = bio_rf
        else:
            raise ValueError("mode must be set to 'embeddings' or 'biomarkers'!")
    else:
        if mode == 'embeddings':
            rf = emb_rf_no_age
        elif mode == 'biomarkers':
            rf = bio_rf_no_age
        else:
            raise ValueError("mode must be set to 'embeddings' or 'biomarkers'!")
    
    probs = rf.predict_proba([x])[0]
    
    pred_class = probs.argmax()
    confidence = probs.max()

    return pred_class, confidence

def get_batch_prediction(xs):
    probs = {}
    for x in xs:
        pred, conf = get_diagnosis(x, mode='embeddings')
        if pred not in probs:
            probs[pred] = []
        probs[pred].append(conf)
    return probs

def pred2word(pred):
    if pred == 0:
        return 'UNKNOWN'
    elif pred == 1:
        return 'HEALTHY'
    elif pred == 2:
        return 'ALZHEIMERS DISEASE'
    elif pred == 3:
        return 'UNKNOWN'
    else:
        raise ValueError('label unclear: {pred}')

def diag2llm(bio_pred, bio_conf, emb_pred, emb_conf):
    bio_pred_word = pred2word(bio_pred)
    emb_pred_word = pred2word(emb_pred)
    llm_text = f"""
        Two classification models were trained to classify an eeg to one of the following 4 categories: HEALTHY, ALZHEIMER DISEASE, OTHER DEMENTIA, UNKNOWN.
        One of the models was trained on calculated biomarkers, and the other one trained on the semantic embeddings of the raw eeg.
        The prediction from the biomarker model is {bio_pred_word} with a confidence of {bio_conf}.
        The prediction from the embedding model is {emb_pred_word} with a confidence of {emb_conf}.
        These predictions and confidences should be used as a guideline rather than the full truth.
    """
    return llm_text