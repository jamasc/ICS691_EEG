This is for all our code to have a working eeg explanation model

I did not put any model checkpoints into this repo and also no data. one of the first cells in workflow.ipynb has the code to download one dataset.
I will keep training the classification model.

LJ: data should be preprocessed to .set format

Zelda: there is a function extract_biomarkers or something like that in utility.py, i just messed around with some random biomarkers there but feel free to incorporate it. Let me know if you are using another format for these biomarker extractions so i can make them work with the rest.

Skylar: The final classification model with give an AD diagnosis, together with biomarkers and biographics something like chatGPT should handle the explanation.
