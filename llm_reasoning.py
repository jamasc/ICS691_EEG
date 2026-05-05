from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pathlib import Path
import os
import io
import httpx
import pathlib

def paper_to_bytes(filepath: str):
    tempPath = pathlib.Path('EEG Papers/' + filepath)

    return types.Part.from_bytes(
        data= tempPath.read_bytes(), 
        mime_type= 'application/pdf',
    )

def main():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    biomarkerFile = "new_biomarkers.txt"
    outputFile = "final_output.txt"
    predictionFile = "prediction.txt"

    #Grabs the text from the biomarker output file
    biomarkers = Path(biomarkerFile).read_text()

    #Grabs text from prediction file
    prediction = Path(predictionFile).read_text()


    #A Deep Learning Approach to Alzheimer’s Diagnosis Using EEG Data: Dual-Attention and Optuna-Optimized SVM
    bytes_paper1 = paper_to_bytes('eegpaper1.pdf')

    #EEG biomarkers for Alzheimer’s disease: A novel automated pipeline for detecting and monitoring disease progression
    bytes_paper2 = paper_to_bytes('eegpaper2.pdf')

    #EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals
    bytes_paper3 = paper_to_bytes('eegpaper3.pdf')

    #Usefulness of EEG Techniques in Distinguishing Frontotemporal Dementia from Alzheimer’s Disease and Other Dementias
    bytes_paper4 = paper_to_bytes('eegpaper4.pdf')

    #EEG-based classification of alzheimer’s disease and frontotemporal dementia using functional connectivity
    bytes_paper5 = paper_to_bytes('eegpaper5.pdf')



    #Explaintion of the task
    taskExplanation = f"""
            Review the provided PDF file(s) before submitting your answer. 
            You may use them to inform your response, \
            but are NOT required to reference them directly in your final answer.
            
            Task:
            You are an expert neuroscience clinician examining a patient for potential health risks.

            You will be provided with labeled biomarker data from the patient, \
            as well as a predicted diagnosis from our other EEG-based prediction model. 


            Please format your analysis structure in the following order:
            a short, one phrase Hypothesized Diagnosis choosing one of three diagnoses \
            [Alzheimer's Disease, Frontaltemporal Disorder, Healthy],
            a thorough, listed review of all evidence supporting that claim,
            and finally a note on the potential contradictory evidence or noticeable gaps in information. 

            
            Formatting should look like:

            Hypothesis:
            Evidence:
            Contradictory Evidence / Gaps:

            Here is the biomarker information:
            {biomarkers}

            And here is the predicted diagnosis from another EEG-based prediction model:
            {prediction}
            
            Try to be brief (one to two sentences per point) wherever possible. \
            Include a confidence percentage at the end of your response related to how plausible your diagnosis appears. 
        """


    # taskExplanation = '''
    #         Review the provided PDF file(s) before submitting your answer. 
    # #         You may use them to inform your response, but are NOT required to reference them directly in your final answer.
            
    # #         Task:
    # #         You are an expert neuroscience clinician examining a patient for potential health risks. Consider the plausibility of each of three potential diagnoses: Alzheimer's Disease,Frontaltemporal Disorder, and Healthy Condition. 

    #     Argue the case for each potential diagnosis, providing a brief list of evidence alongside it.

    #     Then, choose the most plausible diagnosis as your final answer. Write that answer as a single phrase using the format of:

    #     Final Answer: [answer]
    #     '''


    

    #Prompting the robot to generate a final response...
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[taskExplanation, 
                  bytes_paper1, 
                  bytes_paper2, 
                  bytes_paper3, 
                  bytes_paper4, 
                  bytes_paper5 
                ]
    )


    #Clears the output txt file and writes in the newest response
    with open(outputFile, "r+") as file:
        file.seek(0)
        file.truncate()

        file.write(response.text)

    print(response.text)


if __name__ == '__main__':
    main()