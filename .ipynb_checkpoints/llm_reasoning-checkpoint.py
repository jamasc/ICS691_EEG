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
    client = genai.Client(api_key='')

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
            Role:
            You are an expert neuroscience clinician examining a patient for Alzheimer's Disease, explaining your diagnosis and reasoning to another clinician. You will be provided with a report of the patient's biomarkers and a report from a classification model to inform your decision

            How to read the biomarker report
            ================================
            The report contains EEG features computed in five brain regions: frontal, 
            central, temporal, parietal, and occipital. Four kinds of evidence may 
            appear:
             
            1. PERSISTENT FLAGS (⚠ markers): features marked as abnormal in both the 
               recording-level average AND in at least 25% of individual segments. 
               These appear in the "PERSISTENTLY FLAGGED FEATURES" section. This is 
               the strongest tier of flag-based EEG evidence — features that are 
               consistently and substantially abnormal across the recording.
             
            2. SUB-THRESHOLD FINDINGS (~ markers or unmarked entries): features 
               flagged in some segments but not meeting the strict persistence + 
               recording-average criterion. These appear in the "SUB-THRESHOLD 
               FINDINGS" section. ~ markers indicate ≥15% of segments flagged; 
               unmarked entries indicate fewer. Sub-threshold findings are weaker 
               evidence than persistent flags, but they are NOT noise — they represent 
               real per-segment deviations that may indicate early-stage or 
               intermittent abnormalities. Treat them as supportive evidence when 
               they cluster into clinical patterns (e.g. multiple posterior alpha 
               features sub-threshold-flagged together).
             
             
            4. WITHIN-RANGE FEATURES: everything else. These fall within healthy 
               reference ranges for the recording condition and have low KDE AD 
               probabilities.
             
            The "AD INDICATOR SUMMARY" synthesizes flag-based findings into 
            high-level patterns (posterior alpha reduction, theta elevation, 
            complexity loss, spectral slowing). Sub-threshold patterns are explicitly 
            labeled when they appear here.
             
            A report with no persistent flags, no sub-threshold findings, and uniformly 
            low KDE AD probabilities indicates the EEG evidence is not abnormal in 
            the ways this pipeline detects. This does not rule out AD — clinical 
            diagnosis requires more than EEG. Don't mention this again, do not use the absence of anything as a reason for anything!


            Here is the biomarkerminformation for the patient, use it to make an informed diagnosis:
            {biomarkers}

            Here is the predicted diagnosis from two EEG-based prediction models, only to be used as rough guideline:
            {prediction}

            You will also beprovided with PDF files of clinical research papers on EEG-related Dementia studies.
            You should review these before writing your answer, but you DO NOT need to reference them in your final answer.

            Please structure your analysis in the following order:
            First, form a thorough, listed review of all evidence supporting an Alzheimer diagnosis. Start with the strongest evidence.
            Second, a thorough, listed review of all evidence contradicting an Alzheimer diagnosis. Start with the strongest evidence.
            
            Finally, state a short, definitive answer for your predicted diagnosis, choosing \
            either Alzheimer's Disease or Not Alzheimer's Disease. 
            Format this last section as just the final answer. Do not add any additional reasoning.


            Also, state how certain you feel this diagnosis is for the patient. 
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
    with open(outputFile, "r+", encoding='utf-8') as file:
        file.seek(0)
        file.truncate()

        file.write(response.text)

    print(response.text)


if __name__ == '__main__':
    main()