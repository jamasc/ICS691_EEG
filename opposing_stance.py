from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pathlib import Path
import os
import pathlib

def paper_to_bytes(filepath: str):
    tempPath = pathlib.Path('EEG Papers/' + filepath)

    return types.Part.from_bytes(
        data= tempPath.read_bytes(), 
        mime_type= 'application/pdf',
    )


#Explaintion of the task
def getTask(diagnosis, tmpBiomarkers, tmpPrediction):
    return f'''
        Review the provided PDF file(s) before submitting your answer. 
        You may use them to inform your response, \
        but are NOT required to reference them directly in your final answer.
        
        Role:
        You are an expert neuroscience clinician.

        Task:
        You will be provided with labeled biomarker data from the patient, \
        as well as a predicted diagnosis from our other EEG-based prediction model. 

        Provide a strong argument for why the predicted outcome is actually {diagnosis} with a brief explaination (1 or 2 paragraphs), \
        then end your analysis with a percieved confidence rating (from 0 to 100) for your arguement.


        Predition: 
        {tmpPrediction}

        Biomarkers:
        {tmpBiomarkers}
        
        '''

#=====================================================================================

def getPapers():
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

    #Electroencephalogram Based Biomarkers for Detection of Alzheimer's Disease
    bytes_paper6 = paper_to_bytes('eegpaper6.pdf')

    #The EEG analysis and identification of Alzheimer's disease: a review
    bytes_paper7 = paper_to_bytes('eegpaper7.pdf')

    return [bytes_paper1, 
            bytes_paper2, 
            bytes_paper3, 
            bytes_paper4, 
            bytes_paper5,
            bytes_paper6,
            bytes_paper7
        ]


#Prompting the robot to generate a final response...
def getResponse(tmpClient, prompt):
    tmpResponse = tmpClient.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents= prompt
    )

    #DELETE PRINT LATER================================
    print(tmpResponse.text + '\n')

    return tmpResponse.text
    

def main():
    biomarkerFile = "new_biomarkers.txt"
    predictionFile = "prediction.txt"
    outputFile = "final_output.txt"

    #Grabs the text from the biomarker output file
    biomarkers = Path(biomarkerFile).read_text()

    #Grabs text from prediction file
    prediction = Path(predictionFile).read_text()


    #Compiles all EEG papers into a list
    eeg_paper_list = getPapers()


    #Task prompts to argue for each diagnosis' plausibility
    FTD_arg = getTask("Other Dementia/Fronteltemporal", biomarkers, prediction)
    UNK_arg = getTask("Unknown", biomarkers, prediction)
    HC_arg = getTask("Healthy", biomarkers, prediction)
    AZ_arg = getTask("Alzheimer's Disease", biomarkers, prediction)
    

    #Creates a client with the provided Gemini API Key
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    #Calls API to gather response
    print("**OTHER/FTD ARGUEMENT: \n")
    FTD_resp = getResponse(client, [FTD_arg, eeg_paper_list])

    print("**UNKNOWN ARGUEMENT: \n")
    UNK_resp = getResponse(client, [UNK_arg, eeg_paper_list])

    print("**HEALTHY CONDITION ARGUEMENT: \n")
    HC_resp = getResponse(client, [HC_arg, eeg_paper_list])

    print("**ALZHEIMERS ARGUEMENT: \n")
    AZ_resp = getResponse(client, [AZ_arg, eeg_paper_list])


    finalTaskPrompt = f"""
            Review the provided PDF file(s) before submitting your answer. 
            You may use them to inform your response, \
            but are NOT required to reference them directly in your final answer.
            
            Task:
            You are an expert neuroscience clinician examining a patient for potential health risks.
            Your task is to form a concrete diagnosis and analysis using the provided information.

            You will be provided with labeled biomarker data from the patient, \
            as well as a predicted diagnosis from our other EEG-based prediction model. 

            You will also be provided with arguments for why each of the four possible diagnoses \
            (Healthy Condition (HC), Alzheimer's Disease (AD), Other Dementia/FTD, Unknown/Unlabeled) \
            are potentially the most accurate diagnosis for this patient. 

            Consider each of the provided arguments equally when forming your final analysis. 
            Keep in mind that while our prediction model is robust, it is not always accurate.  

            --

            Please format your analysis structure in the following order:
            a short, one phrase Hypothesized Diagnosis choosing one of four possible options \
            [Healthy Condition (HC), Alzheimer's Disease (AD), Other Dementia/FTD, Unknown/Unlabeled],
            a thorough, listed review of all evidence supporting that claim,s
            and finally a note on the potential contradictory evidence or noticeable gaps in information. 

            
            Formatting should look like:

            Hypothesis:
            Evidence:
            Contradictory Evidence / Gaps:

            Try to be brief (one sentence per point) wherever possible. \
            Include a confidence percentage at the end of your response related to how plausible your diagnosis appears. 

            --

            Here is the biomarker information:
            {biomarkers}

            And here is the predicted diagnosis from another EEG-based prediction model:
            {prediction}


            And finally here are the provided arguements for each diagnosis option.

            Alzheimer's Disease:
            {AZ_resp}

            Other Dementia / FTD:
            {FTD_resp}

            Healthy Condition:
            {HC_resp}

            Unknown/Unlabeled:
            {UNK_resp}
 
        """
    
    #Call API for final analysis

    print("**FINAL RESPONSE: \n")
    response = getResponse(client, [finalTaskPrompt, eeg_paper_list])


    #Clears the output txt file and writes in the newest response

    # with open(outputFile, "r+") as file:
    #     file.seek(0)
    #     file.truncate()

    #     file.write(response.text)

    # print(response.text)


if __name__ == '__main__':
    main()