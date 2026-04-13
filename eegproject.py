from google import genai
from pathlib import Path
import os


def main():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    biomarkerFile = "biomarkers.txt"

    outputFile = "output.txt"


    prompt = Path(biomarkerFile).read_text()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are an expert neuroscience researcher examining a patient for potential Alzheimer’s Disease. " \
            "All of the following data has been taken from the same patient. " \
            "Please format your analysis' structure in the following order: " \
            "a short, one sentence Hypothesized Diagnosis, " \
            "a thorough, listed review of all evidence supporting that claim, " \
            "and finally a note on the potential contradictory evidence or noticeable gaps in information." \
            + prompt
    )

    with open(outputFile, "r+") as file:
        file.seek(0)
        file.truncate()

        file.write(response.text)

    print(response.text)


if __name__ == '__main__':
    main()