# ChemSync: An analyser for FDA-verified drugs
## :star: Table of contents
* [General info](#general-info)
* [Functionalities](#functionalities)
* [Project Glimpse](#projectglimpse)
* [Technologies](#technologies)
* [Setup](#setup)

## :bulb: General info
This project is a simple and dynamic application to view chemical compounds based on selected thresholds. Lipinski's rule of 5 or Pfizer's rule helps drugs be classified as 'orally active.' It helps in distinguishing between drug like and non drug like molecules. It predicts high probability of success or failure due to drug likeness for molecules complying with 2 or more of the following rules:
- [x] Molecular mass less than 500 Dalton
- [x] High lipophilicity (expressed as LogP less than 5)
- [x] Less than 5 hydrogen bond donors
- [x] Less than 10 hydrogen bond acceptors

## :rocket: Functionalities
- [x] View similar compounds in 2D, their SMILE representations and their molecular weights based on the selected thresholds using the slider.
- [x] Select a single compound and view its structure in 3D.
- [x] View the top 5 most similar compounds based on cosine similarity of embedded SMILE strings in 3D.
- [x] View the top 5 most similar compounds based on Tanimoto similarity from Morgan Fingerprints in 3D.

## :mag: Project Glimpse

[streamlit-app-2024-07-21-18-07-40.webm](https://github.com/user-attachments/assets/9eac3534-80e7-40db-b4e6-5ae09643513b)

	
## :page_with_curl: Technologies
Project is created with:
* numpy version: 1.26.4
* streamlit version: 1.36.0
* rdkit version: 2024.3.1
* All requirements are available in requirements.txt
	
## :gear: Setup
To run this project, download the executable file from Releases or run the application on the web: https://neermita-chemsync.streamlit.app/	



