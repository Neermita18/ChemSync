# Analyser and Viewer of FDA verified Drugs
## Table of contents
* [General info](#general-info)
* [Project Glimpse](#glimpse)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is a simple and dynamic application to view chemical compounds based on selected thresholds. Lipinski's rule of 5 or Pfizer's rule helps drugs be classified as 'orally active.' It helps in distinguishing between drug like and non drug like molecules. It predicts high probability of success or failure due to drug likeness for molecules complying with 2 or more of the following rules
- [ ] Molecular mass less than 500 Dalton
- [ ] High lipophilicity (expressed as LogP less than 5)
- [ ] Less than 5 hydrogen bond donors
- [ ] Less than 10 hydrogen bond acceptors

## Project Glimpse

[streamlit-app-2024-07-21-18-07-40.webm](https://github.com/user-attachments/assets/9eac3534-80e7-40db-b4e6-5ae09643513b)

	
## Technologies
Project is created with:
* numpy version: 1.26.4
* streamlit version: 1.36.0
* rdkit version: 2024.3.1
* All requirements are available in requirements.txt
	
## Setup
To run this project, download the executable file from Releases or run the application on the web: https://neermita-drugs-analysis.streamlit.app/	



