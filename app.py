import streamlit as st 
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import  NumHDonors, NumHAcceptors
from rdkit.Chem.Crippen import MolLogP
import mols2grid
from rdkit.Chem import *
import rdkit
from rdkit.Chem import AllChem, Draw
import py3Dmol
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO
import tensorflow as tf    
from rdkit.Chem import AllChem
from rdkit import DataStructs


st.set_page_config(page_title="FDA-Approved Drugs Analysis", layout="centered")
st.title("Lipinski's Rule of 5")
st.markdown("The Lipinski rule of five is a guideline used to predict whether a biologically active molecule is likely to have the chemical and physical properties necessary for oral bioavailability. ")


url = 'https://www.cureffi.org/wp-content/uploads/2013/10/drugs.txt?raw=true'
response = requests.get(url, verify=False)

@st.cache_data()
def read():
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data, sep="\t").dropna()
    return df
def mw(smiles):
    m=Chem.MolFromSmiles(smiles)
    return ( ExactMolWt(m))
def logp(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return MolLogP(mol)
def NumHDonor(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return NumHDonors(mol)
def NumHAccep(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return NumHAcceptors(mol)

def generate_3d_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # print(mol) rdkit.Chem.rdchem.Mol object
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Morgan fingerprint with radius 2
    return fp

def calculate_tanimoto_coefficient(smiles1, smiles2):
    fp1 = generate_fingerprint(smiles1)
    fp2 = generate_fingerprint(smiles2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mol_to_pdb_block(mol):
    return Chem.MolToPDBBlock(mol)

df=read().copy()
df.dropna()
df["SMILES"] = df["smiles"]
df["MW"]=df.apply(lambda x: mw(x["smiles"]), axis=1)
df["logP"]=df.apply(lambda x: logp(x["smiles"]), axis=1)
df["NumHDonors"]=df.apply(lambda x: NumHDonor(x["smiles"]), axis=1)
df["NumHAcceptors"]=df.apply(lambda x: NumHAccep(x["smiles"]), axis=1)
df=df.drop(["smiles"], axis=1)
    
st.sidebar.header("Set your desired parameters!")
st.sidebar.write("""This rule assesses pharmacokinetic properties such as absorption, distribution, metabolism, and excretion based on specific physicochemical criteria, which include:
""")
mweight_cutoff= st.sidebar.slider(label="Molecular Weight", min_value=0, max_value=1000, value=500, step=10,)
logp_cutoff=st.sidebar.slider(label="LogP of molecule", min_value=-10, max_value=10, value=5, step=1,)
NumHDonors_cutoff=st.sidebar.slider(label="No. of H Donors", min_value=0, max_value=15, value=5, step=1,)
NumHAccep_cutoff=st.sidebar.slider(label="No. of H Acceptors", min_value=0, max_value=20, value=10, step=1,)

df1= df[df["MW"]<mweight_cutoff]
df2=df1[df1["logP"]<logp_cutoff]
df3=df2[df2["NumHDonors"]<NumHDonors_cutoff]
df4=df3[df3["NumHAcceptors"]<NumHAccep_cutoff]

print(df4.columns)
st.write(df4)
st.markdown(
    """
    <style>
    .mols2grid {
        margin: auto;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

rhtml= mols2grid.display(df4, subset=["img","generic_name", "MW"], mapping={"generic_name": "Name"}, )._repr_html_()

components.html(rhtml, width=900, height=750, scrolling=False)

selected_name = st.selectbox("Select a molecule to view its 3D structure:", df4["generic_name"].unique())

if selected_name:
    selected_smiles = df4[df4["generic_name"] == selected_name]["SMILES"].values[0]
    mol = generate_3d_coordinates(selected_smiles)
    pdb_block = mol_to_pdb_block(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(pdb_block, "pdb")
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    viewer_html = viewer._make_html()
    components.html(viewer_html, height=450)
    
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    # encoded = tokenizer(selected_smiles, return_tensors="pt")
    # output = model(**encoded)
    pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer) ##easier way to find embeddings of smile strings
    # data = pipeline(selected_smiles)
    # print(selected_smiles)
    
    col1, col2= st.columns((1,1))
    with col1:
        st.write("Extracting Embeddings...")
        all_smiles = df4["SMILES"].tolist()
        embeddings = [pipeline(smiles)[0] for smiles in all_smiles]
        # print(embeddings)
        embeddings = np.array([np.mean(embed, axis=0) for embed in embeddings])  # Using mean pooling
        
        # Extract embedding for the selected SMILES
        # print(pipeline(selected_smiles))

        
        selected_embedding = pipeline(selected_smiles)[0]
        # print("***********************************************************************")
        # print(selected_embedding)
        selected_embedding = np.mean(selected_embedding, axis=0).reshape(1, -1)
        print(selected_embedding.shape)
        print(embeddings.shape)
        # Compute cosine similarity
        similarities = cosine_similarity(selected_embedding, embeddings).flatten()
        print(similarities.shape)
        top_indices = similarities.argsort()[-6:][::-1]  
        top_indices = top_indices[1:]  
        st.write(selected_embedding)
        st.write("Top 5 most similar molecules:")
        for idx in top_indices:
            st.write(f"Generic Name: {df4.iloc[idx]['generic_name']}, Similarity: {similarities[idx]}")
            s=df4.iloc[idx]["SMILES"]
            mol = generate_3d_coordinates(s)
            pdb_block = mol_to_pdb_block(mol)
            viewer = py3Dmol.view(width=200, height=200)
            viewer.addModel(pdb_block, "pdb")
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer_html = viewer._make_html()
            components.html(viewer_html, height=200)
            
    with col2:  
        st.write("Calculating Tanimoto coefficients...")
        tanimoto_coeffs = []
        for idx, smiles in enumerate(df4["SMILES"].tolist()):
            if smiles != selected_smiles:
                tanimoto_coeff = calculate_tanimoto_coefficient(selected_smiles, smiles)
                tanimoto_coeffs.append((idx, tanimoto_coeff))
        
        # Sort by Tanimoto coefficient
        tanimoto_coeffs.sort(key=lambda x: x[1], reverse=True)
        top_tanimoto_indices = [idx for idx, _ in tanimoto_coeffs[:5]]

        st.write("Top 5 most similar molecules based on Tanimoto Coefficient:")
        for idx in top_tanimoto_indices:
            st.write(f"Generic Name: {df4.iloc[idx]['generic_name']}, Tanimoto Coefficient: {tanimoto_coeffs[idx][1]}")
            s=df4.iloc[idx]["SMILES"]
            mol = generate_3d_coordinates(s)
            pdb_block = mol_to_pdb_block(mol)
            viewer = py3Dmol.view(width=200, height=200)
            viewer.addModel(pdb_block, "pdb")
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer_html = viewer._make_html()
            components.html(viewer_html, height=200)
 
# abacavir_smiles = df4[df4["generic_name"] == "Abacavir"]["SMILES"].iloc[0]
# ruxolitinib_smiles = df4[df4["generic_name"] == "Ruxolitinib"]["SMILES"].iloc[0]

# # Function to generate fingerprints
# def generate_fingerprint(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Morgan fingerprint with radius 2
#     return fp

# # Generate fingerprints
# fp_abacavir = generate_fingerprint(abacavir_smiles)
# fp_ruxolitinib = generate_fingerprint(ruxolitinib_smiles)

# # Calculate Tanimoto coefficient
# tanimoto_coefficient = DataStructs.TanimotoSimilarity(fp_abacavir, fp_ruxolitinib)

# print(f"Tanimoto coefficient between Abacavir and Ruxolitinib: {tanimoto_coefficient:.4f}")
    
    
    
    

    



# tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
# model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
# s= tokenizer(df4["SMILES"][0], return_tensors='pt')
# with torch.no_grad():
#     outputs= model(**s)
# print(outputs)
# print(s)
