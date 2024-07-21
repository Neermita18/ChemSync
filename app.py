import streamlit as st 
import pandas as pd
import numpy as np
import torch
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D    
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import  NumHDonors, NumHAcceptors
from rdkit.Chem.Crippen import MolLogP
import mols2grid
from rdkit.Chem import *
import rdkit
from rdkit.Chem import AllChem, Draw
import py3Dmol
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO  
from rdkit.Chem import AllChem
from rdkit import DataStructs


st.set_page_config(page_title="FDA-Approved Drugs Analysis", layout="centered")
st.title("Exploring FDA-Approved Drugs: An Analytical Overview")
st.header("Lipinski's Rule of 5")
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
print(df.shape)
df["SMILES"] = df["smiles"]
df["MW"]=df.apply(lambda x: mw(x["smiles"]), axis=1)
df["logP"]=df.apply(lambda x: logp(x["smiles"]), axis=1)
df["NumHDonors"]=df.apply(lambda x: NumHDonor(x["smiles"]), axis=1)
df["NumHAcceptors"]=df.apply(lambda x: NumHAccep(x["smiles"]), axis=1)
df=df.drop(["smiles"], axis=1)
    
st.sidebar.header("Set your desired parameters")
st.sidebar.write("""Lipinski's rule assesses pharmacokinetic properties such as absorption, distribution, metabolism, and excretion based on specific physicochemical criteria, which include:
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
st.header("Select a molecule to view its 3D structure")
selected_name = st.selectbox("", df4["generic_name"].unique())

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
    
    st.markdown("""
    <style>
        .container {
            padding: 10px;
            background-color: #f0f0f0;
            margin: 10px;
            height=200px;
            width=100px;
        }
        .column {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
        """, unsafe_allow_html=True)
row1 = st.columns(2)  
with row1[0]:
    tile = st.container(height=140)
    tile.title(":balloon:")
    cosine_button=tile.button("Show Molecules with Cosine Similarity", key="cosine")
with row1[1]:
    tile = st.container(height=140)
    tile.title(":balloon:")
    tanimoto_button=tile.button("Show Molecules with Tanimoto Similarity", key="tanimoto")
    
if 'selected_smiles' not in st.session_state:
    st.session_state['selected_smiles'] = None

if 'show_cosine' not in st.session_state:
    st.session_state['show_cosine'] = False
    st.session_state['cosine_similarities'] = None
    st.session_state['top_cosine_indices'] = None

if 'show_tanimoto' not in st.session_state:
    st.session_state['show_tanimoto'] = False
    st.session_state['tanimoto_coeffs'] = None
    st.session_state['top_tanimoto_indices'] = None

# Update session state when the selected molecule changes
if selected_smiles != st.session_state['selected_smiles']:
    st.session_state['selected_smiles'] = selected_smiles
    st.session_state['show_cosine'] = False
    st.session_state['show_tanimoto'] = False
    st.session_state['cosine_similarities'] = None
    st.session_state['top_cosine_indices'] = None
    st.session_state['tanimoto_coeffs'] = None
    st.session_state['top_tanimoto_indices'] = None

if cosine_button:
    st.session_state['show_cosine'] = True
    all_smiles = df4["SMILES"].tolist()
    embeddings = [pipeline(smiles)[0] for smiles in all_smiles]
    embeddings = np.array([np.mean(embed, axis=0) for embed in embeddings])  # Using mean pooling
    selected_embedding = pipeline(selected_smiles)[0]
    selected_embedding = np.mean(selected_embedding, axis=0).reshape(1, -1)
    similarities = cosine_similarity(selected_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-6:][::-1]
    top_indices = top_indices[1:]
    st.session_state['cosine_similarities'] = similarities
    st.session_state['top_cosine_indices'] = top_indices

if tanimoto_button:
    st.session_state['show_tanimoto'] = True
    tanimoto_coeffs = []
    for idx, smiles in enumerate(df4["SMILES"].tolist()):
        if smiles != selected_smiles:
            tanimoto_coeff = calculate_tanimoto_coefficient(selected_smiles, smiles)
            tanimoto_coeffs.append((idx, tanimoto_coeff))
    tanimoto_coeffs.sort(key=lambda x: x[1], reverse=True)
    top_tanimoto_indices = [idx for idx, _ in tanimoto_coeffs[:5]]
    st.session_state['tanimoto_coeffs'] = tanimoto_coeffs
    st.session_state['top_tanimoto_indices'] = top_tanimoto_indices

row1 = st.columns(2)

if st.session_state.get('show_cosine', False):
    with row1[0].container(border=True):
        st.markdown("### Top 5 most similar molecules based on Cosine Similarity")
        similarities = st.session_state['cosine_similarities']
        top_indices = st.session_state['top_cosine_indices']
        for idx in top_indices:
            st.write(f"Generic Name: {df4.iloc[idx]['generic_name']}, Similarity: {similarities[idx]}")
            s = df4.iloc[idx]["SMILES"]
            mol = generate_3d_coordinates(s)
            pdb_block = mol_to_pdb_block(mol)
            viewer = py3Dmol.view(width=200, height=200)
            viewer.addModel(pdb_block, "pdb")
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer_html = viewer._make_html()
            st.components.v1.html(viewer_html, height=200)

if st.session_state.get('show_tanimoto', False):
    with row1[1].container(border=True):
        st.markdown("### Top 5 most similar molecules based on Tanimoto Similarity")
        tanimoto_coeffs = st.session_state['tanimoto_coeffs']
        top_tanimoto_indices = st.session_state['top_tanimoto_indices']
        for idx in top_tanimoto_indices:
            st.write(f"Generic Name: {df4.iloc[idx]['generic_name']}, Tanimoto Coefficient: {tanimoto_coeffs[idx][1]}")
            s = df4.iloc[idx]["SMILES"]
            mol = generate_3d_coordinates(s)
            pdb_block = mol_to_pdb_block(mol)
            viewer = py3Dmol.view(width=200, height=200)
            viewer.addModel(pdb_block, "pdb")
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer_html = viewer._make_html()
            st.components.v1.html(viewer_html, height=200)   


