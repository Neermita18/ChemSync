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
st.set_page_config(page_title="Molecular Grid", layout="centered")
st.title("Lipinski's Rule of 5")
st.markdown("This rule helps to predict if a biologically active molecule is likely to have the chemical and physical properties to be orally bioavailable. The Lipinski rule bases pharmacokinetic drug properties such as absorption, distribution, metabolism and excretion on specific physicochemical properties.")


import requests
from io import StringIO

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
st.sidebar.write("*some")
mweight_cutoff= st.sidebar.slider(label="Molecular Weight", min_value=0, max_value=1000, value=500, step=10,)
logp_cutoff=st.sidebar.slider(label="LogP of molecule", min_value=-10, max_value=10, value=5, step=1,)
NumHDonors_cutoff=st.sidebar.slider(label="No. of H Donors", min_value=0, max_value=15, value=5, step=1,)
NumHAccep_cutoff=st.sidebar.slider(label="No. of H Acceptors", min_value=0, max_value=20, value=10, step=1,)

df1= df[df["MW"]<mweight_cutoff]
df2=df1[df1["logP"]<logp_cutoff]
df3=df2[df2["NumHDonors"]<NumHDonors_cutoff]
df4=df3[df3["NumHAcceptors"]<NumHAccep_cutoff]

print(df4.columns)
st.write(df4, width=900)
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
    
    # Embed the HTML content
    components.html(viewer_html, height=450)
   

# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
# model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-10M-MLM")