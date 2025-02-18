import datamol as dm
from rdkit import Chem
import csv

def standardize_smiles(smiles: str) -> str:
    """
    Standardizes a SMILES string using datamol and RDKit.

    Steps:
        1. Convert the SMILES to an RDKit molecule.
        2. Standardize the molecule (e.g., remove salts, adjust charges).
        3. Convert the standardized molecule back to a canonical SMILES.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The standardized SMILES string, or None if standardization fails.
    """
    # Convert SMILES to an RDKit molecule
    mol = dm.to_mol(smiles)
    if mol is None:
        print(f"Error: Unable to parse SMILES '{smiles}'")
        return None

    # Standardize the molecule using datamol's built-in routines.
    std_mol = dm.standardize_mol(mol)
    if std_mol is None:
        print(f"Error: Standardization failed for '{smiles}'")
        return None

    # Convert the standardized molecule back to a canonical SMILES string.
    std_smiles = dm.to_smiles(std_mol)
    return std_smiles
def fix_csv() :
    new_rows = []
    fieldnames=[]
    # open file to read
    with open('toxcast_data.csv', newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      # get field names
      fieldnames = reader.fieldnames
      # loop thru the
      for row in reader:
        s=row['smiles'];
        # standardize
        std= standardize_smiles(s)
        print(f"Original: {s}\nStandardized: {std}\n")
        # update the value
        row['smiles']= std
        # add the row
        new_rows.append(row)
    # write the result to file
    with open('result.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
# Example usage:

if __name__ == "__main__":
    # Example list of SMILES (including one with a salt)
    fix_csv()