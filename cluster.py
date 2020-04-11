from rdkit.Chem import PandasTools
import gzip

import numpy as np
from janitor import chemistry
import pandas as pd
import scipy

import argparse
import time

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import preprocessing


def parse_args():
   parser = argparse.ArgumentParser(description='Run AP clustering on a files containing smiles or a sdf')
   parser.add_argument("-i", help="the input txt/.sd")
   parser.add_argument("-o", help="the output .sd")
   parser.add_argument("-damping", help="the damping factor of the AP clustering")
   parser.add_argument("-max_iter", help="the maximum no of iterations of the AP clustering")
   parser.add_argument("-convergence", help="the maximum no of iterations with no change for stopping upon convergence")
   args = parser.parse_args()
   return args

def parse_text_file(file):
    """
    parse a csv file and return molecules
    """
    data = pd.read_csv(file)
    PandasTools.AddMoleculeColumnToFrame(data, "SMILES", "Molecule")
    return data

def parse_sd_file(file, tgz=False):
    """
    parse a sd file and return molecules
    """
    if tgz == True:
        file = gzip.open(file)
    data = PandasTools.LoadSDF(file, molColName='Molecule', smilesName='smiles')    
    return data

def fp_from_df(data):
    """
    convert molecules to FP df
    """
    FP = chemistry.morgan_fingerprint(data,mols_column_name='Molecule',radius=3, nbits=2048, kind='counts')
    #sparse to prevent OOM errors.
    #FP = scipy.sparse.csc_matrix(FP)
    return FP

def main():
    args = parse_args()
    if (".txt" or ".csv") in args.i:
        df  = parse_text_file(args.i) 
    elif ("sd.gz" or "sdf.gz") in args.i:
        df = parse_sd_file(args.i, tgz=True)
    elif (".sd" or ".sdf") in args.i:
        df = parse_sd_file(args.i)
    FP = fp_from_df(df)
    #FP = preprocessing.normalize(FP)
    labels = AffinityPropagation(damping=float(args.damping), max_iter=int(args.max_iter), convergence_iter=int(args.convergence)).fit(FP).labels_
    print(metrics.silhouette_score(FP,labels, metric='euclidean'))
    df['Cluster'] = labels
    PandasTools.WriteSDF(df, args.o, molColName='Molecule', idName="CID", properties=list(df.columns))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))