import networkx as nx
import pandas as pd
import numpy as np

from tqdm import tqdm
from numpy.linalg import inv


def generate_feature(G, u, v, fN):
    '''
    Generate the value of the feature
    - Parameters:
        G: a network
        u: the source node
        v: the sink node
        fN: a string representing the name of the feature
    - Return:
        a tuple ((u, v), p) where u and v are the source and sink node respectively
        and p is the value of the feature
    '''
    # Common Neighbours
    if fN == "CN":
        if G.has_node(u) and G.has_node(v):
            cn = sorted(nx.common_neighbors(G,u,v))
            return ((u,v), len(cn))
        else:
            return ((u,v), 0)
    
    # Adamic-Adar Index
    if fN == "AA":
        if G.has_node(u) and G.has_node(v):
            aa = list(nx.adamic_adar_index(G, [(u,v)]))[0]
            return ((aa[0], aa[1]), aa[2])
        else:
            return ((u,v), 0)
    
    # Resource Allocation
    if fN == "RA":
        if G.has_node(u) and G.has_node(v):
            ra = list(nx.resource_allocation_index(G, [(u,v)]))[0]
            return ((ra[0], ra[1]), ra[2])
        else:
            return ((u,v), 0)
    
    # Jaccard's Coeffiecient
    if fN == "JC":
        if G.has_node(u) and G.has_node(v):
            jc = list(nx.jaccard_coefficient(G, [(u,v)]))[0]
            return ((jc[0], jc[1]), jc[2])
        else:
            return ((u,v), 0)
    
    # Preferential Attachment
    if fN == "PA":
        if G.has_node(u) and G.has_node(v):
            pa = list(nx.preferential_attachment(G, [(u,v)]))[0]
            return ((pa[0], pa[1]), pa[2])
        else:
            return ((u,v), 0)

def add_feature(G, adj_G, nodeL, df, fN):
    if fN in ["CN", "AA", "RA", "JC", "PA"]:
        fL = []
        for u, v in tqdm(df.Pair):
            fv = generate_feature(G, u, v, fN)
            if fv[0][0] != -1:
                fL.append(fv)
            else:
                print("Value not correctly generated!")
          
        fDF = pd.DataFrame(fL, columns=['Pair', fN])
        df = df.join(fDF.set_index('Pair'), on='Pair')
    
    if fN == "KI":
        # The calculation of the Katz index is referred to "Fast Computation of
        # Katz Index for Efficient Processing of Link Prediction Queries":
        # https://arxiv.org/pdf/1912.06525.pdf
        I = np.identity(len(nodeL))
        beta = 0.05
        # beta is set to 0.05 as it is a commonly accepted value in the research
        # community accoring to Qi et. al 'predicting co-author relationship in
        # medical co-authorship networks'

        K = inv(I - adj_G*beta) - I
        
        fL = []
        offset=0
        for i in range(len(nodeL)):
            for j in range(offset, len(nodeL)):
                if i != j:
                    fL.append(((nodeL[i], nodeL[j]), K[i, j]))
            offset += 1
        
        fDF = pd.DataFrame(fL, columns=['Pair', fN])
        df = df.join(fDF.set_index('Pair'), on='Pair')
        df["KI"] = df["KI"].fillna(0)
    
    if fN == "PR":
        pr = nx.pagerank(G)
        prDF = pd.DataFrame.from_dict(pr, orient='index', columns=['PR_s1'])
        df = df.join(prDF, on="Source")
        df["PR_s1"] = df["PR_s1"].fillna(0)
        prDF = prDF.rename(columns = {'PR_s1':'PR_s2'})
        df = df.join(prDF, on="Sink")
        df["PR_s2"] = df["PR_s2"].fillna(0)
        
    return df

def add_all_feature(G, adj_G, nodeL, df):
    for fN in ["CN", "AA", "RA", "JC", "PA", "KI", "PR"]:
        df = add_feature(G, adj_G, nodeL, df, fN)
    return df
