import pandas as pd
import numpy as np
import re

ent_lab = pd.read_csv('mainkg/embeddings/transe/ent_labels.tsv', sep="\t", header=None)
# print(ent_lab)
try:
    ent_err = pd.read_csv('mainkg/embeddings/transe/error_ent_label.csv')
except:
    exit(1)
# print(ent_err)
rel_lab = pd.read_csv('mainkg/embeddings/transe/rel_labels.tsv', sep="\t", header=None)
# print(rel_lab)
try:
    rel_err = pd.read_csv('mainkg/embeddings/transe/error_rel_label.csv')
except:
    exit(1)
# print(rel_err)
ent_l = np.zeros(ent_lab.shape)
ent_l = ent_lab.to_numpy()
for label, key in zip(ent_err['label'],ent_err['key']):
    ent_l = np.insert(ent_l, key, label, axis=0)
for i in range(ent_l.shape[0]):
    ent_l[i] = re.sub(r'"', '', str(ent_l[i]))
    ent_l[i] = re.sub(r'\\', '', str(ent_l[i][0]))
    ent_l[i] = ent_l[i][0][2:len(ent_l[i][0])-2].strip()
ent_lab = pd.DataFrame(ent_l)
rel_l = np.zeros(rel_lab.shape)
rel_l = rel_lab.to_numpy()
for label, key in zip(rel_err['label'],rel_err['key']):
    rel_l = np.insert(rel_l, key, label, axis=0)
for i in range(rel_l.shape[0]):
    rel_l[i] = re.sub(r'"', '', str(rel_l[i]))
    rel_l[i] = re.sub(r'\\', '', str(rel_l[i][0]))
    rel_l[i] = rel_l[i][0][2:len(rel_l[i][0])-2].strip()
rel_lab = pd.DataFrame(rel_l)
ent_lab.to_csv('mainkg/embeddings/transe/correct_ent_labels.csv')
rel_lab.to_csv('mainkg/embeddings/transe/correct_rel_labels.csv')



