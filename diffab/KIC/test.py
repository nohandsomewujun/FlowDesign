import pickle
import torch

with open ('/datapool/data2/home/majianzhu/rectflow_seq_sabdab/hiv_target/template/template.pkl', 'rb') as f:
    a = pickle.load(f)

print(a)