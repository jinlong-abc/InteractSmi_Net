import pandas as pd
import pickle as pkl

csv_file = '/home/wjl/data/DTI_prj/InteractSmi_Net/data/data.csv'

data = pd.read_csv(csv_file)

# 构建字典：键为 UniProt ID，值为 Sequence
# CSV 列名: ID,UniProt,Assay Result,pInteraction,Normal SMILES,SMILES,FusionSmi,Mapped_SMILES,IntSeq,Sequence
uniprotid_sequence_dict = {}
for index, row in data.iterrows():
    uniprotid = row['UniProt']
    sequence = row['Sequence']
    uniprotid_sequence_dict[uniprotid] = sequence

print(len(uniprotid_sequence_dict))
output_file = '/home/wjl/data/DTI_prj/InteractSmi_Net/data/demo_uniprot_seq_dict.pkl'

# 保存为 pickle 文件
with open(output_file, 'wb') as f:  # 注意 'wb' 模式
    pkl.dump(uniprotid_sequence_dict, f)

print(f"Dictionary saved to {output_file}, total {len(uniprotid_sequence_dict)} entries.")
