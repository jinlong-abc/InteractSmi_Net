#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import h5py
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from data.graph_builder import MolecularGraphBuilder

class HDF5DataProcessor:
    def __init__(self, compression: str = 'gzip', compression_opts: int = 9, args=None):
        self.compression = compression
        self.compression_opts = compression_opts
        # 加载字典
        if args is None or not hasattr(args, 'dict_prefix'):
            self.graph_builder = MolecularGraphBuilder()
        else:
            dicts = MolecularGraphBuilder.load_dicts(prefix=args.dict_prefix)
            self.graph_builder = MolecularGraphBuilder(load_dicts=dicts)

    def save_dataset(self, 
                    output_path: str,
                    compounds: List[np.ndarray],
                    adjacencies: List[np.ndarray],
                    proteins: List[str],
                    interactions: List[float],
                    id_list: List[str],
                    uniprot_list: List[str],
                    smiles_list: List[str],
                    fusionsmi_list: List[str],
                    metadata: Optional[Dict] = None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            molecular_group = f.create_group('molecular')
            compounds_serialized = pickle.dumps(compounds)
            adjacencies_serialized = pickle.dumps(adjacencies)
            
            molecular_group.create_dataset(
                'compounds', 
                data=np.frombuffer(compounds_serialized, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            molecular_group.create_dataset(
                'adjacencies',
                data=np.frombuffer(adjacencies_serialized, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            protein_group = f.create_group('protein')
            proteins_encoded = [s.encode('utf-8') for s in proteins]
            proteins_serialized = pickle.dumps(proteins_encoded)
            
            protein_group.create_dataset(
                'sequences',
                data=np.frombuffer(proteins_serialized, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            interaction_group = f.create_group('interaction')
            
            interactions_array = np.array(interactions, dtype=np.float32)
            interaction_group.create_dataset(
                'values',
                data=interactions_array,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            identifier_group = f.create_group('identifiers')
            id_encoded = [s.encode('utf-8') for s in id_list]
            uniprot_encoded = [s.encode('utf-8') for s in uniprot_list]
            
            identifier_group.create_dataset(
                'compound_ids',
                data=id_encoded,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            identifier_group.create_dataset(
                'uniprot_ids',
                data=uniprot_encoded,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            decoder_group = f.create_group('decoder_data')

            smiles_encoded = [s.encode('utf-8') for s in smiles_list]
            fusionsmi_encoded = [s.encode('utf-8') for s in fusionsmi_list]

            decoder_group.create_dataset(
                'smiles',
                data=smiles_encoded,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            decoder_group.create_dataset(
                'fusionsmi',
                data=fusionsmi_encoded,
                compression=self.compression,
                compression_opts=self.compression_opts
            )

            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.create_dataset(
                            key,
                            data=np.frombuffer(pickle.dumps(value), dtype=np.uint8),
                            compression=self.compression
                        )
            
            f.attrs['version'] = '1.0'
            f.attrs['created_by'] = 'HDF5DataProcessor'
            f.attrs['data_format'] = 'molecular_protein_interaction'
            f.attrs['total_samples'] = len(compounds)
            
        print(f"数据已成功保存到: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    def process_csv_to_hdf5(self, 
                           csv_path: str, 
                           output_path: str,
                           radius: int = 2,
                           chunk_size: int = 1000) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        print(f"读取CSV文件: {csv_path}")
        data = pd.read_csv(csv_path, header=0)
        
        compounds, adjacencies, proteins, interactions = [], [], [], []
        id_list, uniprot_list = [], []
        smiles_list, fusionsmi_list = [], []
        
        total_rows = len(data)
        print(f"开始处理 {total_rows} 行数据...")

        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk = data.iloc[i:end_idx]

            normal_smiles_list = chunk['Normal SMILES'].tolist()
            chunk_compounds, chunk_adjacencies = self.graph_builder.batch_mol_to_graph(
                normal_smiles_list, radius
            )
            
            compounds.extend(chunk_compounds)
            adjacencies.extend(chunk_adjacencies)
            proteins.extend(chunk['Sequence'].tolist())
            interactions.extend(chunk['pInteraction'].tolist())
            id_list.extend(chunk['ID'].tolist())
            uniprot_list.extend(chunk['UniProt'].tolist())
            # decoder
            smiles_list.extend(chunk['SMILES'].tolist())
            fusionsmi_list.extend(chunk['FusionSmi'].tolist())
        
        metadata = {
            'source_csv': csv_path,
            'processing_params': {
                'radius': radius,
                'chunk_size': chunk_size
            },
            'graph_builder_stats': self.graph_builder.get_statistics()
        }
        
        self.save_dataset(
            output_path=output_path,
            compounds=compounds,
            adjacencies=adjacencies,
            proteins=proteins,
            interactions=interactions,
            id_list=id_list,
            uniprot_list=uniprot_list,
            smiles_list=smiles_list, 
            fusionsmi_list=fusionsmi_list, 
            metadata=metadata
        )
        
        print(f"处理完成！数据已保存到: {output_path}")

    def load_dataset(self, data_path: str) -> list:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        with h5py.File(data_path, 'r') as f:
            mol_group = f['molecular']
            compounds_bytes = mol_group['compounds'][()]
            compounds_serialized = compounds_bytes.tobytes()
            compounds = pickle.loads(compounds_serialized)
            
            adjacencies_bytes = mol_group['adjacencies'][()]
            adjacencies_serialized = adjacencies_bytes.tobytes()
            adjacencies = pickle.loads(adjacencies_serialized)
            
            prot_group = f['protein']
            proteins_bytes = prot_group['sequences'][()]
            proteins_serialized = proteins_bytes.tobytes()
            proteins_encoded = pickle.loads(proteins_serialized)
            proteins = [s.decode('utf-8') for s in proteins_encoded]
            
            inter_group = f['interaction']
            interactions = inter_group['values'][()]
            id_group = f['identifiers']
            ID_list = [s.decode('utf-8') for s in id_group['compound_ids'][()]]
            Uniprot_list = [s.decode('utf-8') for s in id_group['uniprot_ids'][()]]
        
            decoder_group = f['decoder_data']
            smiles_list = [s.decode('utf-8') for s in decoder_group['smiles'][()]]
            fusionsmi_list = [s.decode('utf-8') for s in decoder_group['fusionsmi'][()]]

        data_list = [
            np.array(compounds, dtype=object),     # 0
            np.array(adjacencies, dtype=object),   # 1
            np.array(proteins, dtype=object),      # 2
            np.array(interactions, dtype=float),   # 3
            np.array(ID_list, dtype=object),       # 4
            np.array(Uniprot_list, dtype=object),  # 5
            np.array(smiles_list, dtype=object),   # 6
            np.array(fusionsmi_list, dtype=object) # 7
        ]
        
        return data_list



