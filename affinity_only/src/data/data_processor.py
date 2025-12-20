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
        """
        Parameters:
        compression : str, default='gzip'
            压缩方法 ('gzip', 'lzf', 'szip', None)
        compression_opts : int, default=9
            压缩级别 (0-9, 9为最高压缩)
        """
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
                    metadata: Optional[Dict] = None):
        """
        保存完整数据集到HDF5文件
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # 保存分子数据组
            molecular_group = f.create_group('molecular')
            
            # 序列化复杂对象数组
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
            
            # 保存蛋白质数据组
            protein_group = f.create_group('protein')
            
            # 将字符串列表编码为字节
            proteins_encoded = [s.encode('utf-8') for s in proteins]
            proteins_serialized = pickle.dumps(proteins_encoded)
            
            protein_group.create_dataset(
                'sequences',
                data=np.frombuffer(proteins_serialized, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # 保存相互作用数据
            interaction_group = f.create_group('interaction')
            
            interactions_array = np.array(interactions, dtype=np.float32)
            interaction_group.create_dataset(
                'values',
                data=interactions_array,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # 保存标识符数据组
            identifier_group = f.create_group('identifiers')
            
            # 编码字符串ID
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
            
            # 保存元数据
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
                    else:
                        # 复杂对象序列化保存
                        metadata_group.create_dataset(
                            key,
                            data=np.frombuffer(pickle.dumps(value), dtype=np.uint8),
                            compression=self.compression
                        )
            
            # 7. 保存全局属性
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
        """
        csv_path : str
            输入CSV文件路径
        output_path : str
            输出HDF5文件路径
        radius : int, default=2
            分子图特征半径
        chunk_size : int, default=1000
            分块处理大小
        ID,UniProt,Assay Result,pInteraction,Normal SMILES,SMILES,FusionSmi,Mapped_SMILES,IntSeq,Sequence
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            
        # 读取CSV数据
        print(f"读取CSV文件: {csv_path}")
        data = pd.read_csv(csv_path, header=0)
        
        compounds, adjacencies, proteins, interactions = [], [], [], []
        id_list, uniprot_list = [], []
        
        total_rows = len(data)
        print(f"开始处理 {total_rows} 行数据...")
        
        # 分块处理数据
        for i in range(0, total_rows, chunk_size): # 开始、结束、step
            end_idx = min(i + chunk_size, total_rows)
            chunk = data.iloc[i:end_idx]
            
            print(f"处理进度: {i+1}-{end_idx}/{total_rows}")
            
            # 批量处理分子SMILES
            smiles_list = chunk['Normal SMILES'].tolist()
            chunk_compounds, chunk_adjacencies = self.graph_builder.batch_mol_to_graph(
                smiles_list, radius
            )
            
            # 收集其他数据
            compounds.extend(chunk_compounds)
            adjacencies.extend(chunk_adjacencies)
            proteins.extend(chunk['Sequence'].tolist())
            interactions.extend(chunk['pInteraction'].tolist())
            id_list.extend([str(x) for x in chunk['ID'].tolist()])
            uniprot_list.extend(chunk['UniProt'].tolist())
        
        # 准备元数据
        metadata = {
            'source_csv': csv_path,
            'processing_params': {
                'radius': radius,
                'chunk_size': chunk_size
            },
            'graph_builder_stats': self.graph_builder.get_statistics()
        }
        
        # 保存到HDF5
        self.save_dataset(
            output_path=output_path,
            compounds=compounds,
            adjacencies=adjacencies,
            proteins=proteins,
            interactions=interactions,
            id_list=id_list,
            uniprot_list=uniprot_list,
            metadata=metadata
        )
        
        print(f"处理完成！数据已保存到: {output_path}")

    def load_dataset(self, data_path: str) -> list:
        """
        data_path : str
            HDF5文件路径
        ====================
        list
            返回按 batch2tensor 需要的列表格式：
            [compounds, adjacencies, proteins, None, None, interactions, None, ID_list, Uniprot_list]
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        with h5py.File(data_path, 'r') as f:
            # 加载分子数据
            mol_group = f['molecular']
            compounds_bytes = mol_group['compounds'][()]
            compounds_serialized = compounds_bytes.tobytes()
            compounds = pickle.loads(compounds_serialized)
            
            adjacencies_bytes = mol_group['adjacencies'][()]
            adjacencies_serialized = adjacencies_bytes.tobytes()
            adjacencies = pickle.loads(adjacencies_serialized)
            
            # 加载蛋白质数据
            prot_group = f['protein']
            proteins_bytes = prot_group['sequences'][()]
            proteins_serialized = proteins_bytes.tobytes()
            proteins_encoded = pickle.loads(proteins_serialized)
            proteins = [s.decode('utf-8') for s in proteins_encoded]
            
            # 加载相互作用数据
            inter_group = f['interaction']
            interactions = inter_group['values'][()]
            
            # 加载标识符数据
            id_group = f['identifiers']
            ID_list = [s.decode('utf-8') for s in id_group['compound_ids'][()]]
            Uniprot_list = [s.decode('utf-8') for s in id_group['uniprot_ids'][()]]
        
        # 按 batch2tensor 需要的顺序返回列表
        data_list = [
            np.array(compounds, dtype=object),     # 0
            np.array(adjacencies, dtype=object),   # 1
            np.array(proteins, dtype=object),      # 2
            np.array(interactions, dtype=float),   # 3
            np.array(ID_list, dtype=object),       # 4
            np.array(Uniprot_list, dtype=object)   # 5
        ]
        
        return data_list



