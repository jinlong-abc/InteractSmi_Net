import argparse


def get_config():
    parser = argparse.ArgumentParser(description="Argparse for compound-protein interactions prediction")

    # 最好模型参数
    parser.add_argument("-best_model", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/checkpoints/PDBbind_for_fishing_intimedict/best_models/best_test_model.pt")

    # 模型架构
    parser.add_argument('-fusion_type', type=str, default='bidirectional', help='bidirectional, cross_attention, co_attention, multi_head_cross, ban')
    parser.add_argument('-final_predict_layer', type=str, default='OuterProductLinear', 
                        help='final prediction layer: ConcatMLP, HadamardMLP, DiffAbsConcatMLP, BilinearOuterProduct, BilinearProjection, GatedFusion, EnsembleFusion, OuterProductLinear')
    
    # 设备参数
    parser.add_argument("-mode", type=str, default="gpu", help="gpu/cpu")
    parser.add_argument("-cuda", type=str, default="1", help="visible cuda devices")
    
    # 公用数据
    parser.add_argument("-prot_esmc_embed", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/data/BindingDB_BindingNet_PDBbind_all/BindingDB_PDBbind_BindingNet_esmc_pad512.pkl", help="pretrained protein embed")
    parser.add_argument("-fingerprint_dict_path", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/datasets/PDBbind_for_fishing_intimedict/fingerprint_dict.pkl")
    parser.add_argument("-atom_dict_path", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/datasets/PDBbind_for_fishing_intimedict/atom_dict.pkl")
    parser.add_argument("-bond_dict_path", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/datasets/PDBbind_for_fishing_intimedict/bond_dict.pkl")

    # GAT
    parser.add_argument("-gat_dim", type=int, default=100, help="dimension of node feature in graph attention layer")
    parser.add_argument("-comp_dim", type=int, default=100, help="dimension of compound atoms feature")
    parser.add_argument("-prot_dim", type=int, default=100, help="dimension of protein amino feature")
    parser.add_argument("-latent_dim", type=int, default=100, help="dimension of compound and protein feature")
    parser.add_argument("-num_head", type=int, default=5, help="number of graph attention layer head")
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-alpha", type=float, default=0.1, help="LeakyReLU alpha")
    parser.add_argument("-num_layers", type=int, default=4, help="number of layer")

    # 测试集
    parser.add_argument("-test_data", type=str, default="/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/src/test/Target_fishing/利伐沙班/1_Target_fishing_data_LFSB_pdbbind_only.csv", help="test data path")
    parser.add_argument("-batch_size", type=int, default=1, help="test batch size")
    # parser.add_argument("-num_workers", type=int, default=4, help="dataloader num_workers")

    test_params, _ = parser.parse_known_args()
    return test_params
