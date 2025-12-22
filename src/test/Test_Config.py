import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Argparse for compound-protein interactions prediction")
    parser.add_argument("-multitask", type=bool, default=True, help="multitask")
    parser.add_argument("-best_model", type=str, default="../checkpoints/checkpoint_fp/checkpoint.pt")

    parser.add_argument('-fusion_type', type=str, default='bidirectional', 
                        help='bidirectional, cross_attention, co_attention, multi_head_cross, ban')
    parser.add_argument('-final_predict_layer', type=str, default='OuterProductLinear', 
                        help='final prediction layer: ConcatMLP, HadamardMLP, DiffAbsConcatMLP, \
                            BilinearOuterProduct, BilinearProjection, GatedFusion, EnsembleFusion, OuterProductLinear')
    
    parser.add_argument("-mode", type=str, default="gpu", help="gpu/cpu")
    parser.add_argument("-cuda", type=str, default="1", help="visible cuda devices")
    
    parser.add_argument("-vocab", type=str, default="../vocab/1024/vocab.pkl", help="vocab path",)
    parser.add_argument("-vocab_size", type=int, default=287, help="vocab size")
    parser.add_argument("-prot_esmc_embed", type=str, default="../data/demo_dict_esmc.pkl", help="pretrained protein embed")
    parser.add_argument("-fingerprint_dict_path", type=str, default="../datasets/datasets_fp/fingerprint_dict.pkl")
    parser.add_argument("-atom_dict_path", type=str, default="../datasets/datasets_fp/atom_dict.pkl")
    parser.add_argument("-bond_dict_path", type=str, default="../datasets/datasets_fp/bond_dict.pkl")

    # Decoder
    parser.add_argument("-decoder_layers", type=int, default=10, help="decoder_layers")
    parser.add_argument("-num_decoder_head", type=int, default=8, help="num_decoder_head")
    parser.add_argument("-ffn_depth", type=int, default=1148, help="ffn_depth")
    parser.add_argument("-model_depth", type=int, default=80, help="decoder model depth")
    parser.add_argument("-decoder_prot_max_len", type=int, default=1024, help="decoder_prot_max_len")

    parser.add_argument("-gat_dim", type=int, default=100, help="dimension of node feature in graph attention layer")
    parser.add_argument("-comp_dim", type=int, default=100, help="dimension of compound atoms feature")
    parser.add_argument("-prot_dim", type=int, default=100, help="dimension of protein amino feature")
    parser.add_argument("-latent_dim", type=int, default=100, help="dimension of compound and protein feature")
    parser.add_argument("-num_head", type=int, default=5, help="number of graph attention layer head")
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-alpha", type=float, default=0.1, help="LeakyReLU alpha")
    parser.add_argument("-num_layers", type=int, default=4, help="number of layer")

    parser.add_argument("-test_data", type=str, default="../data/data.csv", help="test data path")
    parser.add_argument("-batch_size", type=int, default=1, help="test batch size")
    
    test_params, _ = parser.parse_known_args()
    return test_params
