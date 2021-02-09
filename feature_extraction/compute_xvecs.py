"""
Created by José Vicente Egas López
on 2021. 02. 09. 12 47

Class based on https://github.com/manojpamk/pytorch_xvectors/
"""
import argparse
import os

import torch

import get_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', default='xvecTDNN', help='Choose from dnn_models.py ')
    parser.add_argument('-num_classes', default=10, type=int, help='Number of output labels for model')
    parser.add_argument('-layer_name', default='fc1', help="DNN layer for extracting the embeddings")
    parser.add_argument('-n_proc', default=0, type=int,
                        help='Number of parallel processes. Default=0 (Number of input directory splits)')
    parser.add_argument('model_dir', help='Directory containing the model checkpoints')
    parser.add_argument('feat_dir', help='Directory containing features ready for extraction')
    parser.add_argument('xvecs_out_dir', help='Output directory')

    args = parser.parse_args()

    if not os.path.exists(args.xvecs_out_dir):
        os.makedirs(args.xvecs_out_dir)

    # Load model
    net = eval('{}({}, p_dropout=0)'.format(args.model_type, args.num_classes))
    net.load_state_dict(torch.load(args.model_dir))
    net.eval()
    net.cuda()

    # Getting input feature files
    get_feats.extract_xvecs(source_path=args.feat_dir, out_dir=args.xvecs_out_dir, net=net, layerName=args.layer_name)


if __name__ == '__main__':
    main()
