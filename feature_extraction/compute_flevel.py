"""
Created by José Vicente Egas López
on 2021. 02. 09. 14 40

"""
import sys
sys.path.append('../')

import argparse

import get_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-audio_path', required=True, help='Path to the utterances.')
    parser.add_argument('-out_dir', required=True, help='Path to save the frame-level features.')
    parser.add_argument('-feat_type', type=str, default='mfcc', required=True,
                        help="Type of the frame-level features to load or extract. Available types: mfcc, fbanks, "
                             "spec, melspecT")
    parser.add_argument('-deltas', type=int, required=True,
                        help="Compute delta coefficients of a tensor. '1' for first order derivative, '2' for second "
                             "order. None for not using deltas. Default: None.")
    parser.add_argument('-config', required=True, help='Path to the config (ini) file.')

    args = parser.parse_args()

    get_feats.compute_feats_offline(source_path=args.audio_path, out_dir=args.out_dir, feat_type=args.feat_type,
                                    deltas=args.deltas, config_file=args.config)


if __name__ == '__main__':
    main()
