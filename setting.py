import argparse


def get_args():
    parser = argparse.ArgumentParser(description="BrainHTDL")
    parser.add_argument("--gpu", type=str, default=0, help='gpu设备号')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_atlas", type=int, default=5, help='输入的数据集数量')
    parser.add_argument("--atlas", type=list, default=['cc200','ho','aal','tt','ez'],
                        help="choose in cc200, aal, ho, tt, cc400, ez and dos160")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kFold", type=int, default=5)
    parser.add_argument("--epoch_cf", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--e2e_feature1", type=int, default=16)
    parser.add_argument("--e2e_feature2", type=int, default=64)
    parser.add_argument("--gcn_hid", type=int, default=256)
    parser.add_argument("--gcn_out", type=int, default=64)

    parser.add_argument("--lambda_cont", type=float, default=0.001)
    parser.add_argument("--lambda_cc", type=float, default=0.001)
    # 重要超参
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=0.6)
    # parser.add_argument("--k2", type=int, default=7)

    args = parser.parse_args()

    return args
