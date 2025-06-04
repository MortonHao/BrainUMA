import setting
import train

args = setting.get_args()

if __name__ == '__main__':
    path_list = []
    for atlas in args.atlas:
        path_data = f'./data/correlation/pcc_{atlas}.npy'
        path_list.append(path_data)
    path_label = './data/label.txt'
    print(args)
    train.train_and_test(args, path_list, path_label, args.temperature, args.k, args.threshold)
