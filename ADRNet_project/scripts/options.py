import argparse

def option():
    print("Using MY scripts/options.py")
    # Training settings
    parser = argparse.ArgumentParser(description='LLIE Net')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=200, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--init_seed', type=bool, default=False)
    # parser.add_argument('--seed', type=int, default=83463, help='seed for initiative')
    parser.add_argument('--seed', type=int, default=171809, help='seed for initiative')
    
    parser.add_argument('--gpu_id',type=int,default=0,help='GPU id to use, e.g. 0,1')
    
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
    parser.add_argument('--gtmean_loss', type=bool, default=True, help='Is it uses gtmean_loss ?')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # valing 监控过拟合
    parser.add_argument('--val_dir'   , type=str, default='./datasets/LOLv2/Real_captured/val')

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='/data1/yz/Dataset/LOLv1/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='/data1/yz/Dataset/LOLv2/Real_captured/train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='/data1/yz/Dataset/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='/data1/yz/Dataset/processed_sid_png/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_Nikon'       , type=str, default='/data1/yz/Dataset/Nikon/train')
    parser.add_argument('--data_train_Huawei'       , type=str, default='/data1/yz/Dataset/Huawei/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='/data1/yz/Dataset/LOLv1/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='/data1/yz/Dataset/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='/data1/yz/Dataset/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='/data1/yz/Dataset/processed_sid_png/eval/low')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_Huawei'         , type=str, default='/data1/yz/Dataset/Huawei/eval/low')
    parser.add_argument('--data_val_Nikon'         , type=str, default='/data1/yz/Dataset/Nikon/eval/low')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='/data1/yz/Dataset/LOLv1/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='/data1/yz/Dataset/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='/data1/yz/Dataset/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='/data1/yz/Dataset/processed_sid_png/eval/high/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_Huawei'         , type=str, default='/data/yz/Dataset/Huawei/eval/high/')
    parser.add_argument('--data_valgt_Nikon'         , type=str, default='/data/yz/Dataset/Nikon/eval/high/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights default
    parser.add_argument('--L1_weight'   , type=float, default=1.00)
    parser.add_argument('--ssim_weight' , type=float, default=0.50)
    parser.add_argument('--edge_weight' , type=float, default=50.0)
    parser.add_argument('--per_weight'  , type=float, default=5e-3)

    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    # parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    # parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=False, help='if gradient fluctuates too much, turn-on it')
    parser.add_argument('--grad_clip_norm', type=float, default=2.0, help='max norm for gradient clipping')
    
    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default=False)
    parser.add_argument('--LSRW_Huawei', type=bool, default=False)
    parser.add_argument('--LSRW_Nikon', type=bool, default=False)
    parser.add_argument('--lolv2_real', type=bool, default=True)
    parser.add_argument('--lolv2_syn', type=bool, default=False)
    parser.add_argument('--lol_blur', type=bool, default=False)
    parser.add_argument('--SID', type=bool, default=False)
    parser.add_argument('--SICE_mix', type=bool, default=False)
    parser.add_argument('--SICE_grad', type=bool, default=False)
    
    return parser