import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from scripts.losses import *
from model.net import ADRNet

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--coal', action='store_true', help='output coal dataset')

eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./pretrained_weights/epoch.pth')

ep = eval_parser.parse_args()

def eval(model, testing_data_loader, model_path, output_folder,
         norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0, gamma=1.0):

    torch.set_grad_enabled(False)

    device = next(model.parameters()).device

    if model_path is not None and model_path != "":
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print('Pre-trained model is loaded.')

    model.to(device)
    model.eval()
    print('Evaluation:')

    os.makedirs(output_folder, exist_ok=True)

    for batch in testing_data_loader:
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
                h, w = None, None
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input = input.to(device, non_blocking=True)

            output = model(input ** gamma)
            output = torch.clamp(output, 0, 1)

            if not norm_size:
                h = int(h.item()) if torch.is_tensor(h) else int(h)
                w = int(w.item()) if torch.is_tensor(w) else int(w)
                output = output[:, :, :h, :w]

            output_img = transforms.ToPILImage()(output[0].detach().cpu())
            output_img.save(os.path.join(output_folder, name[0]))

    print('===> End evaluation')
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    if ep.coal:
        eval_data = DataLoader(dataset=get_eval_set_coal("/data/yz/Dataset/coal"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/coal/'
        weight_path = ''
    
    elif ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("/data/yz/LOLv1/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './myresult/'
        weight_path = ''
        
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        weight_path = ''
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("/data1/yz/Dataset/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        weight_path = ''
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/data/yz/Dataset/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/data/yz/Dataset/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/data/yz/Dataset/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/data/yz/Dataset/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/data/yz/Dataset/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        norm_size = False
        weight_path = ep.unpaired_weights
    
    model = ADRNet()
    eval(model, eval_data, weight_path, output_folder,
         norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0, gamma=1.0)


   
