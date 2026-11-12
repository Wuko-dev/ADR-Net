import os
import torch
import random
import numpy as np
import time
from datetime import datetime
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.net import ADRNet
from scripts.options import option
import scripts.options as opt_file
from measure import metrics
from test import eval
from data.data import *
from scripts.losses import *
from scripts.scheduler import *

opt = option().parse_args()


run_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")
print(run_name)
save_path = os.path.join('./pretrained_weights', run_name)
val_folder = os.path.join(opt.val_folder, run_name)
os.makedirs(save_path, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

print(f"===> Checkpoints will be saved to: {save_path}")
print(f"===> Validation images will be saved to: {val_folder}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.benchmark = True
    cudnn.deterministic = False


def train_init():
    if opt.init_seed:
        seed = opt.seed
    else:
        seed = random.randint(1, 1000000)

    print(f"This seed is {seed}.")
    set_seed(seed)

    if opt.gpu_mode and torch.cuda.is_available():
        gpu_id = getattr(opt, "gpu_id", 0)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Device] Using GPU cuda:{gpu_id}")
        print(f"[Device] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[Device] GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    return device


def save_model(model, name):
    path = os.path.join(save_path, name)
    torch.save(model.state_dict(), path)
    # print("Checkpoint saved to {}".format(path))
    return path


def save_training_images(output, gt):
    save_dir = os.path.join(val_folder, 'training')
    os.makedirs(save_dir, exist_ok=True)

    output_img = transforms.ToPILImage()(output[0].detach().cpu().clamp(0, 1).squeeze(0))
    gt_img = transforms.ToPILImage()(gt[0].detach().cpu().clamp(0, 1).squeeze(0))

    output_img.save(os.path.join(save_dir, 'test.png'))
    gt_img.save(os.path.join(save_dir, 'gt.png'))


def get_loaders():
    print('===> Loading datasets')

    train_set = None
    test_set = None
    label_dir = None
    output_folder = ''
    norm_size = True

    if opt.lol_v1:
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        label_dir = opt.data_valgt_lol_v1
        output_folder = 'LOLv1/'

    elif opt.lol_blur:
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)
        label_dir = opt.data_valgt_lol_blur
        output_folder = 'LOL_blur/'

    elif opt.lolv2_real:
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        label_dir = opt.data_valgt_lolv2_real
        output_folder = 'LOLv2_real/'

    elif opt.lolv2_syn:
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
        label_dir = opt.data_valgt_lolv2_syn
        output_folder = 'LOLv2_syn/'

    elif opt.SID:
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        label_dir = opt.data_valgt_SID
        output_folder = 'SID/'

    elif opt.SICE_mix:
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        label_dir = opt.data_valgt_SICE_mix
        output_folder = 'SICE_mix/'
        norm_size = False

    elif opt.SICE_grad:
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        label_dir = opt.data_valgt_SICE_grad
        output_folder = 'SICE_grad/'
        norm_size = False

    else:
        raise Exception("No dataset selected! Please check your options.")

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    return train_loader, test_loader, label_dir, output_folder, norm_size


def build_model(device):
    print('===> Building model ')
    model = ADRNet().to(device)
    return model


def make_scheduler():
    # optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.95), weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001]
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0001, 0.0000001]
            )

    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-6
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7
            )
    else:
        raise Exception("should choose a scheduler")

    return optimizer, scheduler


def init_loss():
    L1_weight = opt.L1_weight
    ssim_weight = opt.ssim_weight
    edge_weight = opt.edge_weight
    per_weight = opt.per_weight

    if opt.gtmean_loss:
        l1_base = L1Loss(loss_weight=L1_weight, reduction='mean').to(device)
        L1_loss = GTMean_L1Loss(module=l1_base, loss_weight='dou_KL_2', reduction='mean', sigma_=0.1).to(device)

        SSIM_base = SSIM(weight=ssim_weight).to(device)
        SSIM_loss = GTMean_SSIM(ssim_module=SSIM_base, loss_weight='dou_KL_2', reduction='mean', sigma_=0.1).to(device)
    else:
        L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean').to(device)
        SSIM_loss = SSIM(weight=ssim_weight).to(device)

    Edge_loss = EdgeLoss(loss_weight=edge_weight).to(device)
    Per_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, perceptual_weight=per_weight, criterion='mse').to(device)

    return L1_loss, SSIM_loss, Edge_loss, Per_loss


def print_hyperparams(opt):
    print("=" * 80)
    print("Hyper-Parameters Configuration")
    print("=" * 80)

    print("[Training]")
    print(f"  gpu_id         : {opt.gpu_id}")
    print(f"  batchSize      : {opt.batchSize}")
    print(f"  cropSize       : {opt.cropSize}")
    print(f"  nEpochs        : {opt.nEpochs}")
    print(f"  snapshots      : {opt.snapshots}")
    print(f"  lr             : {opt.lr}")
    print(f"  shuffle        : {opt.shuffle}")

    print("\n[Scheduler]")
    print(f"  cos_restart_cyclic : {opt.cos_restart_cyclic}")
    print(f"  cos_restart        : {opt.cos_restart}")
    print(f"  warmup_epochs      : {opt.warmup_epochs}")
    print(f"  start_warmup       : {opt.start_warmup}")

    print("\n[Dataset - Train]")
    if opt.lol_v1:
        print("  LOLv1")

    if opt.LSRW_Nikon:
        print("  Nikon")
    
    if opt.LSRW_Huawei:
        print("  Huawei")

    if opt.lolv2_real:
        print("  LOLv2 Real")
    if opt.lolv2_syn:
        print("  LOLv2 Synthetic")
    if opt.lol_blur:
        print("  LOL Blur")
    if opt.SID:
        print("  SID")
    if opt.SICE_mix:
        print("  SICE Mix")
    if opt.SICE_grad:
        print("  SICE Grad")

    
    print("\n[Validation]")
    print(f"  val_folder     : {val_folder}")

    print("\n[Loss Weights]")
    print(f"  L1             : {opt.L1_weight}")
    print(f"  SSIM           : {opt.ssim_weight}")
    print(f"  EDGE           : {opt.edge_weight}")
    print(f"  PERCEPTUAL     : {opt.per_weight}")
    print(f"  gtmean_loss    : {opt.gtmean_loss}")


    print("\n[Stability]")
    print(f"  grad_detect    : {opt.grad_detect}")
    print(f"  grad_clip      : {opt.grad_clip}")
    print(f"  grad_clip_norm : {opt.grad_clip_norm}")

    print("\n[System]")
    print(f"  gpu_mode       : {opt.gpu_mode}")
    print("=" * 80)



def train_one_epoch(model, loader, optimizer, losses, device, epoch):
    model.train()
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    L1_loss_fn, SSIM_loss_fn, Edge_loss_fn, Per_loss_fn = losses

    epoch_metrics = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'edge': 0.0, 'per': 0.0, 'grad_norm': 0.0}
    max_grad_norm = 0.0
    clip_count = 0    
    train_len = len(loader)

    last_output = None
    last_gt = None

    for i, batch in enumerate(loader, start=1):
        im_low = batch[0].to(device)
        im_gt = batch[1].to(device)

        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            input_img = im_low ** gamma
        else:
            input_img = im_low

        output = model(input_img)

        l1 = L1_loss_fn(output, im_gt)
        ds = SSIM_loss_fn(output, im_gt)
        eg = Edge_loss_fn(output, im_gt)
        pe = Per_loss_fn(output, im_gt)[0]

        loss = l1 + ds + eg + pe

        optimizer.zero_grad()
        loss.backward()

        if opt.grad_clip:
            # 注意：clip_grad_norm_ 的返回值是“裁剪前”的总梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip_norm, norm_type=2.0)
        else:
            parameters = [p for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2) if len(parameters) > 0 else torch.tensor(0.0, device=device)

        grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)

        if opt.grad_clip and grad_norm_value > opt.grad_clip_norm:
            clip_count += 1

        max_grad_norm = max(max_grad_norm, grad_norm_value)

        optimizer.step()

        epoch_metrics['total'] += loss.item()
        epoch_metrics['l1'] += l1.item()
        epoch_metrics['ssim'] += ds.item()
        epoch_metrics['edge'] += eg.item()
        epoch_metrics['per'] += pe.item()
        epoch_metrics['grad_norm'] += grad_norm_value

        last_output = output
        last_gt = im_gt

        if i == train_len:
            avg_loss = epoch_metrics['total'] / train_len
            # print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch, avg_loss, optimizer.param_groups[0]['lr']))
            print("===> Epoch[{}]: Loss: {:.4f} || GradNorm: {:.4f} || MaxGradNorm: {:.4f} || ClipRatio: {:.2f}% || Learning rate: lr={}.".format(
                epoch,
                avg_loss,
                epoch_metrics['grad_norm'] / train_len,
                max_grad_norm,
                100.0 * clip_count / train_len,
                optimizer.param_groups[0]['lr']
            ))
            save_training_images(last_output, last_gt)

    # for k in epoch_metrics:
    #     epoch_metrics[k] /= train_len

    # return epoch_metrics
    for k in epoch_metrics:
        epoch_metrics[k] /= train_len

    epoch_metrics['max_grad_norm'] = max_grad_norm
    epoch_metrics['clip_ratio'] = clip_count / train_len

    return epoch_metrics

if __name__ == '__main__':
    print_hyperparams(opt)

    device = train_init()
    model = build_model(device)

    train_loader, test_loader, label_dir, out_subfolder, norm_size = get_loaders()

    optimizer, scheduler = make_scheduler()
    losses = init_loss()

    tb_writer = SummaryWriter(log_dir=f'./runs/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}')

    best_stats = {
        'std_psnr': 0.0,
        'std_ssim': 0.0,
        'std_lpips': 100.0,
        'gt_psnr': 0.0,
        'gt_ssim': 0.0,
        'gt_lpips': 100.0,
    }

    print(f"===> Start Training. Device: {device}")

    start_time = time.time()

    for epoch in range(opt.start_epoch + 1, opt.nEpochs + opt.start_epoch + 1):
        m = train_one_epoch(model, train_loader, optimizer, losses, device, epoch)
        scheduler.step()

        tb_writer.add_scalar('loss/total_loss', m['total'], epoch)
        tb_writer.add_scalar('loss/l1_loss', m['l1'], epoch)
        tb_writer.add_scalar('loss/ssim_loss', m['ssim'], epoch)
        tb_writer.add_scalar('loss/edge_loss', m['edge'], epoch)
        tb_writer.add_scalar('loss/per_loss', m['per'], epoch)
        tb_writer.add_scalar('train/grad_norm_before_clip', m['grad_norm'], epoch)
        tb_writer.add_scalar('train/max_grad_norm_before_clip', m['max_grad_norm'], epoch)
        tb_writer.add_scalar('train/clip_ratio', m['clip_ratio'], epoch)
        
        if epoch % opt.snapshots == 0:
            model_out_path = save_model(model, f"epoch_{epoch}.pth")

            model.eval()

            output_path = os.path.join(val_folder, out_subfolder)
            os.makedirs(output_path, exist_ok=True)

            with torch.no_grad():
                eval(model, test_loader, model_out_path, output_path, norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)

            im_dir = os.path.join(output_path, '*.png')

            avg_psnr, avg_ssim, avg_lpips, gt_avg_psnr, gt_avg_ssim, gt_avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)

            cur_std = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
            cur_gt = {'psnr': gt_avg_psnr, 'ssim': gt_avg_ssim, 'lpips': gt_avg_lpips}

            tb_writer.add_scalar('metrics/psnr', round(avg_psnr, 4), epoch)
            tb_writer.add_scalar('metrics/ssim', round(avg_ssim, 4), epoch)
            tb_writer.add_scalar('metrics/lpips', round(avg_lpips, 4), epoch)
            tb_writer.add_scalar('metrics_gt/gt_psnr', round(gt_avg_psnr, 4), epoch)
            tb_writer.add_scalar('metrics_gt/gt_ssim', round(gt_avg_ssim, 4), epoch)
            tb_writer.add_scalar('metrics_gt/gt_lpips', round(gt_avg_lpips, 4), epoch)

            print("===> Avg.PSNR: {:.4f} dB   Avg.SSIM: {:.4f}   Avg.LPIPS: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips))
            print("===> GTM.PSNR: {:.4f} dB   GTM.SSIM: {:.4f}   GTM.LPIPS: {:.4f}".format(gt_avg_psnr, gt_avg_ssim, gt_avg_lpips))

            if cur_std['psnr'] > best_stats['std_psnr']:
                best_stats['std_psnr'] = cur_std['psnr']
                save_model(model, "best_psnr.pth")

            if cur_std['ssim'] > best_stats['std_ssim']:
                best_stats['std_ssim'] = cur_std['ssim']
                save_model(model, "best_ssim.pth")
                # print(f"Epoch: {epoch} has std_best_ssim !")

            if cur_std['lpips'] < best_stats['std_lpips']:
                best_stats['std_lpips'] = cur_std['lpips']
                save_model(model, "best_lpips.pth")
                # print(f"Epoch: {epoch} has std_best_lpips !")

            if cur_gt['psnr'] > best_stats['gt_psnr']:
                best_stats['gt_psnr'] = cur_gt['psnr']
                save_model(model, "best_gt_psnr.pth")
                # print(f"Epoch: {epoch} has gt_best_psnr !")

            if cur_gt['ssim'] > best_stats['gt_ssim']:
                best_stats['gt_ssim'] = cur_gt['ssim']
                save_model(model, "best_gt_ssim.pth")
                # print(f"Epoch: {epoch} has gt_best_ssim !")

            if cur_gt['lpips'] < best_stats['gt_lpips']:
                best_stats['gt_lpips'] = cur_gt['lpips']
                save_model(model, "best_gt_lpips.pth")
                # print(f"Epoch: {epoch} has gt_best_lpips !")

            # print_best_stats(best_stats)

        torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"total_time: {hours}h {minutes}m {seconds}s")

    tb_writer.close()