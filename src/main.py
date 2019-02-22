#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from os.path import exists, join as join_paths
from os import makedirs, link, remove, listdir
from timeit import default_timer as timer
import src.config as config
from src import loss
from src.data_manager import load_img, load_images
from src.model import Net, _make_target_crop
from src.dataset import get_training_set, get_validation_set, get_visual_test_set, pil_to_tensor
from src.interpolate import interpolate, interpolate3toN
from src.utilities import psnr, writeJson

import json

# ----------------------------------------------------------------------

if __name__ == '__main__':
    if config.ALWAYS_CPU:
        print("===> ALWAYS_CPU is True, proceeding with CPU...")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("===> CUDA available, proceeding with GPU...")
        device = torch.device("cuda")
    else:
        print("===> No GPU found, proceeding with CPU...")
        device = torch.device("cpu")

    if config.SEED is not None:
        torch.manual_seed(config.SEED)

# ----------------------------------------------------------------------

if __name__ == '__main__':
    if not config.GENERATE_PARALLAX_VIEW:
        print('===> Loading datasets...')
        
        train_set = get_training_set()
        training_data_loader = DataLoader(dataset=train_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
                                        shuffle=True)

        if config.VISUAL_TEST_ENABLED: #visual test. I never used this in the parallax generation experiments
            visual_test_set = get_visual_test_set()

        if config.VALIDATION_ENABLED: #davis dataset validation, not parallax view dataset validation
            validation_set = get_validation_set()
            validation_data_loader = DataLoader(dataset=validation_set, num_workers=config.NUM_WORKERS,
                                        batch_size=config.BATCH_SIZE, shuffle=False)

def init_model(file=None, quiet=False):
    if file is not None:
        if not quiet:
            print(f'===> Loading pre-trained model: {file}')
        model = Net.from_file(file, config.NET_MODE)
    else:
        if not quiet:
            print('===> Building model...')
        model = Net(config.NET_MODE)
    model.to(device)
    return model

if __name__ == '__main__':
    model = init_model(config.START_FROM_EXISTING_MODEL)

    if config.LOSS == "l1":
        loss_function = nn.L1Loss()
    elif config.LOSS == "vgg":
        loss_function = loss.VggLoss()
    elif config.LOSS == "ssim":
        loss_function = loss.SsimLoss()
    elif config.LOSS == "l1+vgg":
        loss_function = loss.CombinedLoss()
    else:
        raise ValueError(f"Unknown loss: {config.LOSS}")

    optimizer = optim.Adamax(model.parameters(), lr=config.LEARNING_RATE[0])

    board_writer = SummaryWriter()

# ----------------------------------------------------------------------

def train(epoch):
    print("===> Training...")
    before_pass = [p.data.clone() for p in model.parameters()]
    epoch_loss = 0

    target_crop = _make_target_crop(config.PATCH_SIZE[0], config.PATCH_SIZE[1], config.CROP_SIZE, config.CROP_SIZE)

    epoch_lr = config.LEARNING_RATE[-1]
    if epoch < len(config.LEARNING_RATE):
        epoch_lr = config.LEARNING_RATE[epoch]

    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr

    print(f"Epoch {epoch} uses learning rate of {epoch_lr}")

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        target = target_crop(target)

        optimizer.zero_grad()

        output = model(input, config.SEQUENCE_LENGTH, False, True)

        loss_ = loss_function(output, target)

        loss_.backward()

        optimizer.step()

        loss_val = loss_.item()
        epoch_loss += loss_val

        board_writer.add_scalar('data/iter_training_loss', loss_val, iteration)
        if iteration % 50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_val))

    weight_l2s = 0
    weight_diff_l2s = 0
    gradient_l2s = 0
    for i, p in enumerate(model.parameters()):
        weight_l2s += p.data.norm(2)
        weight_diff_l2s += (p.data - before_pass[i]).norm(2)
        gradient_l2s += p.grad.norm(2)
    board_writer.add_scalar('data/epoch_weight_l2', weight_l2s, epoch)
    board_writer.add_scalar('data/epoch_weight_change_l2', weight_diff_l2s, epoch)
    board_writer.add_scalar('data/epoch_gradient_l2', gradient_l2s, epoch)
    epoch_loss /= len(training_data_loader)
    board_writer.add_scalar('data/epoch_training_loss', epoch_loss, epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))


def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = join_paths(config.OUTPUT_DIR, model_out_path)
    model_latest_path = join_paths(config.OUTPUT_DIR, 'model_epoch_latest.pth')
    if not exists(config.OUTPUT_DIR):
        makedirs(config.OUTPUT_DIR)
    torch.save(model.cpu().state_dict(), model_out_path)
    if exists(model_latest_path):
        remove(model_latest_path)
    link(model_out_path, model_latest_path)
    print("Checkpoint saved to {}".format(model_out_path))
    if device.type != 'cpu':
        model.cuda()


def validate(epoch):
    print("===> Running validation...")
    ssmi = loss.SsimLoss()
    valid_loss, valid_ssmi, valid_psnr = 0, 0, 0
    iters = len(validation_data_loader)
    with torch.no_grad():
        for batch in validation_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            valid_loss += loss_function(output, target).item()
            valid_ssmi -= ssmi(output, target).item()
            valid_psnr += psnr(output, target).item()
    valid_loss /= iters
    valid_ssmi /= iters
    valid_psnr /= iters
    board_writer.add_scalar('data/epoch_validation_loss', valid_loss, epoch)
    board_writer.add_scalar('data/epoch_ssmi', valid_ssmi, epoch)
    board_writer.add_scalar('data/epoch_psnr', valid_psnr, epoch)
    print("===> Validation loss: {:.4f}".format(valid_loss))

def generate_parallax_view(torchModel, t, cam_interval, cam_views, netmode):
    """
    cam_views is expected to be an array of pil images
    returns an array of pil images
    """

    if netmode == "2to1":
        output = []
        for w in range(1, t+1):
            if (w - 1) % cam_interval == 0:
                output.append(cam_views[(w-1) // cam_interval])
            else:
                output.append(None)
        
        while cam_interval > 1:
            r_dot = cam_interval // 2
            for w in range(1, t - cam_interval + 1, cam_interval):
                output[w + r_dot - 1] = interpolate(torchModel, output[w - 1], output[w + cam_interval - 1])
            cam_interval = r_dot

        return output
    else:
        result = []
        seq_len = cam_interval * 2 + 1

        for iv in range(0, len(cam_views)-2, 2):
            result.append(cam_views[iv])
            interpolations = interpolate3toN(torchModel, cam_views[iv], cam_views[iv+1], cam_views[iv+2], seq_len)
            for idx, interpolation in enumerate(interpolations):
                result.append(interpolation)
                if idx == cam_interval - 2:
                    result.append(cam_views[iv+1])
        result.append(cam_views[-1])
        return result


# run a parallax view generation on the images folder with all settings as defined by the config
def run_parallax_view_generation(save_images=True):
    return run_parallax_view_generation0(model, config.PARALLAX_VIEW_CAM_INTERVAL, config.PARALLAX_DATASET_DIR, config.PARALLAX_OUTPUT_DIR, 
                                        config.NET_MODE, numImages=config.PARALLAX_VIEW_T,save_images=save_images)

# parallax view generation, but can take parameters instead of using the config
# may save the images and a psnr.json file that lists the psnr for every generated image
def run_parallax_view_generation0(torchModel, t, inputDir, outputDir, netmode, numImages = -1, save_images=True):
    cam_interval = t
    t = numImages
    parallax_output_dir = outputDir

    if save_images and parallax_output_dir != None:
        im_output = os.path.join(parallax_output_dir, "images")
        json_output = os.path.join(parallax_output_dir, "psnr.json")
        makedirs(im_output, exist_ok = True)

    images = load_images(inputDir)
    
    if t == -1:
        t = len(images)

    input_images = []
    for w in range(0, t, cam_interval):
        input_images.append(images[w])

    parallax_view = generate_parallax_view(torchModel, t, cam_interval, input_images, netmode)

    worstPsnr = 999999999

    resultsDict = {}

    for index, view in enumerate(parallax_view):
        p = 0
        if index % cam_interval != 0:
            p = psnr(pil_to_tensor(view), pil_to_tensor(images[index])).item()
            if p < worstPsnr:
                worstPsnr = p
        resultsDict[index] = p

        if save_images and parallax_output_dir != None:
            view.save(join_paths(im_output, '{}.jpg'.format(index+1)), 'JPEG', quality=95)
            writeJson(json_output, resultsDict)

    return worstPsnr

def visual_test(epoch):
    print("===> Running visual test...")
    for i, tup in enumerate(visual_test_set):
        result = interpolate(model, load_img(tup[0]), load_img(tup[2]))
        result = pil_to_tensor(result)
        tag = 'data/visual_test_{}'.format(i)
        board_writer.add_image(tag, result, epoch)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    tick_t = timer()

    if config.GENERATE_PARALLAX_VIEW:
        if config.EVALUATION_DIR is not None:
            print(f"Evaluating models in {config.EVALUATION_DIR}")
            model_files = [join_paths(config.EVALUATION_DIR, x) for x in listdir(config.EVALUATION_DIR)]
            for m in model_files:
                model = init_model(m, quiet=True)
                print(f"PSNR of {m} is {run_parallax_view_generation(save_images=False)}")
        else:
            print("PSNR is ", run_parallax_view_generation())
    else:
        for epoch in range(config.START_FROM_EPOCH, config.EPOCHS + 1):
            train(epoch)
            if config.SAVE_CHECKPOINS:
                save_checkpoint(epoch)
            if config.VALIDATION_ENABLED:
                validate(epoch)
            if config.PARALLAX_VALIDATION:
                _psnr = run_parallax_view_generation(save_images=False)
                board_writer.add_scalar('data/epoch_validation_psnr', _psnr, epoch)
                print("===> Validation PSNR: {:.3f}".format(_psnr))
            if config.VISUAL_TEST_ENABLED:
                visual_test(epoch)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

    board_writer.close()
