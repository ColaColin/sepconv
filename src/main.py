#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from os.path import exists, join as join_paths
from os import makedirs
from timeit import default_timer as timer
import src.config as config
from src.data_manager import load_img
from src.model import Net, CustomLoss
from src.dataset import get_training_set, get_validation_set, get_visual_test_set, pil_to_tensor
from src.interpolate import interpolate

# ----------------------------------------------------------------------

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

print('===> Loading datasets...')
train_set = get_training_set()
validation_set = get_validation_set()
visual_test_set = get_visual_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
                                  shuffle=True)
validation_data_loader = DataLoader(dataset=validation_set, num_workers=config.NUM_WORKERS,
                                    batch_size=config.BATCH_SIZE, shuffle=False)

print('===> Building model...')
model = Net().to(device)
if config.START_FROM_EXISTING_MODEL is not None:
    print(f'===> Loading pre-trained model: {config.START_FROM_EXISTING_MODEL}')
    state_dict = torch.load(config.START_FROM_EXISTING_MODEL)
    model.load_state_dict(state_dict)
l1_loss = nn.L1Loss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)

board_writer = SummaryWriter()

# ----------------------------------------------------------------------

def train(epoch):
    print("===> Training...")
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        print('Forward pass...')
        output = model(input)

        loss = l1_loss(output, target)

        print('Computing gradients...')
        loss.backward()

        print('Gradients ready.')
        optimizer.step()

        loss_val = loss.item()
        epoch_loss += loss_val

        board_writer.add_scalar('data/iter_training_loss', loss_val, iteration)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_val))

    epoch_loss /= len(training_data_loader)
    board_writer.add_scalar('data/epoch_training_loss', epoch_loss, epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))


def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = join_paths(config.OUTPUT_DIR, model_out_path)
    if not exists(config.OUTPUT_DIR):
        makedirs(config.OUTPUT_DIR)
    torch.save(model.cpu().state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    if device.type != 'cpu':
        model.cuda()


def validate(epoch):
    print("===> Running validation...")
    valid_loss = 0
    with torch.no_grad():
        for batch in validation_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            loss = l1_loss(output, target)
            valid_loss += loss.item()
    valid_loss /= len(validation_data_loader)
    board_writer.add_scalar('data/epoch_validation_loss', valid_loss, epoch)
    print("===> Validation loss: {:.4f}".format(valid_loss))


def visual_test(epoch):
    print("===> Running visual test...")
    for i, tup in enumerate(visual_test_set):
        result = interpolate(model, load_img(tup[0]), load_img(tup[2]))
        result = pil_to_tensor(result)
        tag = 'data/visual_test_{}'.format(i)
        board_writer.add_image(tag, result, epoch)


# ----------------------------------------------------------------------

tick_t = timer()

for epoch in range(1, config.EPOCHS + 1):
    train(epoch)
    if config.SAVE_CHECKPOINS:
        save_checkpoint(epoch)
    if config.VALIDATION_ENABLED:
        validate(epoch)
    if config.VISUAL_TEST_ENABLED:
        visual_test(epoch)

tock_t = timer()

print("Done. Took ~{}s".format(round(tock_t - tick_t)))

board_writer.close()
