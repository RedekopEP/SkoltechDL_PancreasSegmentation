import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from dataloader import CellDataset, ComposeTransforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from modules import RichardsonLucy


def train_one(model, \
          n_iter, \
          num_epochs, \
          b_size=19):
    root = '../../../../raid/data/vpronina/Dataset_for_TRL/'
    psf_path = '../../../../raid/data/vpronina/Dataset_for_TRL/PSF/PSF2.tif'

    # root = '/media/valeriya/Elements/Project/learnable-richardson-lucy/Dataset_for_TRL/'
    # psf_path = '/media/valeriya/Elements/Project/learnable-richardson-lucy/Dataset_for_TRL/PSF/PSF2.tif'

    train_dataset = CellDataset(root, psf_path, train=True, transform=ComposeTransforms())
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=4)

    psf = Image.open(psf_path)
    psf = np.array(psf).astype(float)
    psf = torch.from_numpy(psf)
    psf = psf[None, None]
    psf = psf / psf.max()
    psf = psf / psf.sum()
    psf = psf.double().cuda()

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    model.cuda()

    loss_values = []
    for i in range(num_epochs):
        model.train(True)
        # for i_batch, (gt_batch, image_batch) in enumerate((tqdm(train_loader))):
        image_batch = train_dataset[0][1].reshape((b_size, 1, image_batch.shape[-1],
                                           image_batch.shape[-1])).double().cuda()

        gt_batch = train_dataset[0][0].reshape((b_size, 1, gt_batch.shape[-1],
                                     gt_batch.shape[-1])).double().cuda()
        output = model((image_batch, image_batch, psf, n_iter))
        loss = nn.functional.mse_loss(output, gt_batch)
        opt.zero_grad()
        loss.backward()
        loss_values.append(loss.item())
        opt.step()
        print(loss.item())
        print('Epoch: {}, training loss: {} '.format(i, np.array(loss_values).mean()))
    plt.plot(np.array(loss_values).mean())

    # Visualization
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
        a.axis('off')

    ax[0].imshow(gt_batch[0][0].cpu().numpy())
    ax[0].set_title('Original Data')

    ax[1].imshow(image_batch[0][0].cpu().numpy())
    ax[1].set_title('Blurred data')

    ax[2].imshow(output[0][0].detach().cpu().numpy(), vmin=image_batch.min(), vmax=image_batch.max())
    ax[2].set_title('Restored Data\nnum_epochs = {}'.format(num_epochs))

    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    # plt.show()
    fig.savefig('temp.png', dpi=fig.dpi)

    fig = plt.figure(figsize=(5,5))
    plt.gray()
    plt.imshow(output[0][0].detach().cpu().numpy())
    fig.savefig('res.png', dpi=fig.dpi)

    return gt_batch, image_batch, output


if __name__ == '__main__':
    train_one(RichardsonLucy(), 1, 1)