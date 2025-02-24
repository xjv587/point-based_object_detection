from cycler import V
import torch

from .models import Detector, FocalLoss, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset, PR, point_close
from . import dense_transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path


DENSE_CLASS_DISTRIBUTION = [0.02929112, 0.0044616, 0.00411153]
weight = torch.tensor(DENSE_CLASS_DISTRIBUTION, dtype=torch.float32)
weight = 1 / weight
weight = weight / weight.sum()
pos_weight = torch.tensor([25,70,5]).view(3,1,1)

def train(args):
    model = Detector()
    #train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    writer = SummaryWriter('valid_log')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    #criterion = FocalLoss(weight=weight)
    train_dataset = DetectionSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose([   \
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness=0.5, contrast=0.9, saturation=0.9, hue=0.3), 
        dense_transforms.ToTensor()]))
    valid_dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]

        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
            image, *dets = data
            heatmap, size = dense_transforms.detections_to_heatmap(dets, image.shape[1:])
            hm_pred = model.forward(image).squeeze(0)
            loss = criterion(hm_pred, heatmap)
            pred_det = model.detect(image)
            for j, det in enumerate(dets):
                pr_box[j].add(pred_det[j], det)
                pr_dist[j].add(pred_det[j], det)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.3f}. AP: {pr_box[0].average_prec:.3f}, {pr_box[1].average_prec:.3f}, {pr_box[2].average_prec:.3f}.")
               
        model.eval()
        v_loss = 0.0
        vpr_box = [PR() for _ in range(3)]
        vpr_dist = [PR(is_close=point_close) for _ in range(3)]
        for k, vdata in enumerate(valid_dataset):
            vimage, *vdets = vdata
            vhm_pred = model.forward(vimage).squeeze(0)
            vhm_truth, size = dense_transforms.detections_to_heatmap(vdets, vimage.shape[1:])
            vloss = criterion(vhm_pred, vhm_truth)
            v_loss += vloss
            vpred_det = model.detect(vimage)
            for l, vdet in enumerate(vdets):
                vpr_box[l].add(vpred_det[l], vdet)
                vpr_dist[l].add(vpred_det[l], vdet)
            
            if k % 100 == 0:
                global_step = epoch * len(valid_dataset) + k * 100
                vimage = vimage.unsqueeze(0)
                vhm_truth = vhm_truth.unsqueeze(0)
                vhm_pred = vhm_pred.unsqueeze(0)
                log(logger=writer, imgs=vimage, gt_det= vhm_truth, det=vhm_pred, global_step=global_step)

        avg_valid_loss = v_loss / len(valid_dataset)
        print('Valid Loss: %0.3f' % avg_valid_loss)        
        print(f"Kart box_ap: {vpr_box[0].average_prec:.3f}, bomb box_ap: {vpr_box[1].average_prec:.3f}, pickup box_ap: {vpr_box[2].average_prec:.3f}")        
        print(f"Kart dist_ap: {vpr_dist[0].average_prec:.3f}, bomb dist_ap: {vpr_dist[1].average_prec:.3f}, pickup dist_ap: {vpr_dist[2].average_prec:.3f}")

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs, global_step)
    logger.add_images('label', gt_det, global_step)
    logger.add_images('pred', torch.sigmoid(det), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='')
    parser.add_argument('--num_epochs', type=int, default=60)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
