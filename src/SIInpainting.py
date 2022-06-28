import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from .dataset import Dataset
from .model import SIInpaintingModel
from .util import Progbar, create_dir, stitch_images, imsave
from .metric import PSNR, EdgeAccuracy

from tensorboardX import SummaryWriter

import time

class SIInpainting():
    def __init__(self, config):
        self.config = config
        os.environ['TORCH_HOME'] = './torch' #setting the environment variable
        model_name = 'SIInpainting'

        self.writer = SummaryWriter(os.path.join(config.PATH, 'logs', model_name))

        self.debug = False
        self.model_name = model_name
        self.model = SIInpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        config.TEST_LAYOUT_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         config.TRAIN_LAYOUT_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                        config.VAL_LAYOUT_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        torch.backends.cudnn.benchmark = True

        num_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                num_params += param.numel()
        print('Parameter numbers: ', num_params / 1e6, 'milions')

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            for items in train_loader:
                self.model.train()

                # images, edges, masks, layouts = self.cuda(*items)
                # outputs, gen_loss, dis_loss, logs = self.model.process(images, edges, masks, layouts)
                images, masks, layout, emptys = self.cuda(*items)
                outputs, gen_loss, dis_loss, logs = self.model.process(images, masks, layout, emptys)
                
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))

                # backward
                self.model.backward(gen_loss, dis_loss)
                iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs])


                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0 or iteration == 1:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(psnr)

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE//4,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        
        #precisions = []
        #recalls = []
        psnrs = []
        maes = []
        for items in val_loader:
            iteration += 1
            # images, edges, masks, layouts = self.cuda(*items)
            images, masks, layout, emptys = self.cuda(*items)

            # eval
            # outputs, out_edges = self.model(images, edges, masks, layouts)
            outputs, seg, seg_p, layout_guidence = self.model(images, masks, layout, emptys)

            # metrics
            #precision, recall = self.edgeacc(edges * masks, out_edges * masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = torch.mean(torch.abs(images - outputs_merged)).float()
            
            #precisions.append(precision.item())
            #recalls.append(recall.item())
            psnrs.append(psnr.item())
            maes.append(mae.item())

            logs = [("it", iteration), ]
            progbar.add(len(images), values=logs)
            
        #self.writer.add_scalar('eval/edge_similarity/precision', round(np.mean(precisions), 4), self.model.iteration)
        #self.writer.add_scalar('eval/edge_similarity/recall', round(np.mean(recalls), 4), self.model.iteration)
        self.writer.add_scalar('eval/texture_similarity/psnr', round(np.mean(psnrs), 4), self.model.iteration)
        self.writer.add_scalar('eval/texture_similarity/mae', round(np.mean(maes), 4), self.model.iteration)

    def test(self):
        self.model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        time_start = time.time()

        for items in test_loader:
            name = self.test_dataset.load_name(index)
            # images, edges, masks, layouts = self.cuda(*items)
            images, masks, layout, emptys = self.cuda(*items)

            index += 1


            # outputs, out_edges = self.model(images, edges, masks, layouts)
            outputs, seg, seg_p, layout_guidence = self.model(images, masks, layout, emptys)
            
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # output = self.postprocess(outputs_merged)[0]
            output_s = self.postprocess(outputs)[0]
            output = self.postprocess(outputs_merged)[0]
            # name = str(index).zfill(5) + '.png'
            path_1 = os.path.join(self.results_path, name)
            path_2 = os.path.join(self.results_path,  '2_'+name)
            
            print(index, name)
            
            imsave(output, path_1)
            # imsave(output_s, path_2)
            #imsave(self.postprocess(layout_guidence)[0],os.path.join(self.results_path,'layout_guidence', name))
            #imsave(self.postprocess(seg)[0],os.path.join(self.results_path,'seg', name))
            #seg_p_v = self.seg_p_visualization(seg_p)
            #imsave(self.postprocess(seg_p_v)[0],os.path.join(self.results_path,'seg_p', name))

            if self.debug:
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                masks = self.postprocess(masks)[0]
                fname, fext = name.split('.')

                imsave(masks, os.path.join(self.results_path, fname + '_mask.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print('\nEnd test....')
        
    def seg_p_visualization(self, seg_p):
        b, c, h, w =seg_p.size()
        seg_p_list = []
        for i in range(c):
            r_c = torch.zeros(b, 1, h, w).cuda()
            b_c = torch.zeros(b, 1, h, w).cuda()
            g_c = torch.zeros(b, 1, h, w).cuda()
            r_c = seg_p[:, i, :, :] * torch.LongTensor(1).random_(0, 255).cuda()
            b_c = seg_p[:, i, :, :] * torch.LongTensor(1).random_(0, 255).cuda()
            g_c = seg_p[:, i, :, :] * torch.LongTensor(1).random_(0, 255).cuda()
            rgb_seg_p = torch.cat((r_c.unsqueeze(1), b_c.unsqueeze(1), g_c.unsqueeze(1)), dim=1)
            seg_p_list.append(rgb_seg_p)
            
            
        seg_p_viz = torch.zeros(b, 3, h, w).cuda()
        for i in range(len(seg_p_list)):
          seg_p_viz += seg_p_list[i]

        return seg_p_viz

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.model.eval()

        items = next(self.sample_iterator)
        # images, edges, masks, layouts = self.cuda(*items)
        images, masks, layout, emptys = self.cuda(*items)

        iteration = self.model.iteration
        inputs = (images * (1 - masks)) + masks
        # outputs, out_edges = self.model(images, edges, masks, layouts)
        outputs, seg, seg_p, layout_guidence = self.model(images, masks, layout, emptys)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1


        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(seg),
            self.postprocess(layout_guidence),
            self.postprocess(layout),
            self.postprocess(emptys),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, psnr):
        iteration = self.model.iteration
        for loss_name in self.model.losses:
            self.writer.add_scalar(loss_name, self.model.losses[loss_name], iteration)
        for viz_name in self.model.viz:
            self.writer.add_image(viz_name, self.model.viz[viz_name][0], iteration)
        self.writer.add_scalar('texture_similarity/psnr', psnr.item(), self.model.iteration)

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255
        img = img.permute(0, 2, 3, 1)
        return img.int()
