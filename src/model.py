import os
import numpy as np

from scipy.ndimage.filters import maximum_filter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .horizon_net import HorizonNet

from .network import SIGenerator, SIDiscriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, LayoutSimilarityLoss_Horizon


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + f'_gen_{self.iteration}.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + f'_dis_{self.iteration}.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path.replace('.pth', f'_{self.iteration}.pth'))

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path.replace('.pth', f'_{self.iteration}.pth'))



class SIInpaintingModel(BaseModel):
    def __init__(self, config):
        super(SIInpaintingModel, self).__init__('InpaintingModel', config)
        torch.autograd.set_detect_anomaly(True)
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = SIGenerator(self.config)
        discriminator = SIDiscriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        # layout_similarity_loss = LayoutSimilarityLoss_Horizon('./pretrained_horizon_net/resnet50_rnn__st3d.pth')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        # self.add_module('layout_similarity_loss', layout_similarity_loss)

        self.horizon_net = self.load_trained_model(HorizonNet, './pretrained_horizon_net/resnet50_rnn__st3d.pth').to('cuda')

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )


    def process(self, images, masks, layout, emptys):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs, seg, seg_p, layout_guidence = self(images, masks, layout, emptys)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = emptys
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)                    # in: [rgb(3)+edge(1)]
        dis_fake, gen_fake_feat = self.discriminator(dis_input_fake)                    # in: [rgb(3)+edge(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        # gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_l1_loss_fore = self.l1_loss_f_b(emptys, outputs, masks, 'foreground')
        gen_l1_loss_back = self.l1_loss_f_b(emptys, outputs, masks, 'background')
        gen_l1_loss = gen_l1_loss_fore + gen_l1_loss_back
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, emptys)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, emptys * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # layout consistence loss
        # images_1024_512 = F.interpolate(images, size=[512, 1024], mode='nearest')
        # outputs_1024_512 = F.interpolate(outputs, size=[512, 1024], mode='nearest')

        # gen_layout_loss = 0
        # gen_loss_bon, gen_loss_cor, y_bon_pos, y_cor_pos, y_bon_neg, y_cor_neg = self.layout_similarity_loss(images_1024_512, outputs_1024_512)
        # gen_layout_loss = 0.5 * gen_loss_bon + 0.5 * gen_loss_cor
        # gen_loss += gen_layout_loss

        # create logs
        logs = [
            ("l_d", dis_loss.item()),
            ("l_g", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]
        
        if self.iteration % self.config.LOG_INTERVAL == 0:
            # loss logs
            self.losses = {}
            self.losses['l1_loss'] = gen_l1_loss
            #self.losses['fm_loss'] = gen_fm_loss
            self.losses['perceptual_loss'] = gen_content_loss
            self.losses['style_loss'] = gen_style_loss

            self.losses['gen_total_loss'] = gen_loss
            self.losses['dis/real_loss'] = dis_real_loss
            self.losses['dis/fake_loss'] = dis_fake_loss
            self.losses['adv/gen_loss'] = gen_gan_loss
            self.losses['adv/dis_loss'] = dis_loss

            # visualization
            self.viz = {}
            images_with_masks = images * (1. - masks)
            images_merge_outputs = outputs * masks + images * (1. - masks)
            # layout_one_hot_v = self.VizWithClasses(seg)
            # layout_v = torch.cat((layout, layout, layout), dim=1)
            layout_guidence_v = torch.cat((layout_guidence, layout_guidence, layout_guidence), dim=1)
            seg_p_viz = self.seg_p_visualization(seg_p)
            self.viz['IO_image'] = torch.cat([images, images_with_masks, images_merge_outputs, emptys, outputs, seg, seg_p_viz, layout_guidence_v], dim=3)


        return outputs, gen_loss, dis_loss, logs

    def l1_loss_f_b(self, image, predict, mask, type='foreground'):
        error = torch.abs(predict - image)
        if type == 'foreground':
            loss = torch.sum(mask * error) / torch.sum(mask)    # * tf.reduce_sum(1. - mask) for balance?
        elif type == 'background':
            loss = torch.sum((1. - mask) * error) / torch.sum(1. - mask)
        else:
            loss = torch.sum(mask * torch.abs(predict - image)) / torch.sum(mask)
        return loss


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



    def VizWithClasses(self, seg):
        # seg: [b, c=3, h, w]
        b, c, h, w = seg.size()
        viz = torch.zeros((b, c, h, w)).cuda()

        return viz

    ### get 3-class & plane-wise instance map
    def layout_p_map(self, x, y_bon, y_cor, type='one'):
        b, c, h, w = x.size()
        input_size = self.config.INPUT_SIZE
        h_o = input_size[0]
        w_o = input_size[1]
        layout_p = torch.zeros(b, 1, h_o, w_o).cuda()

        y_cor = torch.nn.functional.interpolate(y_cor, size=w_o, mode='nearest')
        y_bon = torch.nn.functional.interpolate(y_bon, size=w_o, mode='nearest')



        if type=='one':
            y_bon = torch.clamp(((y_bon / np.pi + 0.5) * h_o).round().type(torch.int64), 0, h_o-1)

            for i in range(b):
                layout_p[i, 0, y_bon[i, 0], torch.arange(w_o)] = 1
                layout_p[i, 0, y_bon[i, 1], torch.arange(w_o)] = 1
        else:
            pass

        # layout_p = F.interpolate(layout_p, size=[256, 256], mode='bilinear')

        raw_id = self.find_N_peaks_pt(y_cor) # y_cor[b, 1, w], raw_id[b, n]
        lyt_seg = self.layout_seg(layout_p) # lyt_seg[b, 3, h, w]

        ### 3-class one-hot map & plane-wise one-hot map
        segs = []
        seg_ps = []
        for i in range(b):
            seg, seg_p = self.one_hot(lyt_seg[i], 3, raw_id[i]) # seg[b, 3, h, w], seg_p[b, n, h, w] n => plane nums
            segs.append(seg)
            seg_ps.append(seg_p)
          
        segs = torch.stack(segs)
        seg_ps = torch.stack(seg_ps)

        return segs, seg_ps, layout_p

    ### find corner location of HorizonNet output layout
    def find_N_peaks_pt(self, signal, r=29, min_v=0.05, N=None):
        b, c, w = signal.size()
        r = (r * w) // 1024
        nice_peaks_list = []
        for i in range(b):
            a = signal[i][0]
            window_maxima = torch.nn.functional.max_pool1d_with_indices(a.view(1, 1, -1), r, 1, padding=r // 2)[1].squeeze()
            candidates = window_maxima.unique()
            nice_peaks = candidates[(window_maxima[candidates] == candidates).nonzero()]
            nice_peaks = nice_peaks[a[nice_peaks] > min_v]
            nice_peaks_list.append(nice_peaks)
        if nice_peaks_list != None:
            return nice_peaks_list
        else:
            nice_peaks_list.append(torch.empty(1).cuda())
            return nice_peaks_list

    ### 3-class segmentation map
    def layout_seg(self, lyt):
        b, c, h, w = lyt.size()
        semantic_masks = []
        for i in range(b):
            lyt_per = lyt[i][0]
            top_bottom = lyt_per.cumsum(dim=0)>0
            bottom_up = 2 * (torch.flipud(torch.flipud(lyt_per).cumsum(dim=0) > 0))
            semantic_mask = top_bottom + bottom_up - 1
            semantic_masks.append(semantic_mask.unsqueeze(0))
        semantic = torch.stack(semantic_masks)
        return semantic

    def one_hot(self, labels, C, raw_id):
        '''
            Converts an integer label torch.autograd.Variable to a one-hot Variable.

            Parameters
            ----------
            labels : torch.autograd.Variable of torch.cuda.LongTensor
                N x 1 x H x W, where N is batch size.
                Each value is an integer representing correct classification.
            C : integer.
                number of classes in labels.

            Returns
            -------
            target : torch.autograd.Variable of torch.cuda.FloatTensor
                N x C x H x W, where C is class number. One-hot encoded.
            '''
        # one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        c, h, w = labels.size()

        label = labels
        one_hot = torch.FloatTensor(C, h, w).zero_().cuda()
        target = one_hot.scatter_(0, label.long(), 1)
            
            
        #rd = torch.cat((torch.cuda.LongTensor([0]), raw_id, torch.cuda.LongTensor([w])), dim=-1)                      
        wall_segs = self.seg_by_cor(target, raw_id)

        target_planes = torch.cat((target[0].unsqueeze(0), target[1].unsqueeze(0), wall_segs), dim=0)

        pad = 100
        c, h, w = target_planes.size()
        padding = torch.zeros(pad - c, h, w).cuda()
        target_planes = torch.cat((target_planes, padding), 0)


        return target, target_planes

    ### wall -> planes by corners
    def seg_by_cor(self, target, raw_id):
        clip_idx = []
        c, h, w = target

        raw_id = torch.cat((torch.cuda.LongTensor([0]), raw_id, torch.cuda.LongTensor([self.config.INPUT_SIZE[1]])), dim=-1)

        for i in range(len(raw_id) - 1):
            tmp = i
            if raw_id[i + 1] > raw_id[tmp]:
                clip_idx.append([raw_id[tmp], raw_id[i + 1]])
            else:
                while raw_id[i + 1] < raw_id[tmp]:
                    i = i + 1
                clip_idx.append([raw_id[tmp], raw_id[i + 1]])

        wall_segs = self.wall_seg_by_clip_index(target, 2, clip_idx)

        return wall_segs

    def wall_seg_by_clip_index(self, target, wall_ch_id, clip_idx):
        wall_segs_list = []
        for i in range(len(clip_idx)):
            new_wall_seg = torch.FloatTensor(1, target.size(1), target.size(2)).zero_().cuda()
            new_wall_seg_1 = torch.FloatTensor(1, target.size(1), target.size(2)).zero_().cuda()
            new_wall_seg[:, :, int(clip_idx[i][0]):int(clip_idx[i][1])] = 1
            new_wall_seg = new_wall_seg + target[wall_ch_id]
            # new_wall_seg[new_wall_seg != 2] = 0
            new_wall_seg_1[new_wall_seg == 2] = 1
            wall_segs_list.append(new_wall_seg_1)

        wall_segs = torch.stack(wall_segs_list, dim=1).squeeze(0)
        wall_segs[0] = wall_segs[0] + wall_segs[-1]

        return wall_segs[:-1, :, :]

    ### visualization
    def visualize_layout(self, x, y_bon, y_cor):
        c, h, w = x.size()
        x = x.float()
        y_bon = torch.clamp(((y_bon / np.pi + 0.5) * h).round().type(torch.int64), 0, 511)

        gt_cor = torch.zeros((3, 30, 1024), dtype=torch.float).cuda()
        gt_cor[:] = y_cor.unsqueeze(1).repeat(3, 30, 1)
        img_pad = torch.zeros((3, 3, 1024), dtype=torch.float).cuda() + 1

        img_bon = (x * 0.5)

        img_bon[1, y_bon[0], torch.arange(1024)] = 1
        img_bon[1, y_bon[1], torch.arange(1024)] = 1

        return torch.cat([gt_cor, img_pad, img_bon], dim=1).unsqueeze(0)

    def load_trained_model(self, Net, path):
        state_dict = torch.load(path)
        net = Net(**state_dict['kwargs'])
        net.load_state_dict(state_dict['state_dict'])
        return net

    def forward(self, images, masks, layout, emptys):
        images_masked = (images * (1 - masks)) + masks
        # inputs = torch.cat((images_masked, edges_masked, masks, layouts), dim=1)

        ### HorizonNet only support 1024*512 resolution
        images_masked_1024_512 = F.interpolate(images_masked, size=[512, 1024], mode='nearest')
        
        if self.config.MODE == 2 or self.config.LAYOUT == 2:
            y_bon, y_cor = self.horizon_net(images_masked_1024_512)
        else:
            y_bon, y_cor = torch.split(layout, [2, 1], dim=1)

        ### get 3-class & plane-wise instance map
        seg, seg_p, layout_guidence = self.layout_p_map(images_masked_1024_512, y_bon, y_cor, type='one')

        if self.config.PLANE == 1:
            # inputs = torch.cat((images_masked, masks), dim=1)
            outputs = self.generator(images_masked, masks, seg, seg_p, layout_guidence)                                    # in: [rgb(3) + edge(1)]
        else:
            outputs = self.generator(images_masked, masks, seg, seg, layout_guidence)                                    # in: [rgb(3) + edge(1)]
        
        return outputs, seg, seg_p, layout_guidence

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()
