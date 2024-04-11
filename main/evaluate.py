'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-19 18:43:58
 # @ Description:
 '''

import os
import torch
import clip
from tqdm import tqdm
import os.path as osp
import torch.nn as nn
from torch.autograd import Variable
from utils.misc import correct, clip_encode_noised_text
from decimal import Decimal


def round_2(num):
    return Decimal(num).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")

def eval_zero_shot(clip_model, network, dataloader, prompts, device,
                    cls_nums, adver=None, eval_clip=False):
    sim_metric = torch.nn.CosineSimilarity(dim=-1)

    if not eval_clip:
        network.eval()
    total_correct_c = 0
    total_correct_cp = 0
    total_correct_pc = 0
    total_correct_nc = 0
    total_number = 0
    with torch.no_grad():
        # preparing text embeddings for zero-shot
        p_c = torch.cat([clip.tokenize(p) for p in prompts['C']]).to(device)
        p_cp = torch.cat([clip.tokenize(p) for p in prompts['CP']]).to(device)
        p_pc = torch.cat([clip.tokenize(p) for p in prompts['PC']]).to(device)
        
        t_c = clip_model.encode_text(p_c).type(torch.float32)
        t_cp = clip_model.encode_text(p_cp).type(torch.float32)
        t_pc = clip_model.encode_text(p_pc).type(torch.float32)
        if not eval_clip:
            t_c = network(t_c)
            t_cp = network(t_cp)
            t_pc = network(t_pc)

        for x, y in dataloader:
            # noised prompt
            t_nc = clip_encode_noised_text(clip_model, prompts['C'], n_ctx=4)
            if not eval_clip:
                t_nc = network(t_nc)
            
            x = x.to(device)
            y = y.to(device)

            # No attack if None, or FGSM, PGD-20, CW-20
            if adver is not None:
                x = adversarial_attack(clip_model, network, t_c, images=x,
                                labels=y, num_cls=cls_nums, adver_type=adver, eval_clip=eval_clip)
                
            f_x = clip_model.encode_image(x).type(torch.float32)
            if not eval_clip:
                f_x = network(f_x)

            # <[bs, 1, dim], [1, cls, dim]> -> [bs, cls]
            pred_scores_c = sim_metric(f_x.unsqueeze(-2), t_c.unsqueeze(-3))    
            pred_scores_cp = sim_metric(f_x.unsqueeze(-2), t_cp.unsqueeze(-3))
            pred_scores_pc = sim_metric(f_x.unsqueeze(-2), t_pc.unsqueeze(-3))
            pred_scores_nc = sim_metric(f_x.unsqueeze(-2), t_nc.unsqueeze(-3))
            
            total_number += y.size(0)

            total_correct_c += correct(pred_scores_c, y).item()
            total_correct_cp += correct(pred_scores_cp, y).item()
            total_correct_pc += correct(pred_scores_pc, y).item()
            total_correct_nc += correct(pred_scores_nc, y).item()

        acc_c = round_2(100.0 * total_correct_c / total_number)
        acc_cp = round_2(100.0 * total_correct_cp / total_number)
        acc_pc = round_2(100.0 * total_correct_pc / total_number)
        acc_nc = round_2(100.0 * total_correct_nc / total_number)
        return f"[c->{acc_c}, cp->{acc_cp}, pc->{acc_pc}, nc->{acc_nc}]", \
            (acc_c, acc_cp, acc_pc, acc_nc)


def training_classifier(clip_model, network, classifier, save_path, dataloader, optimizer,\
                                        loss_func, device, epochs=50, eval_clip=False):
    if not osp.exists(save_path):
        os.mkdir(save_path)
    if not eval_clip:
        network.eval()
    classifier.train()
    
    best_loss = float('inf')
    log_file = open(osp.join(save_path, "linear_prob.log"), 'w')
    patience = 10
    stop_count = 0
    for _ in tqdm(range(epochs)):
        losses = []
        for data in dataloader:
            optimizer.zero_grad()

            prompt, label = data
            t = torch.cat([clip.tokenize(p) for p in prompt]).to(device)
            label = label.to(device)

            with torch.no_grad():
                feature = clip_model.encode_text(t).type(torch.float32).to(device)
                if not eval_clip:
                    feature = network(feature)
                # L2 normalization
                feature = torch.nn.functional.normalize(feature).type(torch.float32)
            
            preds = classifier(feature)
            loss = loss_func(preds, label)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        # comparing average losses and save the best checkpoint
        ave_loss = sum(losses) / len(losses)
        if ave_loss < best_loss- 0.001:
            best_loss = ave_loss
            torch.save(classifier.state_dict(), osp.join(save_path, "best.pth"))
        else:
            stop_count += 1
            if stop_count >= patience:
                break

    log_file.write("Finished training classifier.")
    print("Finished training classifier.")
    

def training_classifier_with_img(clip_model, network, classifier, save_path, dataloader, optimizer,\
                                        loss_func, device, epochs=50, eval_clip=False):
    if not osp.exists(save_path):
        os.mkdir(save_path)
    if not eval_clip:
        network.eval()
    classifier.train()
    
    best_loss = float('inf')
    log_file = open(osp.join(save_path, "linear_prob.log"), 'w')
    patience = 10
    stop_count = 0
    for _ in tqdm(range(epochs)):
        losses = []
        for data in dataloader:
            optimizer.zero_grad()

            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)

            with torch.no_grad():
                feature = clip_model.encode_image(imgs).type(torch.float32).to(device)
                if not eval_clip:
                    feature = network(feature)
                # L2 normalization
                feature = torch.nn.functional.normalize(feature).type(torch.float32)
            
            preds = classifier(feature)
            loss = loss_func(preds, label)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        # comparing average losses and save the best checkpoint
        ave_loss = sum(losses) / len(losses)
        if ave_loss < best_loss- 0.001:
            best_loss = ave_loss
            torch.save(classifier.state_dict(), osp.join(save_path, "best.pth"))
        else:
            stop_count += 1
            if stop_count >= patience:
                break

    log_file.write("Finished training classifier.")
    print("Finished training classifier.")


def eval_linear_prob(clip_model, network, classifier, dataloader, device, cls_nums, adver=None,
                     eval_clip=False):
    total_correct = 0
    total_number = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # Not attack if None, or FGSM, PGD-20, CW-20
            x = adversarial_attack(clip_model, network, classifier, images=x,
                               labels=y, num_cls=cls_nums, adver_type=adver, eval_clip=eval_clip,
                               zero_shot=False)
            f_x = clip_model.encode_image(x).type(torch.float32)
            if not eval_clip:
                f_x = network(f_x)
            
            f_x  = torch.nn.functional.normalize(f_x)
            pred_scores = classifier(f_x)

            total_number += y.size(0)
            total_correct += correct(pred_scores, y).item()
        
        acc = 100.0 * total_correct / total_number
        return round_2(acc)


def pgd_attack(clip_model, network, text_feature, images, labels, step_size, k, eps, num_classes, loss_fn='ce',
                    eval_clip=False, zero_shot=True):
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    # 1. initialization
    if k > 1:
        x = images.detach() + torch.zeros_like(images, requires_grad=False).uniform_(-eps, eps).cuda()
        x = torch.clamp(x, 0.0, 1.0)
    else:
        x = images

    # 2. update adversarial examples
    for i in range(k):
        x.requires_grad_()
        if x.grad is not None:
            x.grad.data.fill_(0)

        with torch.enable_grad():
            z_adv = clip_model.encode_image(x).type(torch.float32)
            if not eval_clip:
                z_adv = network(z_adv)
            if zero_shot:
                z_adv = sim_metric(z_adv.unsqueeze(-2), text_feature.unsqueeze(-3))  
            else:
                z_adv = text_feature(z_adv)

            if loss_fn == 'ce':
                loss = nn.CrossEntropyLoss(reduce=False)(z_adv, labels).sum()
            elif loss_fn == 'cw':
                loss = cw_loss(z_adv, labels, num_classes=num_classes).sum()
            else:
                loss = None
            grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x.data + step_size * torch.sign(grad.data)
        x = Variable(torch.clamp(torch.min(torch.max(x_adv, images - eps), images + eps), 0.0, 1.0))
    return x

def cw_loss(output, target, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    return loss


def adversarial_attack(clip_model, network, text_feature,
                        images, labels, num_cls, adver_type="FGSM", eval_clip=False, zero_shot=True):
    if adver_type is None:
        return images

    if adver_type == "FGSM":
        num_steps = 1
        loss_fn = 'ce'
    elif adver_type == "PGD-20":
        num_steps = 20
        loss_fn = 'ce'
    elif adver_type == "CW-20":
        num_steps = 20
        loss_fn = 'cw'
    else:
        raise NotImplementedError

    epsilon = 0.031
    step_size = epsilon if num_steps == 1 else epsilon / 10.
    images = pgd_attack(clip_model, network, text_feature, images, labels, step_size=step_size,
                        k=num_steps, eps=epsilon, num_classes=num_cls, loss_fn=loss_fn, eval_clip=eval_clip, zero_shot=zero_shot)
    return images
