
# import os
# import torch
# import wandb
# import torch.nn as nn
# from torch.autograd import Variable
# from tqdm import tqdm
# from utils_multi.result_utilss import *
# # from utils_multi.result_utils import *
# import torch.nn.functional as F
# from omegaconf import OmegaConf

# class Trainer:

#     def __init__(self, model, data_train, data_valid, data_test, data_test_video, args, device):
#         self.model = model
#         self.data_train = data_train
#         self.data_valid = data_valid
#         self.data_test = data_test
#         self.data_test_video = data_test_video
#         self.args = args
#         self.device = device
#         self.best_mpjpe = float('inf')  
#         self.best_epoch = 0
#         self.best_ap50=-1.0
#         self.best_ap50_epoch = 0
#         os.makedirs(self.args.result.save_path, exist_ok=True)
#         self.best_model_path = os.path.join(
#             self.args.result.save_path, 
#             f"{self.args.result.name}_best.pth"
#         )
#     def train(self):
#         self.model = self.model.to(self.device)
#         loss_fn = nn.MSELoss().to(self.device)
#         loss_fn_leg = nn.MSELoss().to(self.device)
#         loss_fn_hand = nn.MSELoss().to(self.device)
#         optimizer = torch.optim.AdamW(self.model.parameters(), 
#                                         lr=self.args.train.learning_rate, weight_decay=self.args.train.weight_decay)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epoch)
     
#         Epoch_num = self.args.train.epoch
#         step = 0
#         test_loss_best = 100
#         score_history = []
#         for epoch in range(Epoch_num):
#             self.model.train()
#             progress_bar = tqdm(self.data_train)
#             for iter, (x_batch, x_R_batch, y_batch,_) in enumerate(progress_bar):
#                 x_batch = Variable(x_batch.float().to(self.device))
#                 x_R_batch = Variable(x_R_batch.float().to(self.device))
#                 y_batch = Variable(y_batch.float().to(self.device))    
            
#                 y_batch_pred = self.model(x_batch, x_R_batch)
                
#                 # loss_coord = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
#                 # loss_leg = loss_fn_leg(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:])
#                 # loss_hand = loss_fn_hand(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:])
#                 # loss_motion_leg = motion_cal(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:], intervals=[2,4,6,8])
#                 # loss_motion_hand = motion_cal(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:], intervals=[2,4,6,8])
#                 # loss = loss_coord + (loss_leg+loss_hand)*self.args.train.alpha_limb + (loss_motion_leg+loss_motion_hand)*self.args.train.alpha_limb_motion
#                 loss_coord = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
#                 loss_leg = loss_fn_leg(y_batch_pred[:,:,(6,7,9,10),:], y_batch[:,:,(6,7,9,10),:])  # 髋、膝
#                 loss_hand = loss_fn_hand(y_batch_pred[:,:,(2,3,12,13),:], y_batch[:,:,(2,3,12,13),:])  # 肩、肘、手腕
#                 loss_motion_leg = motion_cal(y_batch_pred[:,:,(6,7,9,10),:], y_batch[:,:,(6,7,9,10),:], intervals=[2,4,6,8])
#                 loss_motion_hand = motion_cal(y_batch_pred[:,:,(2,3,12,13),:], y_batch[:,:,(2,3,12,13),:], intervals=[2,4,6,8])
#                 loss = loss_coord + (loss_leg+loss_hand)*self.args.train.alpha_limb + (loss_motion_leg+loss_motion_hand)*self.args.train.alpha_limb_motion

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 # print
#                 step += 1
#                 progress_bar.set_description(
#                 'Step: {}. Epoch: {}/{}. Total loss: {:.3f}. Coord Loss: {:.3f}. Motion Loss_Leg: {:.3f}. Motion Loss_Hand: {:.3f}'.
#                 format(step, epoch+1, Epoch_num, loss.item(), loss_coord.item(), loss_motion_leg.item()*0.05, loss_motion_hand.item()*0.05))

#             test_loss, des_test = test_keypoint(self.data_test, self.device, self.model, output_temporal=True)
#             # test_loss = test_keypoint_600(self.data_test, self.device, self.model, self.args.result.annot_test)
#             current_mpjpe = test_loss['MPJPE'].mean()
#             score_history.append(current_mpjpe)
#             print('test_MPJPE: {:.5f}. test_PCC: {:.3f}. test_PCK: {:.3f}%'.
#                 format(test_loss['MPJPE'].mean(),test_loss['PCC'].mean(),test_loss['PCK'].mean() * 100))
            
#             # ap_dict = test_ap_600(self.data_test, self.device, self.model, self.args.result.annot_test)
#             print('AP50: {:.3f}'.format(test_loss['AP50']))
#             print('AP75: {:.3f}'.format(test_loss['AP75']))
#             print('mAP:  {:.3f}'.format(test_loss['AP']))    
#             current_ap50 = float(test_loss['AP50'])

#             if current_ap50 > self.best_ap50:
#                 self.best_ap50 = current_ap50
#                 self.best_ap50_epoch = epoch + 1

#             print(f"[Best AP50] epoch={self.best_ap50_epoch}/{Epoch_num}, AP50={self.best_ap50:.3f}")
#             if current_mpjpe < self.best_mpjpe:
#                 self.best_mpjpe = current_mpjpe
#                 self.best_epoch = epoch + 1  # 因为epoch从0开始
                
#                 # 保存最佳模型
#                 torch.save({
#                     'epoch': self.best_epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_mpjpe': self.best_mpjpe,
#                     'args': self.args
#                 }, self.best_model_path)  
#             if self.args.wandb.use_wandb:
#                 wandb.log({
#                     'lr': lr_scheduler.optimizer.param_groups[0]['lr'],
#                     'test_MPJPE': test_loss['MPJPE'].mean(),
#                     'test_MPJPE_leg': test_loss['MPJPE'][:,:,:,(2,3,5,6)].mean(), 
#                     'test_MPJPE_hand': test_loss['MPJPE'][:,:,:,(12,13,15,16)].mean(),
#                     'test_PCC': test_loss['PCC'][:,:,:,1:].mean(), 
#                     'test_PCC_leg': test_loss['PCC'][:,:,:,(2,3,5,6)].mean(), 
#                     'test_PCC_hand': test_loss['PCC'][:,:,:,(12,13,15,16)].mean(),
#                     'test_PCK': test_loss['PCK'].mean()*100, 
#                     'test_PCK_leg': test_loss['PCK'][:,:,:,(2,3,5,6)].mean()*100, 
#                     'test_PCK_hand': test_loss['PCK'][:,:,:,(12,13,15,16)].mean()*100,  
#                     })

#             lr_scheduler.step()
#         print(f"\n训练完成! 最佳模型在第 {self.best_epoch} 轮, 最佳MPJPE: {self.best_mpjpe:.3f}")
#         print(f"最佳模型保存路径: {self.best_model_path}")
"""Training process
"""  
import os
import torch
import wandb
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from utils_multi.result_utils import *

from omegaconf import OmegaConf

class Trainer:

    def __init__(self, model, data_train, data_valid, data_test, data_test_video, args, device):
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.data_test_video = data_test_video
        self.args = args
        self.device = device
        self.best_mpjpe = float('inf')
        self.best_epoch = 0
        os.makedirs(self.args.result.save_path, exist_ok=True)
        self.best_model_path = os.path.join(
            self.args.result.save_path,
            f"{self.args.result.name}_best.pth"
        )
    def train(self):
        self.model = self.model.to(self.device)
        loss_fn = nn.MSELoss().to(self.device)
        loss_fn_leg = nn.MSELoss().to(self.device)
        loss_fn_hand = nn.MSELoss().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                        lr=self.args.train.learning_rate, weight_decay=self.args.train.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epoch)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                T_0 = 25,# Number of iterations for the first restart
                                                T_mult = 1, # A factor increases TiTi​ after a restart
                                                eta_min = 0.01*self.args.train.learning_rate) # Minimum learning rate

        Epoch_num = self.args.train.epoch
        step = 0
        test_loss_best = 100
        for epoch in range(Epoch_num):
            self.model.train()
            progress_bar = tqdm.tqdm(self.data_train)
            for iter, (x_batch, x_R_batch, y_batch) in enumerate(progress_bar):
                x_batch = Variable(x_batch.float().to(self.device))
                x_R_batch = Variable(x_R_batch.float().to(self.device))
                y_batch = Variable(y_batch.float().to(self.device))    
            
                y_batch_pred = self.model(x_batch, x_R_batch)
                loss_coord = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
                loss_leg = loss_fn_leg(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:])
                loss_hand = loss_fn_hand(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:])
                loss_motion_leg = motion_cal(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:], intervals=[2,4,6,8])
                loss_motion_hand = motion_cal(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:], intervals=[2,4,6,8])
                # loss = loss_coord
                loss = loss_coord + (loss_leg+loss_hand)*self.args.train.alpha_limb + (loss_motion_leg+loss_motion_hand)*self.args.train.alpha_limb_motion

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print
                step += 1
                progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Total loss: {:.3f}. Coord Loss: {:.3f}. Motion Loss_Leg: {:.3f}. Motion Loss_Hand: {:.3f}'.
                format(step, epoch+1, Epoch_num, loss.item(), loss_coord.item(), loss_motion_leg.item()*0.05, loss_motion_hand.item()*0.05))

            test_loss, des_test = test_keypoint(self.data_test, self.device, self.model, output_temporal=True)
            current_mpjpe = test_loss['MPJPE'].mean()
            print('test_MPJPE: {:.5f}. test_PCC: {:.5f}. test_PCK: {:.3f}%'.
                format(test_loss['MPJPE'].mean(), test_loss['PCC'][:,1:].mean(), test_loss['PCK'].mean()*100))
            if current_mpjpe < self.best_mpjpe:
                self.best_mpjpe = current_mpjpe
                self.best_epoch = epoch + 1
                torch.save({
                    'epoch': self.best_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mpjpe': self.best_mpjpe,
                    'args': OmegaConf.to_container(self.args, resolve=True)
                }, self.best_model_path)
                print(f"[Saved] New best MPJPE: {self.best_mpjpe:.3f} at epoch {self.best_epoch}")
            if self.args.wandb.use_wandb:
                wandb.log({
                    'lr': lr_scheduler.optimizer.param_groups[0]['lr'],
                    'test_MPJPE': test_loss['MPJPE'].mean(),
                    'test_MPJPE_leg': test_loss['MPJPE'][:,:,:,(2,3,5,6)].mean(), 
                    'test_MPJPE_hand': test_loss['MPJPE'][:,:,:,(12,13,15,16)].mean(),
                    'test_PCC': test_loss['PCC'][:,:,:,1:].mean(), 
                    'test_PCC_leg': test_loss['PCC'][:,:,:,(2,3,5,6)].mean(), 
                    'test_PCC_hand': test_loss['PCC'][:,:,:,(12,13,15,16)].mean(),
                    'test_PCK': test_loss['PCK'].mean()*100, 
                    'test_PCK_leg': test_loss['PCK'][:,:,:,(2,3,5,6)].mean()*100, 
                    'test_PCK_hand': test_loss['PCK'][:,:,:,(12,13,15,16)].mean()*100,  
                    })

            lr_scheduler.step()
