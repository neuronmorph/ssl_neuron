import os
import torch
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch, pos_enc_from_eigvec_of_conductive_dist_matrix, compute_eigvec_of_euclidean_dist_matrix
import time
# from torch.profiler import profile, record_function, ProfilerActivity

class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)
        
        
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        

    def train(self):     
        self.curr_iter = 0
        epoch = 0
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)
            
            epoch += 1


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        epoch_time = time.time()
        # print('Starting batch')
        # prof = torch.profiler.profile(
        #             activities=[
        # ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
        # #on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/mnist'),
        #         record_shapes=True
        # #profile_memory=True,
        # #with_stack=True,
        # #with_flops=True,
        # #with_modules=True
        # )    
        # prof.start()
        for i, data in enumerate(self.train_loader, 0):
            f1_cpu, f2_cpu, a1_cpu, a2_cpu = [x.float() for x in data]
            f1, f2, a1, a2 = f1_cpu.to(self.device, non_blocking=True), f2_cpu.to(self.device, non_blocking=True), a1_cpu.to(self.device, non_blocking=True), a2_cpu.to(self.device, non_blocking=True),
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            #l1 = compute_eig_lapl_torch_batch(a1)
            #l2 = compute_eig_lapl_torch_batch(a2)
            

            #cd1 = pos_enc_from_eigvec_of_conductive_dist_matrix(f1_cpu, a1_cpu, self.device)
            #cd2 = pos_enc_from_eigvec_of_conductive_dist_matrix(f2_cpu, a2_cpu, self.device)

            ed1 = compute_eigvec_of_euclidean_dist_matrix(f1)
            ed2 = compute_eigvec_of_euclidean_dist_matrix(f2)
            
            # l1 = torch.cat((l1, cd1), dim=2)
            # l2 = torch.cat((l2, cd2), dim=2)
            
            #l1 = torch.cat((l1, cd1, ed1), dim=2)
            #l2 = torch.cat((l2, cd2, ed2), dim=2)

            #l1 = torch.cat((l1, cd1, ed1), dim=0)
            #l2 = torch.cat((l2, cd2, ed2), dim=0)

            #l1 = cd1
            #l2 = cd2
            
            l1 = ed1
            l2 = ed2

            eig_val1, eig_vec1 = torch.linalg.eigh(l1)
            eig_val2, eig_vec2 = torch.linalg.eigh(l2)

            eig_vec1 = torch.flip(eig_vec1, dims=[2])
            eig_vec2 = torch.flip(eig_vec2, dims=[2])

            pos_enc_dim = 32
            pos_enc1 = eig_vec1[:, :, 1:pos_enc_dim + 1]
            pos_enc2 = eig_vec2[:, :, 1:pos_enc_dim + 1]

            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model(f1, f2, a1, a2, pos_enc1, pos_enc2)
            #loss = self.model(f1, f2, a1, a2, l1, l2)            
            #loss = self.model(f1, f2, a1, a2, cd1, cd2)
            #loss = self.model(f1, f2, a1, a2, l1, l2)

            # optimize 
            loss.sum().backward()
            self.optimizer.step()
            
            # update teacher weights
            self.model.update_moving_average()
            
            losses.update(loss.detach(), n)
            self.curr_iter += 1

        # prof.step()
        # prof.stop()
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f'Epoch {epoch} | in {(time.time() - epoch_time):.1f}s| Loss {losses.avg:.4f}')


    def _save_checkpoint(self, epoch):
        filename = 'ckpt_{}.pt'.format(epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))

        