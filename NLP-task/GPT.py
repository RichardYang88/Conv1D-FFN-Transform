from transformer import Encoder
from torch import nn,optim
from torch.nn.functional import cross_entropy,softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import torch
import utils
import os
import pickle
from utils import debug
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from typing import Tuple, Optional
from transformer import k_size, style

def live_plot(loss_list):
    clear_output(wait=True)
    plt.plot(loss_list)
    plt.show()

def print_model_parameters(model):
    print("Model layers and parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"Layer: {name:<60} | Shape: {str(param.shape):<20} | Parameters: {num_params:,}")
    print(f"Total Trainable Parameters: {total_params:,}\n")

class GPT(nn.Module):
    def __init__(self, model_dim, max_len, num_layer, num_head, n_vocab, lr, max_seg=3, drop_rate=0.3,padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len
        
        self.word_emb = nn.Embedding(n_vocab,model_dim)
        self.word_emb.weight.data.normal_(0,0.1) # [num_word, model_dim]

        self.segment_emb = nn.Embedding(num_embeddings= max_seg, embedding_dim=model_dim)
        self.segment_emb.weight.data.normal_(0,0.1)
        self.position_emb = torch.empty(1,max_len,model_dim)
        nn.init.kaiming_normal_(self.position_emb, mode='fan_out', nonlinearity='relu')
        self.position_emb = nn.Parameter(self.position_emb)
        
        self.encoder = Encoder(n_head=num_head, emb_dim=model_dim, drop_rate=drop_rate, n_layer=num_layer)
        self.task_mlm = nn.Linear(in_features=model_dim, out_features=n_vocab)
        self.task_nsp = nn.Linear(in_features=model_dim*self.max_len, out_features=2)

        self.opt = optim.AdamW(self.parameters(), lr, weight_decay = 5e-4, eps=5e-9)
    
    def forward(self,seqs, segs, training):
        embed = self.input_emb(seqs, segs)
        debug('self.mask(seqs):',self.mask(seqs).shape) # [n, 1, step, step]
        z = self.encoder(embed, training, mask = self.mask(seqs))   # [n, step, model_dim]
   
        mlm_logits = self.task_mlm(z)   # [n, step, n_vocab]
        nsp_logits = self.task_nsp(z.reshape(z.shape[0],-1))    # [n, n_cls]
        return mlm_logits, nsp_logits
    
    def step(self, seqs, segs, seqs_, nsp_labels, training=True):
        """
        seqs: [n, step[:-1]]
        segs: [n, step[:-1]]
        seqs_: [n, step[1:]]
        nsp_labels: [n]
        """
        if training:
            self.opt.zero_grad()
        mlm_logits, nsp_logits = self(seqs, segs, training)
        debug('mlm_logits:',mlm_logits.shape) # [n, step, n_vocab]
        debug('nsp_logits:',nsp_logits.shape) # [n, 2]
        pred_loss = cross_entropy(mlm_logits.reshape(-1,self.n_vocab),seqs_.reshape(-1)) 
        nsp_loss = cross_entropy(nsp_logits,nsp_labels.reshape(-1))
        loss = pred_loss + 0.2 * nsp_loss
        if training:
            loss.backward()
            self.opt.step()
        return loss.cpu().data.numpy(), mlm_logits
    
    def input_emb(self,seqs, segs):
        return self.word_emb(seqs) + self.segment_emb(segs) + self.position_emb
    
    def mask(self, seqs):
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        mask = torch.triu(torch.ones((seq_len,seq_len), dtype=torch.long), diagonal=1).to(device)  # [seq_len ,seq_len]
        pad = torch.eq(seqs,self.padding_idx)   # [n, seq_len]
        mask = torch.where(pad[:,None,None,:],1,mask[None,None,:,:]).to(device)   # [n, 1, seq_len, seq_len]
        return mask>0   # [n, 1, seq_len, seq_len]
    
    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.cpu().data.numpy() for l in self.encoder.encoder_layers]
        }
        return attentions

def calc_sparsity(tensor_np, threshold=1e-3):
    return np.mean(np.abs(tensor_np) < threshold)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in loader:
            seqs, segs, xlen, nsp_labels = batch
            seqs, segs, nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device), nsp_labels.to(device)

            # calculate loss and predictions
            loss, pred = model.step(seqs=seqs[:, :-1], segs=segs[:, :-1], seqs_=seqs[:, 1:], nsp_labels=nsp_labels, training=False)

            total_loss += loss.item()

            pred = pred.argmax(dim=-1)  # take the maximum value as the predicted label
            correct_preds += (pred == seqs[:, 1:]).sum().item()  # compare predicted and actual labels
            total_preds += pred.view(-1).shape[0]  # count total number of samples

    avg_loss = total_loss / len(loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

def train(model, train_loader, device, scheduler):
    model.train()  
    scheduler.step() 
    
    for batch_idx, batch in enumerate(train_loader):
        seqs, segs, xlen, nsp_labels = batch
        seqs, segs, nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device), nsp_labels.to(device)
        loss, pred = model.step(seqs=seqs[:, :-1], segs=segs[:, :-1], seqs_=seqs[:, 1:], nsp_labels=nsp_labels)

def lr_lambda(step):
    """
    Custom learning rate function.
    Args:
        step (int): Current step.
    Returns:
        float: Learning rate.
    """
    warmup_epochs = 50
    step = 1 if step == 0 else step
    return min(step ** -0.5, step * (warmup_epochs ** -1.5))

def run():
    if style == "mlp_ffn":
        model_name = "gpt-mlp"
    elif style == "conv1d_ffn":
        model_name = "gpt-conv1d-k{}".format(k_size)
    epochs = 1000
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    batch_size = 32
    dataset = utils.MRPCData("./MRPC", 50000)
    
    model = GPT(
        model_dim=MODEL_DIM, max_len=dataset.max_len-1, num_layer=N_LAYER, num_head=4, n_vocab=dataset.num_word,
        lr=LEARNING_RATE, max_seg=dataset.num_seg, drop_rate=0.3, padding_idx=dataset.pad_id
    )
    print_model_parameters(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    train_dataset = utils.MRPCData("./MRPC", 50000, data_type="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = utils.MRPCData("./MRPC", 50000, data_type="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.opt, lr_lambda=lr_lambda)

    test_loss_list, test_accuracy_list = [], []
    t0 = time.time()
    best_loss = float("inf")
    patience = 0
    patience_thread=30

    for epoch in range(epochs):
        train(model, train_loader, device, scheduler)
        # Evaluate on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        if test_loss < best_loss:
            patience = 0
            best_loss = test_loss
            os.makedirs("./visual/models", exist_ok=True)
            torch.save(model.state_dict(), f"./visual/models/{model_name}.pth")
        else:
            patience += 1
            if patience > patience_thread:
                print("Early stopping triggered.")
                print(f"patience: {patience} at epoch {epoch + 1} with best loss: {best_loss:.4f}")
                break
        
        print(f'Epoch {epoch+1}: Valid Loss: {test_loss:.3f} |  Valid Acc: {test_accuracy:.3f} | LR: {model.opt.param_groups[0]["lr"]:.6f}')
        debug("time per batch: ", (time.time() - t0) / len(train_loader))

    print("Training complete.")

    os.makedirs("./visual/loss", exist_ok=True)
    np.save(f"./visual/loss/test_loss_{model_name}.npy", np.array(test_loss_list))
    np.save(f"./visual/loss/test_accuracy_{model_name}.npy", np.array(test_accuracy_list))

    # live_plot(test_loss_list)


if __name__ == "__main__":
    run()
