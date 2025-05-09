from pickle import load
import numpy as np
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy,softmax, relu
from utils import debug
import utils
from GPT import GPT
import os
import pickle
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
from transformer import k_size, style
MASK_RATE = 0.1

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


class BERT(GPT):
    def __init__(
        self, model_dim, max_len, num_layer, num_head, n_vocab, lr,
        max_seg=3, drop_rate=0.2, padding_idx=0) -> None:
        super().__init__(model_dim, max_len, num_layer, num_head, n_vocab, lr, max_seg, drop_rate, padding_idx)
        self.opt = optim.AdamW(self.parameters(), lr, weight_decay = 5e-4, eps=5e-9)
    
    def step(self,seqs,segs,seqs_, loss_mask,nsp_labels, parameters, training=True):
        device = next(self.parameters()).device
        if training:
            self.opt.zero_grad()
        mlm_logits, nsp_logits = self(seqs, segs, training=True)    # [n, step, n_vocab], [n, n_cls]
        mlm_loss = cross_entropy(
            torch.masked_select(mlm_logits,loss_mask).reshape(-1,mlm_logits.shape[2]),
            torch.masked_select(seqs_,loss_mask.squeeze(2))
            )
        nsp_loss = cross_entropy(nsp_logits,nsp_labels.reshape(-1))
        loss = mlm_loss + 0.1 * nsp_loss
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, norm_type=2)
            self.opt.step()
        return loss.cpu().data.numpy(),mlm_logits

    def mask(self, seqs):
        mask = torch.eq(seqs,self.padding_idx)
        return mask[:, None, None, :]

def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool_)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id

def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask

def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = torch.from_numpy(np.random.choice(word_ids, size=len(rand_id))).type(torch.IntTensor)
    return loss_mask

def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask

def random_mask_or_replace(data,arange,dataset):
    seqs, segs,xlen,nsp_labels = data
    seqs_ = seqs.data.clone()
    p = np.random.random()
    if p < 0.8:
        # mask
        loss_mask = np.concatenate([
            do_mask(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id,
                dataset.mask_id
                )
                for i in range(len(seqs))], axis=0)
    elif p < 0.9:
        # do nothing
        loss_mask = np.concatenate([
            do_nothing(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id
                )
                for i in range(len(seqs))],  axis=0)
    else:
        # replace
        loss_mask = np.concatenate([
            do_replace(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id,
                dataset.word_ids
                )
                for i in range(len(seqs))],  axis=0)
    loss_mask = torch.from_numpy(loss_mask).unsqueeze(2)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


def evaluate(model, loader, device,test_dataset):
    model.eval()  
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    arange = np.arange(test_dataset.max_len)
    with torch.no_grad():  
        for batch in loader:
            seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(batch,arange,test_dataset)

            seqs, segs, seqs_, nsp_labels, loss_mask = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),seqs_.type(torch.LongTensor).to(device), nsp_labels.to(device), loss_mask.to(device)
            param = model.parameters()
            loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels,param,False)
            total_loss += loss.item()

            pred = pred.argmax(dim=-1) 
            correct_preds += (pred == seqs).sum().item() 
            total_preds += pred.view(-1).shape[0] 

    avg_loss = total_loss / len(loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

warmup_epochs = 50
# custom learning rate scheduler
def lr_lambda(step):
    if step == 0:
        step = 1
    return min(step ** -0.5, step * (warmup_epochs ** -1.5))

def train(model, train_loader, device, arange, train_dataset, scheduler):
    model.train()  
    scheduler.step()  

    for batch_idx, batch in enumerate(train_loader):
        seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(batch,arange,train_dataset)

        seqs, segs, seqs_, nsp_labels, loss_mask = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),seqs_.type(torch.LongTensor).to(device), nsp_labels.to(device), loss_mask.to(device)
        param = model.parameters()
        loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels,param)

def run():
    if style == "mlp_ffn":
        model_name = "bert-mlp"
    elif style == "conv1d_ffn":
        model_name = "bert-conv1d-k{}".format(k_size)

    epochs = 1000
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    batch_size = 32
    dataset = utils.MRPCData("./MRPC", 50000)
    debug("dataset.num_seg: ", dataset.num_seg)
    debug("dataset.pad_id: ", dataset.pad_id)
    debug("dataset.i2v: ", dataset.i2v.keys().__len__())
    
    model = BERT(
        model_dim=MODEL_DIM, max_len=dataset.max_len, num_layer=N_LAYER, num_head=4, n_vocab=dataset.num_word,
        lr=LEARNING_RATE, max_seg=dataset.num_seg, drop_rate=0.2, padding_idx=dataset.pad_id
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print_model_parameters(model)

    train_dataset = utils.MRPCData("./MRPC", 50000, data_type="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = utils.MRPCData("./MRPC", 50000, data_type="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scheduler = torch.optim.lr_scheduler.LambdaLR(model.opt, lr_lambda=lr_lambda)
    arange = np.arange(train_dataset.max_len)

    test_accuracy_list, test_loss_list = [], []
    t0 = time.time()
    best_loss = float("inf")
    patience = 0
    patience_thread = 30

    for epoch in range(epochs):
        train(model, train_loader, device, arange, train_dataset, scheduler)
        test_loss, test_accuracy = evaluate(model, test_loader, device,test_dataset)
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

        print(f'Epoch {epoch+1}: Valid Loss: {test_loss:.3f} | Valid Acc: {test_accuracy:.3f} | LR: {model.opt.param_groups[0]["lr"]:.6f}')

        debug("time per batch: ", (time.time() - t0) / len(train_loader))

    print("Training complete.")
    os.makedirs("./visual/loss", exist_ok=True)
    np.save(f"./visual/loss/test_loss_{model_name}.npy", np.array(test_loss_list))
    np.save(f"./visual/loss/test_accuracy_{model_name}.npy", np.array(test_accuracy_list))

    # live_plot(test_loss_list)

if __name__ == "__main__":
    run()