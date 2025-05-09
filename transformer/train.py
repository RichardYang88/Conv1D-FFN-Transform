"""
@author : Qingchuan Ynag
@when : 2025-04-20
"""
import os
import numpy as np
from torch.optim import AdamW
from data import *
from models.model.transformer import Transformer
from torch import nn

def print_model_parameters(model):
    print("Model layers and parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"Layer: {name:<60} | Shape: {str(param.shape):<20} | Parameters: {num_params:,}")
    print(f"Total Trainable Parameters: {total_params:,}\n")

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device,
                    style=style).to(device)

print_model_parameters(model)
model.apply(initialize_weights)
optimizer = AdamW(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# custom learning rate scheduler
def lr_lambda(step):
    step = 1 if step == 0 else step
    return min(step ** -0.5, step * (warmup ** -1.5))

# create a learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])        
        pred = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(pred, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            pred = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(pred, trg)
            epoch_loss += loss.item()
            pred = pred.argmax(dim=-1) 
            correct_preds += (pred == trg).sum().item() 
            total_preds += pred.view(-1).shape[0] 
    return epoch_loss / len(iterator), correct_preds / total_preds


def run(total_epoch, best_loss):
    patience = 0
    model_name = style
    valid_losses, accuracy_list = [], []
    for epoch in range(total_epoch):
        train(model, train_iter, optimizer, criterion, clip)
        valid_loss, accuracy = evaluate(model, valid_iter, criterion)

        scheduler.step()  
        valid_losses.append(valid_loss)
        accuracy_list.append(accuracy)

        if valid_loss < best_loss:
            patience = 0
            best_loss = valid_loss
            torch.save(model.state_dict(), f"./saved/transformer/{model_name}.pth")
            print(f'Best Valid Loss: {best_loss:.3f} |  Best Valid Acc: {accuracy:.3f}')
        else:
            patience += 1
            if patience > patience_thread:
                print("Early stopping triggered.")
                print(f"patience: {patience} at epoch {epoch + 1} with best loss: {best_loss:.4f}")
                break

        print(f'Epoch {epoch+1}: Valid Loss: {valid_loss:.3f} | Valid Acc: {accuracy:.3f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    os.makedirs("./saved/transformer", exist_ok=True)
    np.save(f"saved/transformer/valid_loss_{model_name}.npy", np.array(valid_losses))
    np.save(f"saved/transformer/accuracy_{model_name}.npy", np.array(accuracy_list))
            

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
