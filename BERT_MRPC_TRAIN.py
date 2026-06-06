import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim.adamw import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

#为模型注入GNI噪声
def inject_gni_noise(model: torch.nn.Module, alpha: float):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad_norm = torch.norm(p.grad, p='fro')
            #逆梯度
            beta_t_p = alpha / (1.0 + grad_norm)
            # 正梯度
            # beta_t_p = alpha * (1.0 + grad_norm)
            # 随机采样范围 [0.5, 1.5]
            # rfactor = random.uniform(0.5, 1.5)
            # beta_t_p = alpha * rfactor

            std_p = torch.std(p.data)
            if std_p == 0:
                continue
            noise = torch.rand_like(p.data) * 2 - 1
            adaptive_noise = beta_t_p * noise * std_p
            p.data.add_(adaptive_noise)

#为模型注入NoisyTune噪声
def noisytune(model, noise_lambda, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'embedding' not in name:
                std = torch.std(param)
                # 生成与param相同设备和数据类型的噪声，生成[-0.5, 0.5)范围内的随机噪声
                noise = (torch.rand(param.size(), device=param.device, dtype=param.dtype) - 0.5) * noise_lambda * std
                param.add_(noise)
    return model



#设置随机种子
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    MODEL_NAME = "/root/autodl-tmp/model/bert-base-uncased" #加载bert模型位置
    BEST_MODEL_PATH = '/root/autodl-tmp/GNI/mrpc_bert_best_mode.pth'#模型保存位置
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5
    ALPHA = 0.0005 #GNI噪声强度，强度为0时则不添加GNI
    PATIENCE = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("/root/autodl-tmp/GLUE/MRPC")#数据集加载位置

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=512)

    print("Tokenizing the dataset... (This may take a while for MRPC)")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['text1', 'text2', 'idx', 'label_text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')
    #切割训练集为训练集和测试集
    full_validation_dataset = tokenized_datasets['validation']
    num_val_samples = len(full_validation_dataset)
    new_val_size = int(num_val_samples * 0.5)
    new_validation_dataset = full_validation_dataset.select(range(new_val_size))

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(new_validation_dataset, batch_size=BATCH_SIZE)

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    # model = noisytune(model, noise_lambda=0.15, seed=1)  #注入策略为NoisyTune
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            inject_gni_noise(model, alpha=ALPHA)  #注入策略为GNI
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\nEpoch {epoch + 1} Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_epoch = epoch + 1
            print(f"# New best model found! Saving model to {BEST_MODEL_PATH}")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        print("--------------------------------------\n")

    print("Training finished!")
    print(f"Best model was saved from Epoch {best_epoch} with Validation Accuracy: {best_validation_accuracy:.4f}")
    print(f"Model weights saved to: {BEST_MODEL_PATH}")

if __name__ == '__main__':
    main()