import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score # +++ 导入评估指标 +++


def main():
    # 1. 路径和参数设置
    # ==================================
    MODEL_NAME = '/root/autodl-tmp/model/bert-base-uncased'
    DATASET_NAME = '/root/autodl-tmp/GLUE/MRPC'
    FINETUNED_MODEL_PATH = '/root/autodl-tmp/IGN/mrpc_bert_best_mode.pth'
    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载分词器和训练好的模型
    # ==================================
    print("Loading tokenizer and fine-tuned model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
    model.to(device)
    model.eval()

    # 3. 加载并预处理测试数据集
    print(f"Loading validation split of '{DATASET_NAME}' dataset to create a new test set...")
    original_validation_dataset = load_dataset(DATASET_NAME, split='validation')

    # 计算后50%的起始索引
    num_val_samples = len(original_validation_dataset)
    start_index = int(num_val_samples * 0.5)

    # 选择后50%的数据作为新的测试集
    new_test_dataset = original_validation_dataset.select(range(start_index, num_val_samples))
    print(f"Original validation set size: {num_val_samples}")
    print(f"Using the latter 50% as the new test set. New test set size: {len(new_test_dataset)}")

    # 预处理函数，与训练时保持一致
    def preprocess_function(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=512)

    tokenized_test_dataset = new_test_dataset.map(preprocess_function, batched=True)

    # +++ 调整列以进行评估 +++
    # 移除原始文本列，并将 'label' 重命名为模型期望的 'labels'
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text1', 'text2', 'idx','label_text'])
    tokenized_test_dataset = tokenized_test_dataset.rename_column('label', 'labels')
    tokenized_test_dataset.set_format('torch')

    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=BATCH_SIZE)

    # 4. 执行预测与评估
    # ==================================
    all_predictions = []
    all_labels = [] # +++ 新增：用于存储真实标签 +++
    print("Running evaluation on the new test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # +++ 将真实标签也收集起来 +++
            true_labels = batch['labels'].cpu().numpy()
            all_labels.extend(true_labels)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # 5. 计算并打印评估结果
    # ==================================
    # +++ 移除文件写入功能，替换为性能评估 +++
    accuracy = accuracy_score(all_labels, all_predictions)
    print("=" * 50)
    print("Evaluation finished!")
    print(f"Accuracy on the new test set (latter 50% of validation set): {accuracy:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()

