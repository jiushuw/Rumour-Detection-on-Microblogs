import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import Transform_Numpy_Tensor, Transform_Tensor_Variable, load_data
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import argparse


class BERTLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, max_len, num_labels, dropout=0.5):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, text, image=None, attention_mask=None):
        with torch.no_grad():
            embedded = self.bert(text, attention_mask=attention_mask)[0]
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        batch_size = text.size(0)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--text_only", type=bool, default=True)
    parser.add_argument('--prefix', type=str, default='data/weibo/', help='')
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data, W = load_data(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset = Transform_Numpy_Tensor('train', train_data, tokenizer)
    val_dataset = Transform_Numpy_Tensor('val', val_data, tokenizer)
    test_dataset = Transform_Numpy_Tensor('test', test_data, tokenizer)

    def custom_collate(batch):
        inputs, labels, event_labels = zip(*batch)
        input_ids, images, attention_masks = zip(*inputs)
        images = [img if img is not None else torch.zeros(1) for img in images]
        input_ids = torch.stack(input_ids)
        images = torch.stack(images)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        event_labels = torch.stack(event_labels)
        return (input_ids, images, attention_masks), labels, event_labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate)
    model = BERTLSTMClassifier(hidden_dim=100, max_len=args.max_len, num_labels=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    is_text_only = len(train_dataset.image) == 0

    # Lists to store losses and accuracies
    losses = []
    accuracies = []

    for epoch in range(args.epochs):
        print(f"Training Epoch: {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0

        for (input_ids, image, attention_mask), label, event_label in train_loader:
            input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
            optimizer.zero_grad()
            input_ids = Transform_Tensor_Variable(input_ids)
            label = Transform_Tensor_Variable(label).long()
            output = model(input_ids, None if is_text_only else image, attention_mask)
            loss = criterion(output, label)
            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_preds += (predicted == label).sum().item()
            total_preds += label.size(0)
            loss.backward()
            optimizer.step()

        # Append average epoch loss
        losses.append(epoch_loss / len(train_loader))
        # Append epoch accuracy
        accuracies.append(correct_preds / total_preds)

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Test model after training
    model.eval()
    all_preds = []
    all_labels = []
    batch_counter = 0

    with torch.no_grad():
        for (input_ids, image, attention_mask), label, event_label in test_loader:
            input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
            input_ids = Transform_Tensor_Variable(input_ids)
            label = Transform_Tensor_Variable(label)
            output = model(input_ids, None if is_text_only else image, attention_mask)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            batch_counter += 1
            print(f"Processed batch {batch_counter} of {len(test_loader)}")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
