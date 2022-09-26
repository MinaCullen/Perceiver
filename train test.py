import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AdamW
from sklearn.metrics import accuracy_score
from datasets import load_metric
import time
from transformers import PerceiverFeatureExtractor
import Perceiver
import Lucidrains


def labels_to_vec(labels, num_labels):
    labels_list = []
    for label in labels:
        label_as_list = [0] * num_labels
        label_as_list[label.item()] = 1
        labels_list.append(label_as_list)
    return torch.Tensor(labels_list)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        print("val loss - train loss")
        print(validation_loss - train_loss)
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def run_train_test(model_type="Perceiver"):
    train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])

    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    id2label = {idx: label for idx, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: idx for idx, label in id2label.items()}
    print(id2label)
    print(label2id)
    feature_extractor = PerceiverFeatureExtractor()

    def preprocess_images(examples):
        examples['pixel_values'] = feature_extractor(examples['img'], return_tensors="pt").pixel_values
        examples['pixel_values'] = examples['pixel_values'].transpose(2, 3).transpose(1, 3)
        return examples

    train_ds.set_transform(preprocess_images)
    val_ds.set_transform(preprocess_images)
    test_ds.set_transform(preprocess_images)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_batch_size = 4
    eval_batch_size = 4

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

    batch = next(iter(train_dataloader))
    assert batch['pixel_values'].shape == (train_batch_size, 224, 224, 3)
    assert batch['labels'].shape == (train_batch_size,)

    if model_type == "Perceiver":
        model = Perceiver.make_model_img_classification(
            max_seq_len=1024,
            dim=29,
            num_output_tokens=10,
            N=6,
            dim_latents=29,
            num_latents=256,
            self_attn_heads=8,
            cross_attn_heads=1,
            dim_cross_attn=64,
            dim_self_attn=512,
            depth=1,
            dropout_rate=0.1
        )
    elif model_type == "Lucidrains":
        model = Lucidrains.Perceiver(
            input_channels=3,          # number of channels for each token of the input
            input_axis=2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=6,          # number of freq bands, with original value (2 * K + 1)
            max_freq=10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth=6,
            num_latents=256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,            # latent dimension
            cross_heads=1,             # number of heads for cross attention. paper said 1
            latent_heads=8,            # number of heads for latent self attention, 8
            cross_dim_head=64,         # number of dimensions per cross attention head
            latent_dim_head=64,        # number of dimensions per latent self attention head
            num_classes=10,          # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=2      # number of self attention blocks per cross attention
        )
    start_time = time.time()
    train_model(model, train_dataloader, val_dataloader, 10)
    elapsed = time.time() - start_time
    print("Training finished in %f seconds" % elapsed)
    test_model(model, test_dataloader)


def train_model(model, train_dataloader, val_dataloader, num_labels):
    module = model

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_func = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(tolerance=4, min_delta=1)

    model.train()
    step = 0

    for epoch in range(20):  # loop over the dataset multiple times
        loss_calcs = 0
        val_acc = load_metric("accuracy")
        epoch_train_loss = 0
        print("Epoch:", epoch)
        for batch in enumerate(train_dataloader, 0):
            # get the inputs;
            batch = batch[1]

            inputs = batch["pixel_values"]
            labels = batch["labels"]

            labels = labels_to_vec(labels, num_labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, logits = model(inputs)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            loss_calcs += 1

            if step % 100 == 0:
                predictions = logits.argmax(-1).cpu().detach().numpy()
                accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
                print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
                print(f"Labels: {batch['labels'].numpy()}, Predictions: {predictions}")
                print()

            step += 1

        epoch_train_loss = epoch_train_loss / loss_calcs

        loss_calcs = 0

        print("Validating")
        epoch_val_loss = 0
        for batch in enumerate(val_dataloader, 0):
            batch = batch[1]
            inputs = batch["pixel_values"]
            labels = batch["labels"]

            labels = labels_to_vec(labels, num_labels)

            outputs, logits = model(inputs)
            predictions = logits.argmax(-1).cpu().detach().numpy()
            val_acc.add_batch(predictions=predictions, references=batch["labels"].numpy())

            loss = loss_func(logits, labels)
            epoch_val_loss += loss.item()
            loss_calcs += 1

            if step % 100 == 0:
                accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
                print(f"Accuracy: {accuracy}")
                print(f"Labels: {batch['labels'].numpy()}, Predictions: {predictions}")
                print()

        epoch_val_loss = epoch_val_loss / loss_calcs

        val_score = val_acc.compute()
        print(f"epoch {epoch} complete, validation accuracy: {val_score}")
        print()

        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print("Stopping at epoch:", epoch)
            print()
            break

        file_path = "cifar10_Perceiver_final.pt"
        torch.save(module.state_dict(), file_path)


def test_model(model, test_dataloader):
    accuracy = load_metric("accuracy")

    model.eval()
    step = 0
    for batch in enumerate(test_dataloader, 0):
        batch = batch[1]
        # get the inputs;
        inputs = batch["pixel_values"]
        labels = batch["labels"]

        # forward pass
        outputs, logits = model(inputs)

        predictions = logits.argmax(-1).cpu().detach().numpy()
        references = batch["labels"].numpy()
        accuracy.add_batch(predictions=predictions, references=references)

        if step % 100 == 0:
            print(f"Labels: {batch['labels'].numpy()}, Predictions: {predictions}")
            print()

    final_score = accuracy.compute()
    print("Accuracy on test set:", final_score)


run_train_test()
