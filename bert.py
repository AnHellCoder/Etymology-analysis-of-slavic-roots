import os

from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import random as rd

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        class_weights = torch.tensor([0.27, 0.73]).to(logits.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def replace_protoslavic(example):
    if example["label"] == "Proto-Slavic":
        example["label"] = "INHERITED"
    return example

def to_labels(example):
  if example["label"] == 'INHERITED':
    example["label"] = 0
  else:
    example["label"] = 1
  return example

def tokenize(batch):
    return tokenizer(
        batch["word"],
        padding="max_length",
        truncation=True,
        max_length=16,  # для слов достаточно
    )

model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(
    "csv",
    data_files={
        "train": "datasets/dataset_multi_augmented.csv",
    }
)

dataset = dataset.remove_columns(
    [c for c in dataset["train"].column_names if c not in ["word",
    "protolanguage"]]
)

dataset = dataset.rename_column("protolanguage", "label")

dataset = dataset.map(replace_protoslavic)
dataset = dataset.map(to_labels)

#Заимствованные корни с наследованными приставками и суффиксами. Необходимо рассмотреть распределение внимания для этих слов
test_words = ["дебажить", "электричество", "тональность", "штуковина", "сервак", "акк", "агриться", "залутать", "видос", "криповый", "имба"]

declinable = dataset["train"].filter(lambda x: x["declinable"] == True)
undeclinable = dataset["train"].filter(lambda x: x["declinable"] == False)

declinable = rd.sample(list(declinable["word"]), 5)
undeclinable = rd.sample(list(undeclinable["word"]), 5)

words = declinable + undeclinable

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

training_args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,

    num_train_epochs=100,
    weight_decay=0.01,

    logging_steps=10,
    load_best_model_at_end=True,

    report_to="none",
    optim="adamw_torch"
)

fold_histories = []

labels = np.array(dataset["train"]["labels"])
#centuries = np.array(dataset["train"]["borrowed_century"])

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_f1 = []
general_correct_predictions = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    #========== Визуализация предобученной модели ==========#
    #========== Визуализация усреднения распределения внимания по всем головам для предобуч. RuBERT ==========#
    os.makedirs("./etymology-analysis/RuBERT_attentions", exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        attn_implementation="eager",
    )

    inputs = tokenizer(
        words,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_attentions=True
    )

    attentions = outputs.attentions

    layer = -1
    head = 0

    for i, w in enumerate(words):
        plt.figure()
        attn_mean = attentions[layer][i].mean(dim=0).detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][i]
        )

        plt.imshow(attn_mean, interpolation="nearest")
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title("RuBERT mean attention for word " + w)
        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/(PRETRAINED)RuBERT_mean_attention_fold{fold+1}_word{i}.png")
    
        plt.close()
    #========== Конец визуализации усреднения распределения внимания по всем головам для предобуч. RuBERT ==========#


    #========== Визуализация распределений внимания по всем головам для предобуч. RuBERT на одном слове ==========#
    num_heads = attentions[layer].shape[1]
    for h in range(num_heads):
        plt.figure()
        attn = attentions[layer][0, h].detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        plt.imshow(attn, interpolation="nearest")
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"RuBERT attn: word {words[0]} - head {h}")
        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/(PRETRAINED)RuBERT_attention_fold{fold+1}_head{h}.png")

        plt.close()
    #========== Конец визуализации распределений внимания по всем головам для предобуч. RuBERT на одном слове ==========#

    #========== Визуализация усреднения распределения внимания по всем головам для предобуч. RuBERT на всех словах ==========#
    inputs = tokenizer(
        test_words,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_attentions=True,
    )

    attentions = outputs.attentions
    for i, w in enumerate(test_words):
        plt.figure()
        attn = attentions[layer][i].mean(dim=0).detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][i]
        )

        plt.imshow(attn, interpolation="nearest")

        plt.xticks(
            range(len(tokens)),
            tokens,
            rotation=90
        )

        plt.yticks(
            range(len(tokens)),
            tokens
        )

        plt.title(f"RuBERT attention for test word {w}")

        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/(PRETRAINED)RuBERT_mean_attention_fold{fold+1}_testword{i}.png")

        plt.close()
    #========== Конец визуализации усреднения распределения внимания по всем головам для предобуч. RuBERT на всех словах ==========#
    #========== Конец визуализации предобученной модели ==========#


    print(f"\n===== Fold {fold+1} =====")

    train_dataset = dataset["train"].select(train_idx)
    val_dataset = dataset["train"].select(val_idx)

    train_dataset = train_dataset.remove_columns(["word"])
    val_dataset = val_dataset.remove_columns(["word"])

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    # собираем F1 по эпохам
    history = trainer.state.log_history

    epochs = []
    f1_scores = []

    for record in history:
        if "eval_f1" in record:
            epochs.append(record["epoch"])
            f1_scores.append(record["eval_f1"])

    fold_histories.append((epochs, f1_scores))

    metrics = trainer.evaluate()

    fold_f1 = metrics["eval_f1"]
    print("Fold F1:", fold_f1)

    all_f1.append(fold_f1)

    #Устанавливаем тестовые датасеты, чтобы проверить, насколько хорошо модель определяет конкретные заимствования
    arabian_testset = load_dataset('csv', data_files='datasets/dataset_arabian.csv')['train']
    chinese_testset = load_dataset('csv', data_files='datasets/dataset_chinese.csv')['train']
    english_testset = load_dataset('csv', data_files='datasets/dataset_english.csv')['train']
    french_testset = load_dataset('csv', data_files='datasets/dataset_french.csv')['train']
    german_testset = load_dataset('csv', data_files='datasets/dataset_german.csv')['train']
    italian_testset = load_dataset('csv', data_files='datasets/dataset_italian.csv')['train']
    japanese_testset = load_dataset('csv', data_files='datasets/dataset_japanese.csv')['train']
    polish_testset = load_dataset('csv', data_files='datasets/dataset_polish.csv')['train']

    #Приводим датасеты к требуемому для теста виду
    arabian_testset = arabian_testset.map(tokenize, batched=True)
    arabian_testset.set_format("torch")
    arabian_testset = arabian_testset.remove_columns(['word',"Unnamed: 0"])
    arabian_testset = arabian_testset.rename_column('protolanguage','label')
    arabian_testset = arabian_testset.map(replace_protoslavic)
    arabian_testset = arabian_testset.map(to_labels)

    chinese_testset = chinese_testset.map(tokenize, batched=True)
    chinese_testset.set_format("torch")
    chinese_testset = chinese_testset.remove_columns(["word","Unnamed: 0"])
    chinese_testset = chinese_testset.rename_column('protolanguage','label')
    chinese_testset = chinese_testset.map(replace_protoslavic)
    chinese_testset = chinese_testset.map(to_labels)

    english_testset = english_testset.map(tokenize, batched=True)
    english_testset.set_format("torch")
    english_testset = english_testset.remove_columns(['word', "Unnamed: 0"])
    english_testset = english_testset.rename_column('protolanguage','label')
    english_testset = english_testset.map(replace_protoslavic)
    english_testset = english_testset.map(to_labels)

    french_testset = french_testset.map(tokenize, batched=True)
    french_testset.set_format("torch")
    french_testset = french_testset.remove_columns(['word',"Unnamed: 0"])
    french_testset = french_testset.rename_column('protolanguage','label')
    french_testset = french_testset.map(replace_protoslavic)
    french_testset = french_testset.map(to_labels)

    german_testset = german_testset.map(tokenize, batched=True)
    german_testset.set_format("torch")
    german_testset = german_testset.remove_columns(['word',"Unnamed: 0"])
    german_testset = german_testset.rename_column('protolanguage','label')
    german_testset = german_testset.map(replace_protoslavic)
    german_testset = german_testset.map(to_labels)

    italian_testset = italian_testset.map(tokenize, batched=True)
    italian_testset.set_format("torch")
    italian_testset = italian_testset.remove_columns(['word',"Unnamed: 0"])
    italian_testset = italian_testset.rename_column('protolanguage','label')
    italian_testset = italian_testset.map(replace_protoslavic)
    italian_testset = italian_testset.map(to_labels)

    japanese_testset = japanese_testset.map(tokenize, batched=True)
    japanese_testset.set_format("torch")
    japanese_testset = japanese_testset.remove_columns(['word',"Unnamed: 0"])
    japanese_testset = japanese_testset.rename_column('protolanguage','label')
    japanese_testset = japanese_testset.map(replace_protoslavic)
    japanese_testset = japanese_testset.map(to_labels)

    polish_testset = polish_testset.map(tokenize, batched=True)
    polish_testset.set_format("torch")
    polish_testset = polish_testset.remove_columns(['word',"Unnamed: 0"])
    polish_testset = polish_testset.rename_column('protolanguage','label')
    polish_testset = polish_testset.map(replace_protoslavic)
    polish_testset = polish_testset.map(to_labels)

    # предсказания
    predictions_arabian = trainer.predict(arabian_testset)
    predictions_chinese = trainer.predict(chinese_testset)
    predictions_english = trainer.predict(english_testset)
    predictions_french = trainer.predict(french_testset)
    predictions_german = trainer.predict(german_testset)
    predictions_italian = trainer.predict(italian_testset)
    predictions_japanese = trainer.predict(japanese_testset)
    predictions_polish = trainer.predict(polish_testset)

    print("Точность определения арабских заимств.:", compute_metrics((predictions_arabian.predictions,predictions_arabian.label_ids)))
    print("Точность определения китайских заимств.:", compute_metrics((predictions_chinese.predictions,predictions_chinese.label_ids)))
    print("Точность определения английских заимств.:", compute_metrics((predictions_english.predictions,predictions_english.label_ids)))
    print("Точность определения французских заимств.:", compute_metrics((predictions_french.predictions,predictions_french.label_ids)))
    print("Точность определения немецких заимств.:", compute_metrics((predictions_german.predictions,predictions_german.label_ids)))
    print("Точность определения итальянских заимтсв.:", compute_metrics((predictions_italian.predictions,predictions_italian.label_ids)))
    print("Точность определения японских заимств.:", compute_metrics((predictions_japanese.predictions,predictions_japanese.label_ids)))
    print("Точность определения польских заимств.:", compute_metrics((predictions_polish.predictions,predictions_polish.label_ids)))

    general_correct_predictions.append(
        (predictions_arabian.predictions.argmax(axis=1) == predictions_arabian.label_ids).sum().item()+
        (predictions_chinese.predictions.argmax(axis=1) == predictions_chinese.label_ids).sum().item()+
        (predictions_english.predictions.argmax(axis=1) == predictions_english.label_ids).sum().item()+
        (predictions_french.predictions.argmax(axis=1) == predictions_french.label_ids).sum().item()+
        (predictions_german.predictions.argmax(axis=1) == predictions_german.label_ids).sum().item()+
        (predictions_italian.predictions.argmax(axis=1) == predictions_italian.label_ids).sum().item()+
        (predictions_japanese.predictions.argmax(axis=1) == predictions_japanese.label_ids).sum().item()+
        (predictions_polish.predictions.argmax(axis=1) == predictions_polish.label_ids).sum().item()
    )

    #===== Визуализация дообученной модели =====#
    #===== Визуализация усреднений распределений внимания по всем словам на дообученной модели =====#
    inputs = tokenizer(
        words,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_attentions=True
    )

    attentions = outputs.attentions

    layer = -1
    head = 0

    for i, w in enumerate(words):
        plt.figure()
        attn_mean = attentions[layer][i].mean(dim=0).detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][i]
        )

        plt.imshow(attn_mean, interpolation="nearest")
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title("RuBERT mean attention for word " + w)
        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/RuBERT_mean_attention_fold{fold+1}_word{i}.png")
    
        plt.close()
    #===== Конец визуализации усреднений распределений внимания по всем словам на дообученной модели =====#


    #===== Визуализация распределений внимания по всем головам на дообученной модели на одном слове =====#
    num_heads = attentions[layer].shape[1]
    for h in range(num_heads):
        plt.figure()
        attn = attentions[layer][0, h].detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        plt.imshow(attn, interpolation="nearest")
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"RuBERT attention for word {words[0]} - head {h}")
        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/RuBERT_attention_fold{fold+1}_head{h}.png")

        plt.close()
    #===== Конец визуализации распределений внимания по всем головам на дообученной модели на одном слове =====#


    #===== Визуализация усреднений распределений внимания по всем словам на дообученной модели =====#
    inputs = tokenizer(
        test_words,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_attentions=True
    )

    attentions = outputs.attentions

    for i, w in enumerate(test_words):
        plt.figure()
        attn = attentions[layer][i].mean(dim=0).detach().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][i]
        )

        plt.imshow(attn, interpolation="nearest")

        plt.xticks(
            range(len(tokens)),
            tokens,
            rotation=90
        )

        plt.yticks(
           range(len(tokens)),
            tokens
        )

        plt.title(f"RuBERT attention for test word {w}")

        plt.colorbar()

        plt.savefig(f"./etymology-analysis/RuBERT_attentions/RuBERT_mean_attention_fold{fold+1}_testword{i}.png")

        plt.close()
    #===== Конец визуализации дообученной модели =====#

print("\n===== FINAL RESULT =====")
print("Mean F1:", np.mean(all_f1))
print("Std F1:", np.std(all_f1))
print("Total correct predictions per fold:", general_correct_predictions)