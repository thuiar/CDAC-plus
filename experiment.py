from model import BertForConstrainClustering
from utils import *
import argparse
import random
import torch
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import trange
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from datetime import datetime
import warnings
warnings.warn = warn

results_all = {}
seed = 44
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task_name = dataset = sys.argv[1]
fraction = int(sys.argv[2])
labeled_ratio = float(sys.argv[3])
unknown_cls_ratio = float(sys.argv[4])

data_dir = 'data/' + task_name
output_dir = '/tmp/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
bert_model = "bert-base-uncased"
num_train_epochs = 46

max_seq_task = {
    "snips": 35,
    'dbpedia': 54,
    "stackoverflow": 20,
}
max_seq_length = max_seq_task[task_name]
train_batch_size = 256
eval_batch_size = 256
learning_rate = 2e-5
warmup_proportion = 0.1

processors = {
    "snips": SnipsProcessor,
    'dbpedia': Dbpedia_Processor,
    "stackoverflow": Stackoverflow_Processor,
}

num_labels_task = {
    "snips": 7,
    'dbpedia': 14,
    "stackoverflow": 20,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))
logger.disabled = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if os.path.exists(output_dir) and os.listdir(output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
num_labels = num_labels_task[task_name]
label_list = processor.get_labels()
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

# Freezing all transformer (except the last layer)
model = BertForConstrainClustering.from_pretrained(bert_model, num_labels = num_labels*fraction)
for name, param in model.bert.named_parameters():  
    param.requires_grad = False
    if "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train_examples = processor.get_train_examples(data_dir)
num_train_optimization_steps = int(len(train_examples) / train_batch_size) * num_train_epochs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

## Settings of unknown classes discovery
# 1. Select 25% classes as unknown(-1)
# 2. Set 90% of examples as unknown(-1)
n_unknown_cls = round(num_labels*unknown_cls_ratio)
label_unknown = np.random.choice(np.array(label_list), n_unknown_cls, replace=False)

train_labeled_examples, train_unlabeled_examples = [], []
for example in train_examples:
    if (example.label not in label_unknown) and (np.random.uniform(0, 1) <= labeled_ratio):
        train_labeled_examples.append(example)
    else:
        train_unlabeled_examples.append(example)


train_loss = 0
train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Loading training set *****")
logger.info("  Num examples = %d", len(train_examples))
train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)


train_labeled_features = convert_examples_to_features(train_labeled_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Running training(labeled) *****")
logger.info("  Num examples = %d", len(train_labeled_features))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
train_labeled_input_ids = torch.tensor([f.input_ids for f in train_labeled_features], dtype=torch.long)
train_labeled_input_mask = torch.tensor([f.input_mask for f in train_labeled_features], dtype=torch.long)
train_labeled_segment_ids = torch.tensor([f.segment_ids for f in train_labeled_features], dtype=torch.long)
train_labeled_label_ids = torch.tensor([f.label_id for f in train_labeled_features], dtype=torch.long)

train_labeled_data = TensorDataset(train_labeled_input_ids, train_labeled_input_mask, train_labeled_segment_ids, train_labeled_label_ids)
train_labeled_sampler = RandomSampler(train_labeled_data)
train_labeled_dataloader = DataLoader(train_labeled_data, sampler=train_labeled_sampler, batch_size=train_batch_size)


train_unlabeled_features = convert_examples_to_features(train_unlabeled_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Running training(unlabeled) *****")
logger.info("  Num examples = %d", len(train_unlabeled_features))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
train_unlabeled_input_ids = torch.tensor([f.input_ids for f in train_unlabeled_features], dtype=torch.long)
train_unlabeled_input_mask = torch.tensor([f.input_mask for f in train_unlabeled_features], dtype=torch.long)
train_unlabeled_segment_ids = torch.tensor([f.segment_ids for f in train_unlabeled_features], dtype=torch.long)
train_unlabeled_label_ids = torch.tensor([-1 for f in train_unlabeled_features], dtype=torch.long)
train_semi_input_ids = torch.cat([train_labeled_input_ids, train_unlabeled_input_ids])
train_semi_input_mask = torch.cat([train_labeled_input_mask, train_unlabeled_input_mask])
train_semi_segment_ids = torch.cat([train_labeled_segment_ids, train_unlabeled_segment_ids])
train_semi_label_ids = torch.cat([train_labeled_label_ids, train_unlabeled_label_ids])

train_semi_data = TensorDataset(train_semi_input_ids, train_semi_input_mask, train_semi_segment_ids, train_semi_label_ids)
train_semi_sampler = RandomSampler(train_semi_data)
train_semi_dataloader = DataLoader(train_semi_data, sampler=train_semi_sampler, batch_size=train_batch_size)

## Evaluate for each epcoh
eval_examples = processor.get_dev_examples(data_dir)
eval_labeled_examples = []
for example in eval_examples:
    if example.label not in label_unknown:
        eval_labeled_examples.append(example)
        
eval_features = convert_examples_to_features(eval_labeled_examples, label_list, max_seq_length, tokenizer)
logger.info("")
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_features))
logger.info("  Batch size = %d", eval_batch_size)
eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)


global_step = 0
u = 0.95
l = 0.455 
eta = 0

y_pred_last = np.zeros_like(eval_label_ids)
for _ in trange(int(num_train_epochs), desc="Epoch"):
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_labeled_dataloader, desc="Iteration (labeled)")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, u, l, 'train', label_ids)
        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    train_labeled_loss = tr_loss / nb_tr_steps
    
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_semi_dataloader, desc="Iteration (all train)")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, u, l, 'train', label_ids, True)
        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    train_loss = tr_loss / nb_tr_steps

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    y_preds = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, u, l, 'train', label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        eval_loss += tmp_eval_loss.mean().item()
        y_preds.append(np.argmax(logits, 1))
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    y_pred = np.hstack(y_preds)
    y_eval = eval_label_ids.numpy()

    results = clustering_score(y_eval, y_pred)
    ## Confusion matrix(reorder y_pred for alignment)
    ind, w = hungray_aligment(y_eval, y_pred)
    d_ind = {i[0]: i[1] for i in ind}
    y_pred_ = pd.Series(y_pred).map(d_ind)
    cm = confusion_matrix(y_eval, y_pred_)

    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
    y_pred_last = np.copy(y_pred)

    results['u'] = u
    results['l'] = l
    loss = tr_loss
    result = {'eval_loss': eval_loss,
              'results': results,
              'global_step': global_step,
              'train_labeled_loss': train_labeled_loss,
              'train_loss': train_loss,
              'delta_label': delta_label}
    print(cm)
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    print(f"{_}: loss={train_loss}, (u, l) = ({round(u, 4)},{round(l, 4)})")
    eta += 1.1 * 0.009
    u = 0.95 - eta
    l = 0.455 + eta*0.1
    if u < l:
        break
results_all.update({'CDAC': results})
print(results, label_unknown)

plot_confusion_matrix(cm, label_list, normalize=False, figsize=(8, 8),
                      title='Confusion matrix, accuracy=' + str(results['ACC']))
# Save a trained model and the associated configuration
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
with open(output_config_file, 'w') as f:
    f.write(model_to_save.config.to_json_string())
    
    

test_examples = processor.get_test_examples(data_dir)
test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer)
logger.info("")
logger.info("***** Running testuation *****")
logger.info("  Num examples = %d", len(test_examples))
logger.info("  Batch size = %d", eval_batch_size)
test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

k = num_labels*fraction
y_true = train_label_ids
num_train_epochs = 100
learning_rate = 5e-5

# Initialize cluster centroids U with representation I
embs_train = []
for batch in tqdm(train_dataloader, desc="Extracting representation I"):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    
    with torch.no_grad():
        logits, q = model(input_ids, segment_ids, input_mask, mode='finetune')
    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    embs_train.append(logits)
emb_train = np.vstack(embs_train)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=k, n_jobs=-1, random_state=seed)
km.fit(emb_train)
y_pred_last = np.copy(km.cluster_centers_)
model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(device)

model.eval()

# Extracting probabilities Q
embs_test = []
qs = []
for batch in tqdm(test_dataloader, desc="Extracting probabilities Q"):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    with torch.no_grad():
        logits, q = model(input_ids, segment_ids, input_mask, mode='finetune')
    q = q.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    qs.append(q)
    embs_test.append(logits)

q_all = np.vstack(qs)
y_pred = q_all.argmax(1)
results = clustering_score(test_label_ids, y_pred)
results_all.update({'CDAC-KM': results})
print(results)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(len(train_examples) / train_batch_size) * num_train_epochs
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion, t_total=num_train_optimization_steps)


import copy
model_best = None
nmi_best = 0
wait, patient = 0, 5
for epoch in range(num_train_epochs):
    # Calculate probabilities P (as target)
    model.eval()
    qs = []
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits, q = model(input_ids, segment_ids, input_mask, mode='finetune')
        q = q.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        qs.append(q)
    q_all = np.vstack(qs)
    p_all = target_distribution(q_all)
    y_pred = q_all.argmax(1)
    results = clustering_score(y_true, y_pred)
    
    # early stop
    if results['NMI'] > nmi_best:
        model_best = copy.deepcopy(model)
        wait = 0
        nmi_best = results['NMI']
    else:
        wait += 1
        if wait > patient:
            model = model_best
            break
    
    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
    y_pred_last = np.copy(y_pred)
    if epoch > 0 and delta_label < 0.001:
        print(epoch, delta_label, 'break')
        break

    # Fine-tuning with auxiliary distribution
    model.train()
    tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
    qs = []
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, q = model(input_ids, segment_ids, input_mask, mode='finetune')
        kl_loss = F.kl_div(q.log(), torch.Tensor(p_all[step*train_batch_size: (step+1)*train_batch_size]).cuda())
        kl_loss.backward()

        tr_loss += kl_loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()        
    train_loss = tr_loss / nb_tr_steps
    results['kl_loss'] = round(train_loss, 4)
    results['delta_label'] = delta_label.round(4)
    print(epoch, results)


# Extracting probabilities Q
embs_test = []
qs = []
for batch in tqdm(test_dataloader, desc="Extracting probabilities Q"):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    with torch.no_grad():
        logits, q = model(input_ids, segment_ids, input_mask, mode='finetune')
    q = q.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    qs.append(q)
    embs_test.append(logits)

q_all = np.vstack(qs)
y_pred = q_all.argmax(1)
y_true = test_label_ids

results = clustering_score(y_true, y_pred)
results_all.update({'CDAC+': results})
print(results)
