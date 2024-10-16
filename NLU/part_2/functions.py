# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from   tqdm import tqdm
import copy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report
import torch.nn as nn

from transformers import BertTokenizer, BertModel

from utils import *
from conll import evaluate


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
class ParametersBert:
    
    TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    PAD_TOKEN = TOKENIZER.pad_token_id
    CLS_TOKEN = TOKENIZER.cls_token_id
    SEP_TOKEN = TOKENIZER.sep_token_id
    UNK_TOKEN = TOKENIZER.unk_token_id

def get_dataset(train_raw, dev_raw, test_raw):
    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
    # however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    val_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return train_dataset, val_dataset, test_dataset, lang

def pre_preparation_train():


    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    portion = 0.10
    download_dataset()
    tmp_train_raw,test_raw = set_dataset()
    train_raw,dev_raw,test_raw = set_develop_dataset(tmp_train_raw,test_raw)

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                                # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels,
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    return train_raw,dev_raw,test_raw,lang

def train_part(N_EPOCHS,CLIP,PATIENCE,dev_loader,train_loader,test_loader,lang,optimizer,CRITERSION_SLOTS,CRITERSION_INTENTS,model):
    '''
        Full training cycle of a deep learning model, including assessment cycles and early stopping mechanism
    '''
    print("START TRAINING")
    pOg= copy.deepcopy(PATIENCE)
    n_epochs = N_EPOCHS
    patience = PATIENCE
    losses_train = []
    losses_dev = []
    sampled_epochs = []

    best_f1 = 0

    best_model = None

    for x in tqdm(range(1,N_EPOCHS)):

        loss = train_loop(CLIP,
                          train_loader,
                        optimizer,
                        CRITERSION_SLOTS,
                        CRITERSION_INTENTS,
                        model,)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader,
                                                          CRITERSION_SLOTS,
                                                          CRITERSION_SLOTS,
                                                          model,
                                                          lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                patience = PATIENCE
                # best_model = copy.deepcopy(model)
                print ("find NEW BEST MODEL")
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, CRITERSION_SLOTS,
                                            CRITERSION_INTENTS, model, lang)
    best_model = copy.deepcopy(model)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])


    return sampled_epochs,losses_train,losses_dev,best_model

def eval_part(CRITERSION_SLOTS,CRITERSION_INTENTS, model,test_loader,lang,slot_res,intent_res):
    results_test, intent_test, _ = eval_loop(test_loader, CRITERSION_SLOTS, CRITERSION_INTENTS, model, lang)
    print("- Slot F1:", results_test["total"]["f"], end=" ")
    print("- Intent Accuracy:", intent_test["accuracy"])
    
    slot_res = [results_test['total']['f'], 
                results_test['total']['p'], 
                results_test['total']['r']]
    intent_res = [intent_test['weighted avg']['f1-score'], 
                  intent_test['weighted avg']['precision'], 
                  intent_test['weighted avg']['recall']]
    
    return slot_res,intent_res
   
def train_loop(CLIP,data, optimizer, criterion_slots, criterion_intents, model):
    '''
        Takes as input the training data, an optimizer, 
        two loss criteria (one for slots and one for intents), 
        the model, and a gradient clipping value (with a default value of 5)
    '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample["utterances"], sample["utt_mask"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()  # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["utt_mask"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        pass

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array

def save_model(best_model,name):
    '''
        Save model to load in a second moment and check the evaluation
    '''
    print ("salvataggio modello...")

    if not os.path.exists("model_pt"):
      os.makedirs("model_pt")
    torch.save(best_model, "model_pt/"+name+".pth")

def load_eval_model(DEVICE,name):
    '''
        load model to check the evaluation
    '''
    model_load = torch.load("model_pt/"+name+'.pth', map_location=DEVICE)
    model_load.eval()
    return model_load

def save_object(best_model,lang,test_loader,name):
    '''
        Save model and some variables  to load in a second moment and check the evaluation
    '''
    print ("salvataggio dizionario...")
    saving_obj = { 
                 "model": best_model, 
                "lang": lang,
                 "test_loader":test_loader}

    if not os.path.exists("model_pt"):
      os.makedirs("model_pt")
    torch.save(saving_obj, "model_pt/"+name+".pth")
    
    
def load_eval_object(DEVICE,name):
    '''
        load model to check the evaluation
    '''
    print("caricamento dizionario...")
    
    saving_obj =  torch.load("model_pt/"+name+'.pth', map_location=DEVICE)
    
    model           =saving_obj["model"]
    lang            =saving_obj["lang"]
    test_loader     =saving_obj["test_loader"]
    return model,lang,test_loader


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        #Dizionari inversi che mappano indicatori univoci alle loro rispettive parole slot e intenti
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    #Questo metodo prende una lista di parole e restituisce un dizionario che mappa ogni parola al suo identificatore univoco.

    def w2id(self, elements, cutoff=None, unk=True):
            vocab = {"pad": ParametersBert.PAD_TOKEN, "cls": ParametersBert.CLS_TOKEN, "sep": ParametersBert.SEP_TOKEN}
            if unk:
                vocab["unk"] = ParametersBert.UNK_TOKEN
            elements = set(elements)
            input_ids = ParametersBert.TOKENIZER.convert_tokens_to_ids(elements)
            for elem, input_id in zip(elements, input_ids):
                vocab[elem] = input_id

            return vocab

    def lab2id(self, elements, pad=True):

            vocab = {}
            if pad:
                vocab["pad"] = ParametersBert.PAD_TOKEN
            for elem in elements:
                vocab[elem] = len(vocab)
            return vocab


def collate_fn(data):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(
            ParametersBert.PAD_TOKEN
        )
        attention_mask = torch.LongTensor(len(sequences), max_len).fill_(0)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
            attention_mask[i, :end] = 1

        # print(padded_seqs)
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        return padded_seqs, attention_mask, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["utterance"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, utt_mask, _ = merge(new_item["utterance"])
    y_slots, _, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(
        DEVICE
    )  # We load the Tensor on our seleceted device
    utt_mask = utt_mask.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    new_item["utterances"] = src_utt
    new_item["utt_mask"] = utt_mask
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item
