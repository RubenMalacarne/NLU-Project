# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
from torch.utils.data import DataLoader
import torch.optim as optim


class Parameters:

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    HID_SIZE = 200
    EMB_SIZE = 300

    LR = 0.00001  # learning rate
    CLIP = 5  # Clip the gradient

    OUT_SLOT = lambda x: len(x.slot2id)  # Number of output slot
    OUT_INT = lambda x: len(x.intent2id)  # Number of output intent
    VOCAB_LEN = lambda x: len(x.word2id)  # Vocabulary size

    CRITERSION_SLOTS = nn.CrossEntropyLoss(ignore_index=ParametersBert.PAD_TOKEN)
    CRITERSION_INTENTS = nn.CrossEntropyLoss()  # Because we do not have the pad token

    N_EPOCHS = 200 #100
    PATIENCE = 3 #7
    
    BATCH_SIZE_TRAIN  = 128
    BATCH_SIZE        = 64
    
    TEST               = True
    TRAIN              = False

slot_met, intent_met = [[],[],[]], [[],[],[]]
final_intent_met = {}
final_slot_met = {}

if __name__ == "__main__":
    
    if Parameters.TRAIN:
        '''
          in this session there is the part of training and preparation of variables for the model
        '''
        train_raw,dev_raw,test_raw,lang =pre_preparation_train()
        # Create our datasets
        train_dataset, dev_loader, test_dataset, lang =get_dataset(train_raw, dev_raw, test_raw)

        # Dataloader instantiations
        train_loader =  DataLoader(train_dataset, batch_size=Parameters.BATCH_SIZE_TRAIN, collate_fn=collate_fn,  shuffle=True)
        dev_loader =    DataLoader(dev_loader, batch_size=Parameters.BATCH_SIZE, collate_fn=collate_fn)
        test_loader =   DataLoader(test_dataset, batch_size=Parameters.BATCH_SIZE, collate_fn=collate_fn)

        out_slot =  len(lang.slot2id)
        out_int =   len(lang.intent2id)
        vocab_len = len(lang.word2id)

        model = BertCstm(Parameters.HID_SIZE, out_slot, out_int, Parameters.EMB_SIZE, vocab_len,pad_index=ParametersBert.PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)
        # if Parameters.TRAIN:
        sampled_epochs,losses_train,losses_dev,best_model = train_part(Parameters.N_EPOCHS,Parameters.CLIP,Parameters.PATIENCE,dev_loader,train_loader,test_loader,lang,optimizer,Parameters.CRITERSION_SLOTS,Parameters.CRITERSION_INTENTS,model)
        
        save_object(best_model,lang,test_loader, "bert")
        
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    if Parameters.TEST:  
        ''' 
          in this session there is the evaluation part of the model which allows to verify 
          the model towed previously and in the end convert the dictionary format of Intent values and slots for the next graphs
          for the final plot i should to use another code placed on plot_NLU
        '''
        model,lang,test_loader=load_eval_object(Parameters.DEVICE,"bert")
        model.eval()
        slot_res, intent_res = eval_part(Parameters.CRITERSION_SLOTS,
                                         Parameters.CRITERSION_INTENTS,
                                         model,
                                         test_loader,
                                         lang,
                                        slot_met,
                                        intent_met)
        
        final_intent_met = intent_res
        final_slot_met = slot_res

        param_sets_intent = []
        param_sets_slot = []
        
        intent_dict = {
            "F1_intent": final_intent_met[0],
            "P-intent": final_intent_met[1],
            "R-intent": final_intent_met[2],
            "description": "Bert"
        }

        param_sets_intent.append(intent_dict)
        print(param_sets_intent)
        
        slot_dict = {
            "F1_slot": final_slot_met[0],
            "P-slot": final_slot_met[1],
            "R-slot": final_slot_met[2],
            "description": "Bert"
        }
        param_sets_slot.append(slot_dict)
        print(param_sets_slot)