# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from torch.utils.data import DataLoader
import torch.optim as optim

class Parameters:
  '''
    here you can find the hyperparameters 
  '''
  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

  HID_SIZE = 350
  EMB_SIZE = 350
  LR = 0.0001
  CLIP = 5
  VOCAB_LEN = lambda x: len(x.word2id)
  N_EPOCHS = 100
  PATIENCE = 5
  BIDIRECTIONAL = False
  DRPOUT = True
  CRITERSION_SLOTS =nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
  CRITERSION_INTENTS = nn.CrossEntropyLoss()  # Because we do not have the pad token
  
  BATCH_SIZE_TRAIN  = 64
  BATCH_SIZE        = 128
  
  TRAIN = True
  TEST = True


'''
  these are the best parameters found that I decided to bring for each type
'''

parameter_sets = [
    {"BIDIRECTIONAL": False, "DROPOUT": False,"description": "Original"},
    {"BIDIRECTIONAL": True,  "DROPOUT": False,"description": "Bidirectional"},
    {"BIDIRECTIONAL": True,  "DROPOUT": True, "description": "Bidirectional_dropout"}
]

slot_met, intent_met = [[],[],[]], [[],[],[]]
final_intent_met = {}
final_slot_met = {}

if __name__ == "__main__":

    if Parameters.TRAIN:
      '''
          in this session there is the part of training and preparation of variables for the model
      '''
      for sample_params  in parameter_sets:
        params = {}
        for key, value in sample_params.items():
          params[key] = value

        intent2id,slot2id,w2id,train_raw,dev_raw,test_raw,lang = pre_preparation_train()
        # Create our datasets
        train_dataset = IntentsAndSlots(train_raw, lang)
        dev_dataset = IntentsAndSlots(dev_raw, lang)
        test_dataset = IntentsAndSlots(test_raw, lang)

        # Dataloader instantiations
        train_loader = DataLoader(train_dataset, batch_size=Parameters.BATCH_SIZE_TRAIN, collate_fn=collate_fn,  shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=Parameters.BATCH_SIZE, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=Parameters.BATCH_SIZE, collate_fn=collate_fn)

        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        vocab_len = len(lang.word2id)

        model = ModelIAS(params["BIDIRECTIONAL"],params["DROPOUT"],Parameters.HID_SIZE, out_slot, out_int, Parameters.EMB_SIZE, vocab_len,pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)
        #Add optimizer
        optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)

        sampled_epochs,losses_train,losses_dev,results_slot,results_intent,best_model = train_part(Parameters.PATIENCE,Parameters.N_EPOCHS,Parameters.CLIP,dev_loader,train_loader,test_loader,lang,optimizer,Parameters.CRITERSION_SLOTS,Parameters.CRITERSION_INTENTS,model)

        # save_model(best_model, sample_params["description"])
        # model_load=load_eval_model(Parameters.DEVICE,sample_params["description"])
        # model_load.eval()
        
        # eval_part(Parameters.CRITERSION_SLOTS,
        #       Parameters.CRITERSION_INTENTS,
        #       model_load,
        #       test_loader,lang)
        
        save_object(best_model,lang,test_loader,sample_params["description"])

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    
    if Parameters.TEST:   
      '''
          in this session there is the evaluation part of the model which allows to verify 
          the model towed previously and in the end convert the dictionary format of Intent values and slots for the next graphs
          for the final plot i should to use another code placed on plot_NLU
      '''
      for sample_params in parameter_sets:
        model, lang, test_loader = load_eval_object(Parameters.DEVICE, sample_params["description"])
          
        model.eval()
        
        slot_res, intent_res = eval_part(Parameters.CRITERSION_SLOTS,
                                          Parameters.CRITERSION_INTENTS,
                                          model,
                                          test_loader,
                                          lang,
                                          slot_met,
                                          intent_met
                                        )      

        final_intent_met[sample_params["description"]] = intent_res
        final_slot_met[sample_params["description"]] = slot_res

      param_sets_intent = []
      param_sets_slot = []
    

      convertion_dictionary (final_intent_met, param_sets_intent)
      convertion_dictionary (final_slot_met, param_sets_slot)
      