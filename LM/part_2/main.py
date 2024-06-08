from functions import *

class Parameters:  
    '''
      here you can find the hyperparameters 
    '''
    #Parameter
    BOOL_WAIT_TYINING = True
    BOOL_VD = True
    BOOL_NTASGD = False

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    HID_SIZE = 256
    EMB_SIZE = 256

    LR = 0.001
    CLIP = 5

    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 100
    PATIENCE = 10

    TRAINING = False
    EVALUATION = True

    BATCH_SIZE_TRAIN  = 256#16
    BATCH_SIZE        = 1024#128

final_ppl_dict = {}

'''
  these are the best parameters found that I decided to bring for each type
'''
# parameter_sets = [      1
#     {"BOOL_WAIT_TYINING": False,"BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.001,"description": "LSTM"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.0001,"description": "LSTM+WT"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": False,"LR": 0.0001,"description": "LSTM+WT+VD"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": True, "LR": 0.95,"description": "LSTM+VD+WT+NTASG"}
# ]

# parameter_sets = [      2
#     {"BOOL_WAIT_TYINING": False,"BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.0001,"description": "LSTM"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.0001,"description": "LSTM+WT"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": False,"LR": 0.0001,"description": "LSTM+WT+VD"},
#     {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": True, "LR": 0.95,"description": "LSTM+VD+WT+NTASG"}
# ]

parameter_sets = [
    {"BOOL_WAIT_TYINING": False,"BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.001,"BATCH_SIZE_TRAIN" : 256 ,"BATCH_SIZE" : 1024,"description": "LSTM"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.001,"BATCH_SIZE_TRAIN" : 256 ,"BATCH_SIZE" : 1024,"description": "LSTM+WT"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": False,"LR": 0.001,"BATCH_SIZE_TRAIN" : 256 ,"BATCH_SIZE" : 1024,"description": "LSTM+WT+VD"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": True, "LR": 0.9,  "BATCH_SIZE_TRAIN" : 16 ,"BATCH_SIZE" : 1024 ,"description": "LSTM+VD+WT+NTASG"}
] 
if __name__ == "__main__":


    for sample_params  in parameter_sets:
      params = {}
      for key, value in sample_params.items():
        params[key] = value
      
      train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval,lang= pre_preparation_train(params["BOOL_NTASGD"],
                                                                                                                Parameters.EMB_SIZE,
                                                                                                                Parameters.HID_SIZE,
                                                                                                                Parameters.VOCAB_LEN,
                                                                                                                params["LR"],
                                                                                                                Parameters.DEVICE,
                                                                                                                  params["BOOL_WAIT_TYINING"],
                                                                                                                  params["BOOL_VD"],
                                                                                                                  params["BATCH_SIZE"],
                                                                                                                  params["BATCH_SIZE_TRAIN"])
      restemp=[]

      if Parameters.TRAINING:
        model.apply(init_weights)
        if params["BOOL_NTASGD"]:
          #OPTIMIZER
          optimizer =  optim.SGD(model.parameters(), lr=params["LR"], weight_decay=1.2e-6) #same config as source code

          config = {
                      'n': 5,  # Non-monotone interval
                      'lr':params["LR"],
                      'logs':[] #list to store validation losses
                      }

          best_model,final_ppl_dict[sample_params["description"]] = train_part(
                                          Parameters.TRAINING,
                                          Parameters.N_EPOCHS,
                                          Parameters.DEVICE,
                                          Parameters.CLIP,
                                          Parameters.PATIENCE,
                                          params["BOOL_NTASGD"],
                                          optimizer,
                                          model,
                                          train_loader,
                                          dev_loader,
                                          test_loader,
                                          criterion_eval,
                                          criterion_train,
                                          config=config)
        else:
          best_model,final_ppl_dict[sample_params["description"]] = train_part(
                                          Parameters.TRAINING,
                                          Parameters.N_EPOCHS,
                                          Parameters.DEVICE,
                                          Parameters.CLIP,
                                          Parameters.PATIENCE,
                                          params["BOOL_NTASGD"],
                                          optimizer,
                                          model,
                                          train_loader,
                                          dev_loader,
                                          test_loader,
                                          criterion_eval,
                                          criterion_train)

        save_model(best_model,sample_params["description"])
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
    if Parameters.EVALUATION:
      for sample_params  in parameter_sets:
          #evaluation part
          final_ppl_dict[sample_params["description"]] = eval_part(Parameters.EVALUATION,
                    test_loader,
                    criterion_eval,
                    load_eval_model(Parameters.DEVICE,sample_params["description"]))
      plot_result(final_ppl_dict)
