from functions import *
import matplotlib.pyplot as plt

class Parameters:
    '''
      here you can find the hyperparameters 
    '''
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    SGD = False
    ADAM = False
    RNN = False

    VOCAB_LEN = lambda x: len(x.word2id)
    CLIP = 5
    N_EPOCHS = 100

    HID_SIZE = 350
    EMB_SIZE = 350
    LR = 0.00001

    PATIENCE = 5

    TRAINING = False
    EVALUATION = True
    
    BATCH_SIZE_TRAIN  = 16
    BATCH_SIZE        = 128

'''
  these are the best parameters found that I decided to bring for each type
'''
parameter_sets = [
    {"SGD": True,  "ADAM": False,"LR": 0.3, "BATCH_SIZE": 128,"RNN": True, "description": "RNN_SGD"},
    {"SGD": False, "ADAM": True, "LR": 0.0001,"BATCH_SIZE":128,   "RNN": True, "description": "RNN_ADAM"},
    {"SGD": True, "ADAM": False, "LR": 0.8,"BATCH_SIZE":128,"RNN": False,"description": "LSTM_SGD"},
    {"SGD": False, "ADAM": True, "LR": 0.0001,"BATCH_SIZE":128,"RNN": False,"description": "LSTM_ADAM"}
]

#all result for each parameter_sets
final_ppl_dict = {}

if __name__ == "__main__":


  for sample_params  in parameter_sets:
    params = {}
    for key, value in sample_params.items():
      params[key] = value
      
    train_loader,dev_loader,test_loader,model,optimizer,criterion_train,criterion_eval= pre_preparation_train(params["RNN"],
                                            Parameters.EMB_SIZE,
                                            Parameters.HID_SIZE,
                                            Parameters.VOCAB_LEN,
                                            Parameters.DEVICE,
                                            params["LR"],
                                            params["SGD"],
                                            params["ADAM"],
                                            Parameters.N_EPOCHS,
                                            Parameters.PATIENCE,
                                            Parameters.CLIP,
                                            params["BATCH_SIZE"],
                                            Parameters.BATCH_SIZE_TRAIN)
    
    if Parameters.TRAINING:
      final_ppl_dict[sample_params["description"]],best_model=train_part(Parameters.TRAINING,
                                                              Parameters.PATIENCE,
                                                              Parameters.N_EPOCHS,
                                                              Parameters.CLIP,
                                                              Parameters.DEVICE,
                                                              train_loader,
                                                              dev_loader,
                                                              test_loader,
                                                              model,
                                                              optimizer,
                                                              criterion_train,
                                                              criterion_eval)
      
      save_model(best_model,sample_params["description"])

      torch.cuda.empty_cache()
  if Parameters.EVALUATION:
    for sample_params  in parameter_sets:
        final_ppl_dict[sample_params["description"]] = eval_part(Parameters.EVALUATION,
                                                                  test_loader, 
                                                                  criterion_eval, 
                                                                  load_eval_model(Parameters.DEVICE,sample_params["description"]))
    plot_result(final_ppl_dict)
