import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
class ModelIAS(nn.Module):

    def __init__(self, bidirectional,dropout,hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, out_dropout=0.1):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        #bidirectional:
        if bidirectional:
          self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        else:
          self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)

        if bidirectional: self.slot_out = nn.Linear(hid_size*2, out_slot)
        else: self.slot_out = nn.Linear(hid_size, out_slot)

        if dropout:
          self.dropout = nn.Dropout(out_dropout)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):
        self.utt_encoder.flatten_parameters()  
        utt_emb = self.embedding(utterance)

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        #print("Dimensione dell'output LSTM prima del padding:", packed_output.data.size())

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # print("Dimensione dell'output LSTM dopo il padding:", utt_encoded.size())
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        #print("Dimensione dell'ultimo hidden state:", last_hidden.size())


        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        #print("Dimensione dell'output dei slot:", slots.size())
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        #print("Dimensione dell'output dell'intent:", intent.size())

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
