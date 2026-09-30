import unittest,sys
from pathlib import Path
import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from models_v2 import PooledLSTM,SearchCNN,reverse_valid_prefix,masked_mean_max
class V2Invariants(unittest.TestCase):
    def test_reversal_preserves_padding_and_is_involution(self):
        x=torch.arange(14,dtype=torch.float32).reshape(2,7,1);lengths=torch.tensor([3,5])
        y=reverse_valid_prefix(x,lengths)
        torch.testing.assert_close(reverse_valid_prefix(y,lengths),x)
        self.assertEqual(y[0,:,0].tolist(),[2,1,0,3,4,5,6])
    def test_bidirectional_matches_packed_pytorch(self):
        torch.manual_seed(1);x=torch.randn(3,8,4);lengths=torch.tensor([8,5,2])
        reference=torch.nn.LSTM(4,3,bidirectional=True,batch_first=True).eval()
        forward=torch.nn.LSTM(4,3,batch_first=True).eval();backward=torch.nn.LSTM(4,3,batch_first=True).eval()
        forward.load_state_dict({k:v for k,v in reference.state_dict().items() if '_reverse' not in k})
        backward.load_state_dict({k.replace('_reverse',''):v for k,v in reference.state_dict().items() if '_reverse' in k})
        expected,_=reference(pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False));expected,_=pad_packed_sequence(expected,batch_first=True,total_length=8)
        f,_=forward(x);b,_=backward(reverse_valid_prefix(x,lengths));actual=torch.cat([f,reverse_valid_prefix(b,lengths)],2)
        mask=torch.arange(8)[None,:]<lengths[:,None]
        torch.testing.assert_close(actual[mask],expected[mask],atol=2e-6,rtol=2e-5)
    def test_padding_and_batch_composition_do_not_change_prediction(self):
        torch.manual_seed(2)
        for model in [SearchCNN(50),PooledLSTM(50),PooledLSTM(50,bidirectional=True),PooledLSTM(50,pooling='last')]:
            model.eval();short=torch.tensor([[2,3,4,5,6,7]]);lengths=torch.tensor([6])
            padded=torch.nn.functional.pad(short,(0,7))
            batch=torch.cat([padded,torch.tensor([[8]*13])]);batch_lengths=torch.tensor([6,13])
            with torch.inference_mode():
                torch.testing.assert_close(model(short,lengths)[0],model(batch,batch_lengths)[0],atol=2e-6,rtol=2e-5)
    def test_backward_direction_receives_gradient(self):
        model=PooledLSTM(20,bidirectional=True)
        loss=model(torch.tensor([[2,3,4,5,0]]),torch.tensor([4])).sum();loss.backward()
        self.assertGreater(model.backward_lstm.weight_ih_l0.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
