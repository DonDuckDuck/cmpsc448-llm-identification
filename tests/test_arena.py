"""Critical Arena invariants: attribution, leakage boundaries, and four-class outputs."""
import copy,sys,unittest,json,tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from arena_data import extract,mask
from arena_train import vocab_for,BASE,load
from models_v2 import make_model

def message(role,text):return {'role':role,'content':[{'type':'text','text':text,'image':None}]}
class ArenaTests(unittest.TestCase):
    def row(self):
        return {'id':'battle','model_a':'gpt-4.1-2025-04-14','language':'en','evaluation_order':1,
                'conversation_a':[message('user','Explain how rain forms.'),message('assistant','Water evaporates and then condenses into droplets that eventually fall as rain.')],
                'evaluation_session_id':'session','is_code':False,'timestamp':'2025-06-01'}
    def test_attribution_rejects_mixed_history(self):
        r=self.row();x,reason=extract(r,'a');self.assertIsNone(reason);self.assertEqual(x['family'],'GPT')
        r['evaluation_order']=2;self.assertIsNone(extract(r,'a')[0])
        r=self.row();r['conversation_a']+=[message('user','And snow?'),message('assistant','Another response')];self.assertIsNone(extract(r,'a')[0])
    def test_identity_and_sensitive_exclusion(self):
        text=mask('I am Claude Sonnet 4 by Anthropic, not GPT-4.1, Grok 3, or DeepSeek-V3.')
        for s in ['Anthropic','GPT','Grok','DeepSeek','Sonnet']:self.assertNotIn(s,text)
        self.assertEqual(mask('DeepSeek-V3-0324 and grok-3-preview-02-24'),'[MODEL] and [MODEL]')
        self.assertEqual(mask('Deepseek v3-0324 and DeepSeek V3-0324.'),'[MODEL] and [MODEL].')
        r=self.row();r['conversation_a'][0]['content'][0]['text']='Please contact person@example.com now';self.assertIsNone(extract(r,'a')[0])
    def test_vocab_uses_supplied_training_rows_only(self):
        train=[{'LLM_Input':'ordinary ordinary','LLM_output':'trainingword trainingword'}]
        v=vocab_for(train,'output',BASE);self.assertIn('trainingword',v);self.assertNotIn('heldoutword',v);self.assertNotIn('ordinary',v)
    def test_four_class_forward_and_padding(self):
        torch.manual_seed(10);torch.set_num_threads(2)
        for arch in ['cnn','lstm']:
            m=make_model(dict(BASE,architecture=arch),20).eval()
            a=torch.tensor([[2,3,4,5,6,7]]);b=torch.tensor([[2,3,4,5,6,7,0,0,0]])
            with torch.no_grad():x=m(a,torch.tensor([6]));y=m(b,torch.tensor([6]))
            self.assertEqual(tuple(x.shape),(1,4));torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    def test_jsonl_unicode_line_separator_is_not_a_record_boundary(self):
        row={'text':'first\u2028second\u0085third'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'train.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
            with patch('arena_train.DEST',root),patch('arena_train.ROOT',root):self.assertEqual(load('train'),[row])
if __name__=='__main__':unittest.main()
