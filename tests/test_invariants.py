"""Checks for leakage and padding bugs that can invalidate text experiments."""
import json,sys,unittest
from pathlib import Path
from collections import Counter,defaultdict
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from models import TextCNN,TextLSTM
from train import load_rows,make_vocab,TextData,collate,components
class Invariants(unittest.TestCase):
    def test_padding_does_not_change_logits(self):
        torch.manual_seed(448)
        for cls in (TextCNN,TextLSTM):
            model=cls(50).eval()
            short=torch.tensor([[2,3,4,5,6,7]])
            padded=torch.nn.functional.pad(short,(0,9))
            lengths=torch.tensor([6])
            with torch.inference_mode():
                torch.testing.assert_close(model(short,lengths),model(padded,lengths),atol=2e-6,rtol=2e-5)
    def test_group_disjoint_and_prompt_balanced(self):
        root=Path(__file__).resolve().parents[1]
        # A fresh Arena clone still includes the historical non-text manifest.
        if all((root/f'data/processed/{s}.jsonl').exists() for s in ['train','val','test']):
            splits={s:load_rows(s) for s in ['train','val','test']}
        else:
            manifest=json.loads((root/'data/processed/split_manifest.json').read_text())
            splits={s:[r for r in manifest if r['split']==s] for s in ['train','val','test']}
        for s,rows in splits.items():
            groups=defaultdict(list)
            for row in rows:groups[row['prompt_id']].append(row)
            for rows_for_prompt in groups.values():
                self.assertEqual(Counter(r['family'] for r in rows_for_prompt),{'Falcon':1,'GPT':1,'Llama':1})
                if 'LLM_Input' in rows_for_prompt[0]:
                    self.assertEqual(len(set(r['LLM_Input'] for r in rows_for_prompt)),1)
            for other in splits:
                if other!=s:
                    self.assertFalse(set(r['duplicate_group'] for r in rows)&set(r['duplicate_group'] for r in splits[other]))
    def test_vocab_cannot_see_heldout_tokens(self):
        row={'LLM_Input':'trainword trainword','LLM_output':'answer answer','family':'GPT'}
        vocab=make_vocab([row],'combined')
        held=dict(row,LLM_output='never_seen_token')
        self.assertNotIn('never_seen_token',vocab)
        ids,_=TextData([held],vocab,'output')[0]
        self.assertEqual(ids,[1])
    def test_combined_reserves_response_budget(self):
        row={'LLM_Input':'question '*300,'LLM_output':'answer '*300}
        tokens=components(row,'combined')
        self.assertEqual(tokens.count('question'),96)
        self.assertEqual(tokens.count('answer'),192)
        self.assertIn('[sep]',tokens)
if __name__=='__main__':unittest.main()
