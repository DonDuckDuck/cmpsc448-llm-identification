"""Follow-up study invariants: stratified subsets and redirected fits leave the frozen study untouched."""
import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import arena_train
from arena_followup import subsample,redirected,as_domain,OUT

class FollowupTests(unittest.TestCase):
    def rows(self):
        return [{'row_id':f'{f}{i}','family':f} for f,n in [('Claude',40),('GPT',32),('Grok',24),('DeepSeek',16)] for i in range(n)]
    def test_subsample_is_stratified_deterministic_and_ordered(self):
        rows=self.rows();a=subsample(rows,.25,1448)
        self.assertEqual(Counter(r['family'] for r in a),{'Claude':10,'GPT':8,'Grok':6,'DeepSeek':4})
        self.assertEqual(a,subsample(rows,.25,1448))
        self.assertNotEqual(a,subsample(rows,.25,1449))
        order=[rows.index(r) for r in a];self.assertEqual(order,sorted(order))
    def test_redirect_restores_frozen_paths_and_loader(self):
        before=arena_train.RESULT,arena_train.CHECK,arena_train.load
        with redirected(OUT/'curve'/'unit_test_probe',[{'family':'GPT'}]):
            self.assertNotEqual(arena_train.RESULT,before[0]);self.assertEqual(arena_train.load('train'),[{'family':'GPT'}])
        self.assertEqual((arena_train.RESULT,arena_train.CHECK,arena_train.load),before)
        (OUT/'curve'/'unit_test_probe').rmdir();(arena_train.ROOT/'results/checkpoints/arena_followup/unit_test_probe').rmdir()
    def test_domain_switch_copies_rows_and_redirects_validation(self):
        rows=[{'row_id':'a','is_code':True},{'row_id':'b','is_code':False}]
        moved=as_domain(rows,[False,True])
        self.assertEqual([r['is_code'] for r in moved],[False,True]);self.assertEqual([r['is_code'] for r in rows],[True,False])
        with redirected(OUT/'curve'/'unit_test_probe',[{'family':'GPT'}],[{'family':'Grok'}]):
            self.assertEqual(arena_train.load('val'),[{'family':'Grok'}])
        (OUT/'curve'/'unit_test_probe').rmdir();(arena_train.ROOT/'results/checkpoints/arena_followup/unit_test_probe').rmdir()
if __name__=='__main__':unittest.main()
