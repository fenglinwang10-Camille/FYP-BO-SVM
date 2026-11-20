"""Check that increasing a saved search budget preserves its first candidates."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nested_bo_rs as base
from extend_nested_search import verify_and_fork


class SearchExtensionTests(unittest.TestCase):
    def test_continue_without_reinitializing_and_resume_exactly(self):
        old=dict(evaluations_per_method=4,shared_initial_evaluations=2,search_seed=42,xi=.01,
                 fixed_reference_parameters=base.MANUAL_PARAMS)
        new=dict(old,evaluations_per_method=6,training_only=True)
        split=base.make_splits(np.repeat(np.arange(5),100))[0]
        data=(None,None,np.array(split['train_segment_ids']))
        calls=[]
        def evaluate(train_data,folds,point,iteration,phase):
            calls.append(iteration)
            return dict(iteration=iteration,phase=phase,score=float(.7+.01*point[0]),
                        candidate_seconds=0.,**dict(zip(base.PARAMETERS,map(float,point))))
        with tempfile.TemporaryDirectory(dir=base.PROJECT_DIR/'tmp') as temp:
            source=Path(temp)/'source';source.mkdir()
            with patch.object(base,'_evaluate_candidate',side_effect=evaluate):
                saved,done=base.run_searches(data,split,old,source)
                self.assertTrue(done)
                source_X=[list(x) for x in saved['optimizer'].Xi]
                fork=verify_and_fork(saved,old,new,split)
                self.assertEqual(saved['budget'],4)
                self.assertEqual(fork['budget'],6)
                self.assertEqual(fork['optimizer'].Xi,source_X)
                continuous=Path(temp)/'continuous';continuous.mkdir()
                resumed=Path(temp)/'resumed';resumed.mkdir()
                base._save_state(continuous,fork)
                base._save_state(resumed,verify_and_fork(saved,old,new,split))
                calls.clear()
                full,done=base.run_searches(data,split,new,continuous)
                self.assertTrue(done)
                self.assertEqual(calls,[5,5,6,6])
                calls.clear()
                partial,done=base.run_searches(data,split,new,resumed,max_new=1)
                self.assertFalse(done)
                final,done=base.run_searches(data,split,new,resumed)
                self.assertTrue(done)
                self.assertEqual(calls,[5,5,6,6])
                self.assertEqual(final['optimizer'].Xi,full['optimizer'].Xi)
                self.assertEqual(final['shared'],saved['shared'])
                for method in ['bo','rs']:
                    self.assertEqual(final[method][:2],saved[method])
                    self.assertEqual([r['score'] for r in final[method]],
                                     [r['score'] for r in full[method]])
                self.assertEqual(saved['optimizer'].Xi,source_X)

    def test_wrong_source_signature_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_and_fork({'signature':'bad'},{},{},{})


if __name__=='__main__':
    unittest.main()
