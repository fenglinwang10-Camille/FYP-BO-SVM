"""Check recording tie rules, fold isolation and outer evaluation ordering."""
import unittest
from unittest.mock import patch
import numpy as np
import nested_score_tiebreak as experiment


class ScoreTieTests(unittest.TestCase):
    def test_tie_only_chooses_tied_classes_and_preserves_unique_winner(self):
        predictions=np.array([1,1,3,3,4, 1,1,1,3,4])
        scores=np.tile([0.,1.,0.,3.,100.],(10,1))
        groups=np.repeat([10,11],5)
        truth,voted,details=experiment.vote_with_scores(predictions,scores,np.repeat([3,1],5),groups)
        self.assertEqual(truth.tolist(),[3,1])
        self.assertEqual(voted.tolist(),[3,1])
        self.assertTrue(details[0]['changed_by_tiebreak'])
        self.assertFalse(details[1]['vote_tie'])

    def test_exact_score_tie_retains_lowest_index(self):
        _,voted,_=experiment.vote_with_scores(np.array([2,4]),np.zeros((2,5)),np.array([2,2]),np.array([7,7]))
        self.assertEqual(voted.tolist(),[2])

    def test_inconsistent_recording_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            experiment.vote_with_scores(np.array([1,1]),np.zeros((2,5)),np.array([1,2]),np.array([7,7]))

    def test_inner_fit_uses_only_each_saved_training_fold(self):
        ids=np.arange(10)
        groups=np.repeat(ids,2)
        labels=groups%5
        X=np.column_stack([groups,labels])
        folds=[(ids[ids%5!=i],ids[ids%5==i]) for i in range(5)]
        fitted=[]
        class Model:
            classes_=np.arange(5)
            def fit(self,features,target):
                fitted.append(sorted(set(features[:,0].astype(int))))
            def predict(self,features):
                return features[:,1].astype(int)
            def decision_function(self,features):
                return np.eye(5)[features[:,1].astype(int)]
        with patch.object(experiment.pipeline,'_build_feature_matrix',return_value=(X,labels,groups)), \
             patch.object(experiment.pipeline,'_make_model',side_effect=lambda params:Model()):
            score,_,_,_=experiment.cross_validate(None,None,ids,
                dict(f_low=.5,f_high=40,window_size=2,C=10,gamma=.1),fixed_folds=folds)
        self.assertEqual(score,1.)
        self.assertEqual(fitted,[a.tolist() for a,b in folds])

    def test_outer_features_are_transformed_after_training_fit(self):
        events=[]
        def build(signals,labels,ids,params):
            events.append('train_features' if ids[0]==0 else 'test_features')
            return np.array(labels).reshape(-1,1),np.array(labels),np.array(ids)
        class Model:
            classes_=np.arange(5)
            def fit(self,X,y):
                events.append('fit')
            def predict(self,X):
                return X[:,0].astype(int)
            def decision_function(self,X):
                return np.eye(5)[X[:,0].astype(int)]
        with patch.object(experiment.pipeline,'_build_feature_matrix',side_effect=build), \
             patch.object(experiment.pipeline,'_make_model',return_value=Model()):
            result=experiment.evaluate_outer((None,np.arange(5),np.arange(5)),
                (None,np.arange(5),np.arange(10,15)),{})
        self.assertEqual(events,['train_features','fit','test_features'])
        self.assertEqual(result['segment_level_test_metrics']['accuracy'],1.)
        self.assertEqual(len(result['recording_predictions']),5)


if __name__=='__main__':
    unittest.main()
