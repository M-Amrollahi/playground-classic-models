from lazypredict.Supervised import LazyClassifier
from pycaret.classification import ClassificationExperiment
import numpy as np

class cls_model:
    def __init__(self) -> None:
        self.m_models =  None
        self.m_experiment = ClassificationExperiment()
        
        
        self.m_result = None

    def __len__(self):
        return len(self.m_model.models)
    
    def f_scores(self, data):

        self.m_experiment.setup(data[:,:2], target = data[:,-1], session_id = 123,memory=False,n_jobs=1,preprocess=False)
        
        self.m_models = self.m_experiment.compare_models(n_select=100,cross_validation=False)
        self.m_result = self.m_experiment.get_leaderboard()
        
        return self.m_result
    
    def f_iter_predict_range(self, XY):

        for model in self.m_models:
            Z = model.predict(XY)
            yield Z

