import unittest
import numpy as np
import pandas as pd
from evaluation import forward_window, frozen_zscore, chronological_partitions

class EvaluationRegressionTests(unittest.TestCase):
    def test_forward_labels_exclude_today_and_require_full_horizon(self):
        s=pd.Series([99.,1.,2.,3.,4.])
        result=forward_window(s,2)
        self.assertEqual(result.iloc[:3].tolist(),[2.,3.,4.])
        self.assertTrue(result.iloc[-2:].isna().all())
    def test_forward_mean(self):
        self.assertEqual(forward_window(pd.Series([9.,1.,3.]),2,'mean').iloc[0],2.)
    def test_future_extremes_cannot_change_historical_transform(self):
        s=pd.Series([1.,2.,3.,4.],index=pd.date_range('2018-12-29',periods=4))
        expected=frozen_zscore(s)
        s.iloc[-1]=1e9
        np.testing.assert_array_equal(expected.iloc[:3],frozen_zscore(s).iloc[:3])
    def test_disjoint_purged_partitions(self):
        fit,val,cal=chronological_partitions(500,gap=30)
        self.assertEqual(val.start-fit.stop,30)
        self.assertEqual(cal.start-val.stop,30)
        self.assertEqual(cal.stop,500)
    def test_insufficient_history_rejected(self):
        with self.assertRaises(ValueError): chronological_partitions(10,gap=30)

class NotebookPipelineRegressionTests(unittest.TestCase):
    @staticmethod
    def definitions(name):
        import ast,json
        from pathlib import Path
        p=Path(__file__).resolve().parents[1]/'SCRI_SupplyChainRisk.ipynb'
        s='\n'.join(''.join(c['source']) for c in json.loads(p.read_text())['cells'] if c['cell_type']=='code')
        tree=ast.parse(s)
        module=ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name],type_ignores=[]))
        ns=dict(np=np,pd=pd,forward_window=forward_window,frozen_zscore=frozen_zscore)
        exec(compile(module,str(p),'exec'),ns);return ns[name]
    def test_alert_weights_spend_the_full_budget_once(self):
        from types import SimpleNamespace
        Model=lambda:SimpleNamespace(feature_name_=['x'],predict_proba=lambda x:np.array([[0.,1.]]))
        calibrator=SimpleNamespace(predict=lambda x:np.array(x))
        engine=self.definitions('SCRIAlertEngine')({h:Model() for h in [7,14,30]},
             {h:calibrator for h in [7,14,30]},pd.Series([100.],index=['today']))
        self.assertEqual(engine.score_day('today',pd.Series({'x':1.}))['ensemble_score'],1.)
    def test_full_scsi_history_unchanged_by_future_extreme(self):
        index=pd.bdate_range('2015-01-01','2020-01-31')
        df=pd.DataFrame({'vix':20+np.sin(np.arange(len(index))/20)},index=index)
        build=self.definitions('build_scsi'); expected=build(df)
        df.iloc[-1,0]=1e9;observed=build(df)
        np.testing.assert_allclose(expected.scsi.iloc[:-1],observed.scsi.iloc[:-1])
        self.assertTrue(observed.stress_event_30d.iloc[-30:].isna().all())
