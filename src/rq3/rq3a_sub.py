"""rq3a_tournament.py restricted to a subset of models (all files: random forest, CatBoost, XGBoost). Usage: python rq3a_sub.py <scripts_dir> <Model,Model,...> [tournament args]"""
import sys
sys.path.insert(0, sys.argv[1]); keep = set(sys.argv[2].split(','))
import rq3a_tournament as T
_g = T.get_models
T.get_models = lambda: {k: v for k, v in _g().items() if k in keep}
sys.argv = ['rq3a_tournament.py'] + sys.argv[3:]
T.main()
