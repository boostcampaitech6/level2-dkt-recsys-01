import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import os
from lgbm.lgbm import LGBM_CONFIG


def train_valid(data):
    model = lgb.train(
    LGBM_CONFIG,
    data["lgb_train"],
    valid_sets=[data["lgb_train"], data["lgb_test"]],
    num_boost_round=15)
    
    best_iter = model.best_iteration
    preds = model.predict(data["train"], num_iteration=best_iter)
    acc = accuracy_score(data["y_train"], np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(data["y_train"], preds)
    print(f'TRAIN AUC : {auc} ACC : {acc}\n')
    
    
    preds = model.predict(data["test"], num_iteration=best_iter)
    acc = accuracy_score(data["y_test"], np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(data["y_test"], preds)
    print(f'VALID AUC : {auc} ACC : {acc}\n')
    return model

def inference(file_name, testdata, model):
    total_preds = model.predict(testdata)

    # SAVE OUTPUT
    output_dir = 'output/'
    write_path = os.path.join(output_dir, f"{file_name}_submission.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
            
