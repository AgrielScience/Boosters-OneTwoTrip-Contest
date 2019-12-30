import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


def calc_smooth_mean(df, by, on, weight):
    mean = df[on].mean()
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + weight * mean) / (counts + weight)
    return df[by].map(smooth)


def add_mean_target_encoding(merged_df, fields, weight, target):
    df_train = merged_df[merged_df.type == 'train']
    df_test = merged_df[merged_df.type == 'test']
    for field in fields:
        df_train[f'{field}_m'] = calc_smooth_mean(df_train, by=field, on=target, weight=weight)
        df_test = pd.merge(df_test, df_train[[field, f'{field}_m']].drop_duplicates(), how='left', on=field)
    return pd.concat([df_train, df_test], axis=0, sort=False)


def add_stat_features(main_features, sub_features, df):
    merged_df = df.copy()
    for field in main_features:
        for sub_field in sub_features:
            sub_var_mean = merged_df.groupby([field])[
                sub_field].mean().reset_index().rename(
                columns={sub_field: f"mean_{field}_{sub_field}"}).fillna(0)
            sub_var_median = merged_df.groupby([field])[
                sub_field].median().reset_index().rename(
                columns={sub_field: f"med_{field}_{sub_field}"}).fillna(0)
            sub_var_std = merged_df.groupby([field])[
                sub_field].std().reset_index().rename(
                columns={sub_field: f"std_{field}_{sub_field}"}).fillna(0)
            merged_df = pd.merge(merged_df, sub_var_mean, how='left', on=field)
            merged_df = pd.merge(merged_df, sub_var_median, how='left',
                                 on=field)
            merged_df = pd.merge(merged_df, sub_var_std, how='left', on=field)
        print("Finish ", field)
    return merged_df


def add_smpl_features(features, merged_df):
    for field in features:
        cnt = merged_df.groupby([field]).size().reset_index().rename(
            columns={0: f"cnt_{field}"})
        merged_df = pd.merge(merged_df, cnt, how='left', on=field)
        merged_df[f'log_{field}'] = np.log(merged_df[field]).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
    print("Finish ", field)
    return merged_df


if __name__ == "__main__":
    df_train = pd.read_csv('data/onetwotrip_challenge_train.csv')
    df_train['type'] = 'train'
    df_test = pd.read_csv('data/onetwotrip_challenge_test.csv')
    df_test['type'] = 'test'
    print("Shape of train data: ", df_train.shape)
    print("Shape of test data: ", df_test.shape)

    features = list(filter(lambda x: 'field' in x, df_train.columns))
    merged_df = pd.concat([df_train, df_test], axis=0, sort=False)
    print("Shape of merged data: ", merged_df.shape)
    main_features = ['field16', 'field1', 'field12', 'field25', 'field14',
                     'field22', 'field17', 'field13', 'field0', 'field8']
    group_features = [x for x in features if x not in main_features]
    merged_df = add_stat_features(main_features, group_features, merged_df)
    merged_df = add_smpl_features(features, merged_df)
    features = list(filter(lambda x: 'field' in x, merged_df.columns))
    df_train = merged_df[merged_df.type == "train"]
    df_test = merged_df[merged_df.type == "test"]

    kfolds = KFold(n_splits=5, shuffle=True, random_state=236)
    X = df_train[features]
    y = df_train['goal1']

    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(df_test.shape[0])

    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        # X_train, y_train = augment(X_train.values, y_train.values)

        print(f"Fold idx: {n_fold + 1}")

        model = cb.CatBoostClassifier(
            bagging_temperature=0.2,
            # depth = 5,
            # bootstrap_type='Bernoulli',
            od_wait=20,
            scale_pos_weight=44,
            subsample=0.36,
            custom_loss='Logloss',
            random_strength=0,
            max_depth=3,
            eval_metric="AUC",
            learning_rate=0.03,
            iterations=60000,
            l2_leaf_reg=0.3,
            random_seed=432013,
            od_type="Iter",
            border_count=128
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=1500,
            early_stopping_rounds=100,
            use_best_model=True
        )

        oof_preds[val_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds = model.predict_proba(df_test[features])[:, 1]
        sub_preds += test_preds / kfolds.n_splits

    print("ROC_AUC score: ", roc_auc_score(y, oof_preds))
    now = datetime.now()
    pd.DataFrame(sub_preds, columns=['proba'],
                 index=df_test['orderid']).to_csv(
        f'results/sub-{str(now)[:19]}-{round(roc_auc_score(y, oof_preds),4)}.csv')
