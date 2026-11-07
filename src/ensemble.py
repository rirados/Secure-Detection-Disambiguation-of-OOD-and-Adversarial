import numpy as np
from sklearn.linear_model import LogisticRegression

def train_judge(X_clean, X_adv):
    X = np.vstack([X_clean, X_adv])
    y = np.hstack([np.ones(len(X_clean)), np.zeros(len(X_adv))])

    lr = LogisticRegression()
    lr.fit(X, y)

    return lr