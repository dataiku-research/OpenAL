import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from bench.samplers import AutoEmbedder


X = np.random.random((100, 10))
y = np.random.choice(3, replace=True, size=100)

for model in [RandomForestClassifier(), GradientBoostingClassifier()]:
    model.fit(X, y)
    ae = AutoEmbedder(model, X=X)
    print(ae(X)[:10])


