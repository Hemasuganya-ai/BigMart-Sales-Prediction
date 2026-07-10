# Part 3 — Ensembles, Tuning, and a Reproducible Pipeline

Builds on Part 2 above: same 'cleaned_data.csv', same feature encoding, same
'train_test_split(..., test_size=0.2, random_state=42)', same 'StandardScaler'
fit only on training data. 'modeling_part3.py' rebuilds that exact
'X_train_scaled' / 'X_test_scaled' / 'y_clf_train' / 'y_clf_test' at the top of
the script (identical code path, same random state) so it can run standalone.
'''
This produces all console output below, 'best_model.pkl', and 'results_part3.json'.

## Files in this repository

- 'part3_Ensenbling_Tuning.ipynb' — Part 3 pipeline (ensembles, CV, tuning, serialization); rebuilds Part 2's exact preprocessing at the top so it runs standalone.
- 'cleaned_data.csv' — input dataset.
- 'best_model.pkl' — serialized best pipeline (Imputer + Scaler + tuned RandomForestClassifier).
- 'results_part3.json' — machine-readable metric snapshots.

## 1. Decision Tree baseline (unconstrained)

| | Train accuracy | Test accuracy |
|---|---|---|
| Unconstrained ('max_depth=None') | 1.0000 | 0.7408 |

**Yes, this shows clear overfitting.** A perfect 100% training accuracy paired with a 26-point drop on the test set is the textbook signature of an overfit model — it has memorized the training rows rather than learned a generalizable pattern.

**Why decision trees are high-variance:** at each node, a decision tree greedily picks the single best split for the data *currently* in that node, then moves on and never revisits that choice. Left unconstrained, it keeps splitting until nodes are pure (or tiny), which lets it carve out a rule for almost every individual training example, including noise and one-off quirks. Because that structure is so tightly fit to the specific training sample, a different training sample (or the test set) produces a fairly different tree — hence "high variance."

## 2. Controlled Decision Tree ('max_depth=5', 'min_samples_split=20')

| | Train accuracy | Test accuracy | Train − test gap |
|---|---|---|---|
| Unconstrained | 1.0000 | 0.7408 | 0.2592 |
| Controlled (depth=5, min_split=20) | 0.7985 | 0.7818 | 0.0167 |

- **'max_depth'** caps how many sequential splits a path from root to leaf can have. Limiting it stops the tree from carving out ever-smaller, ever-more-specific regions of the feature space, which trades a bit of bias (the tree can express fewer patterns) for a large reduction in variance.
- **'min_samples_split'** blocks a node from splitting further if it has fewer than that many samples. This prevents the tree from creating splits that are only "discovered" because a handful of training rows happened to correlate by chance — small subsets are much more likely to contain noise than real structure.
- The controlled tree's train/test gap (0.0167) is **more than 15x smaller** than the unconstrained tree's gap (0.2592), and its test accuracy is actually *higher* (0.7818 vs 0.7408) despite lower train accuracy — a clean illustration of controlling variance improving generalization.

## 3. Gini vs Entropy

Both are impurity measures used to decide which split is "best" at a node, where 'p_i' is the proportion of samples in the node belonging to class 'i':

- **Gini impurity:** 'Gini = 1 - Σ pᵢ²'
- **Entropy:** 'Entropy = -Σ pᵢ log2(pᵢ)'

A node with **Gini = 0** is a **pure** node — every sample in it belongs to the same class, so there is nothing left to split on.

| Criterion (max_depth=5) | Test accuracy |
|---|---|
| Gini | 0.7818 |
| Entropy | 0.8088 |

Entropy edges out Gini by about 2.7 points of test accuracy on this run. In practice the two criteria usually produce very similar trees (they're both concave impurity measures that penalize mixed nodes), and the small gap here is within the range of ordinary variation from which exact splits get selected — not evidence that entropy is fundamentally superior for this dataset.

## 4. Random Forest

| | Train accuracy | Test accuracy | Test AUC |
|---|---|---|---|
| Random Forest (n_estimators=100, max_depth=10) | 0.8439 | 0.8111 | 0.8898 |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| 'Item_MRP' | 0.4880 |
| 'Item_Visibility' | 0.0619 |
| 'Outlet_Age' | 0.0614 |
| 'Outlet_Establishment_Year' | 0.0571 |
| 'Outlet_Type_Supermarket Type1' | 0.0544 |

**How Random Forest computes importance:** for each feature, the algorithm tracks how much every split that uses that feature reduces Gini impurity, weighted by how many samples pass through that split, then averages this reduction across every tree in the forest. A feature that is picked often, and produces big, reliable impurity reductions when it is picked, ends up with high importance.

**Why this differs from a linear regression coefficient:** a regression coefficient measures a feature's estimated *marginal linear* effect on the target, holding other features fixed — it has a sign and a size that map onto "one unit of this feature changes the prediction by this much." Random Forest importance has no sign and no direct unit interpretation; it only measures how *useful* that feature was for reducing impurity across many nonlinear, interaction-capturing splits. A feature could have zero linear correlation with the target (and thus a near-zero regression coefficient) but still be highly useful as part of a nonlinear split rule, and vice versa — so the two rankings are not directly comparable, only broadly complementary. It's worth noting here that 'Item_MRP' tops both rankings, which is a reassuring cross-check between the linear and ensemble models.

**Bagging (bootstrap aggregating), in one paragraph:** each tree in the forest is trained on its own bootstrap sample — a random sample of the same size as the training set, drawn *with* replacement, so on average each tree sees about 63% of the unique training rows (some repeated, some left out). On top of that, at every split each tree is only allowed to consider a random subset of √(number of features) candidate features, rather than all of them. Both mechanisms deliberately decorrelate the individual trees: since each tree sees a different sample of rows and a different subset of features at each split, their individual errors and quirks tend to be different from one tree to the next. Averaging (for classification, majority-voting or averaging predicted probabilities) across many such decorrelated trees cancels out much of that individual noise while keeping the genuine signal that most trees agree on, which is exactly why the forest's train/test gap (0.0328) is far smaller than the unconstrained single tree's gap (0.2592) — the ensemble trades a little bit of the single deep tree's raw training-set fit for a large reduction in variance.

### 4a. Gradient Boosting

| | Train accuracy | Test accuracy | Test AUC |
|---|---|---|---|
| Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3) | 0.8319 | 0.8147 | 0.8938 |

Unlike Random Forest's parallel, independent trees, Gradient Boosting builds trees **sequentially**, where each new (shallow, 'max_depth=3') tree is trained to correct the residual errors of the ensemble built so far, scaled down by the 'learning_rate'. This achieved a slightly higher test AUC than the Random Forest here (0.8938 vs 0.8898) while using shallower individual trees.

### 4b. Feature ablation study

**5 lowest-importance features (from the Task 4 Random Forest):**

| Feature | Importance |
|---|---|
| 'Item_Type_Others' | 0.00139 |
| 'Item_Type_Hard Drinks' | 0.00139 |
| 'Item_Type_Seafood' | 0.00138 |
| 'Outlet_Identifier_OUT049' | 0.00111 |
| 'Item_Type_Breakfast' | 0.00102 |

| Model | Test AUC |
|---|---|
| Full model (35 features) | 0.8898 |
| Reduced model (30 features, 5 dropped) | 0.8910 |
| Change | **+0.0012** |

The reduced model's AUC is essentially unchanged — if anything, marginally *higher* — after dropping these 5 features. This strongly suggests these particular one-hot indicators (a handful of rare 'Item_Type' categories and one specific outlet ID) really were close to uninformative noise for this Random Forest, rather than features whose removal cost real predictive power.

**Production trade-off:** this is a case where dropping the 5 lowest-importance features looks safely acceptable — the model gets simpler (fewer columns to maintain, validate, and serve at inference time) at essentially zero accuracy cost. In general this trade-off is only worth taking when the AUC change stays within whatever tolerance the business sets (e.g. "no more than a 0.005 AUC drop"); here we're comfortably inside that kind of tolerance, so a leaner, lower-dimensional model would be a reasonable production choice if inference cost or feature-pipeline maintenance is a real concern.

## 5. Cross-validated comparison

5-fold 'StratifiedKFold(shuffle=True, random_state=42)', 'scoring='roc_auc'', evaluated on 'X_train_scaled' / 'y_clf_train':

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.8966 | 0.0089 |
| Decision Tree (max_depth=5) | 0.8720 | 0.0215 |
| Random Forest | 0.8895 | 0.0087 |
| Gradient Boosting | 0.8964 | 0.0092 |

**Why cross-validation beats a single train/test split:** a single split gives you exactly one estimate of test performance, which depends heavily on which particular rows happened to land in the test set — an unlucky or lucky split can make a model look worse or better than it really is. K-fold cross-validation rotates every row through the test role exactly once across k folds, so the reported mean AUC averages over k different held-out samples, and the standard deviation across folds tells you how sensitive that estimate is to exactly which rows are held out. That combination — a more stable mean estimate, plus an explicit measure of its variability — is exactly the kind of "how confident am I in this claim" evidence a client asking about robustness wants to see. Note also that the Decision Tree has the largest std (0.0215), consistent with it being the highest-variance model discussed in Task 1/2.

## 6. GridSearchCV over a full sklearn Pipeline

Pipeline: 'make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))', fit on **unscaled** 'X_train' / 'y_clf_train' (the pipeline does its own imputation and scaling internally).

'''python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
'''

- Grid size: 3 × 3 × 2 = **18 hyperparameter configurations**, each evaluated across **5 folds** → **90 total model fits**.
- **Best params:** '{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}'
- **Best CV AUC score:** 0.8921
- Test-set AUC of this tuned pipeline (held-out 'X_test'): 0.8901

**Grid Search vs Randomized Search trade-off:** Grid Search exhaustively tries every combination in the grid, guaranteeing you find the best combination *within that grid*, but the number of fits grows multiplicatively with every hyperparameter and every value you add (here, 18 configs × 5 folds = 90 fits just for this modest 3-hyperparameter grid) — it becomes computationally prohibitive quickly. Randomized Search instead samples a fixed number of random combinations from the specified distributions, which lets you cover a much wider or finer-grained hyperparameter space for a fixed compute budget, at the cost of no longer guaranteeing that the single best combination in that space is actually tried. In practice, Randomized Search is generally preferred once a grid grows beyond a handful of hyperparameters, reserving exhaustive Grid Search for smaller, already-narrowed-down searches like this one.

## 7. Manual learning curve (best pipeline)

| Training fraction | n rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 1,363 | 0.9441 | 0.8798 |
| 0.4 | 2,727 | 0.9347 | 0.8864 |
| 0.6 | 4,090 | 0.9295 | 0.8898 |
| 0.8 | 5,454 | 0.9276 | 0.8902 |
| 1.0 | 6,818 | 0.9253 | 0.8901 |

- **(i) Training AUC decreases as the training set grows** (0.9441 → 0.9253). This is the expected pattern for a high-capacity model like a tuned Random Forest: with fewer rows, it can fit those specific rows almost perfectly (higher apparent training AUC), and as more, more-varied rows are added it can no longer fit all of them as tightly, so training AUC edges down toward the model's true generalization ceiling.
- **(ii) Test AUC rises with more data, but only up to a point** — it climbs from 0.8798 at 20% of the data to about 0.8901–0.8902 by 80–100%, then **flattens** between 80% and 100% (0.8902 → 0.8901, essentially no further gain).
- **(iii) Conclusion: the model is capacity-limited, not data-limited, at this point.** Test AUC has plateaued well before reaching 100% of the available training data — the last 20% of the data bought essentially zero additional test performance. That means simply collecting more rows of this same kind of data is unlikely to meaningfully improve this model further; further gains would more likely come from better features, more model complexity/capacity, or a different modeling approach, rather than more of the same data.

## 8. Serialized model

The best pipeline from Task 6 ('GridSearchCV.best_estimator_') was saved with:
'''python
joblib.dump(best_pipeline, 'best_model.pkl')
'''

Reload-and-predict block (included in 'modeling_part3.py', confirmed to run without errors):
'''python
import joblib

loaded_model = joblib.load('best_model.pkl')
hand_crafted_rows = X_test.iloc[[0, 1]].copy()  # two feature rows in the model's expected column format
reloaded_predictions = loaded_model.predict(hand_crafted_rows)
reloaded_probabilities = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
print(reloaded_predictions)       # [0 0]
print(reloaded_probabilities)     # [0.199 0.190]
'''
'best_model.pkl' is ~2 MB, well under the 100 MB limit, and is committed directly to the repository.

## 9. Final summary comparison table (Parts 2 & 3)

| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression (Part 2, C=1.0, 'class_weight='balanced'') | 0.8966 | 0.0089 | 0.8951 |
| Decision Tree (max_depth=5) | 0.8720 | 0.0215 | 0.8480 |
| Random Forest (n_estimators=100, max_depth=10) | 0.8895 | 0.0087 | 0.8898 |
| Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3) | 0.8964 | 0.0092 | 0.8938 |
| **Tuned RF Pipeline (GridSearchCV best)** | 0.8921 | *(single fit, no per-fold std reported)* | 0.8901 |

**Recommendation: Logistic Regression (C=1.0, 'class_weight='balanced'').** It posts the best or tied-best score on every axis that matters here — the highest 5-fold CV mean AUC (0.8966, essentially tied with Gradient Boosting's 0.8964), a cross-validation standard deviation as low as any ensemble candidate, and the highest test-set AUC (0.8951) of any model evaluated. It is also the simplest, fastest, and most interpretable model in the comparison — its coefficients (Part 2) can be directly explained to the client in plain language, unlike a forest's aggregated feature importances or a boosted ensemble's opaque sequence of correction trees. Given that the more complex models (Random Forest, Gradient Boosting, tuned RF pipeline) do not meaningfully outperform it on AUC, there's no accuracy trade-off being made by choosing the simpler, more transparent, and cheaper-to-serve model — a case where added model complexity buys essentially nothing on this dataset.


- 'README.md' — this file (documents both Part 2 and Part 3).

