# -*- coding: utf-8 -*-
import time

start_time = time.perf_counter()
print("Notebook timing started...")

"""
Flexible Decision Tree + GridSearchCV for Sleep Disorder Classification

Key improvement:
- This version does NOT require a fixed list of feature columns.
- It automatically uses all columns except the target column as input features.
- Therefore, different datasets can have different feature columns, such as:
  Dataset A: includes Heart Rate
  Dataset B: includes Stress Level
- You only need to change DATA_PATH.

The target column is still expected to be "Sleep Disorder" unless you add more aliases
in TARGET_CANDIDATES below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline


# ===================== Basic Settings =====================
SEED = 42

# Only change this path when you use another dataset.
# Examples:
# DATA_PATH = r"D:\study\ELEC2112\finalcombined_sleep_dataset.csv"
# DATA_PATH = r"D:\study\ELEC2112\finalcombined_sleep_dataset2.0.xlsx"
DATA_PATH = r"D:\study\ELEC2112\combined_sleep_dataset.csv"

# The code will automatically find the target column using these possible names.
TARGET_CANDIDATES = [
    "Sleep Disorder",
    "sleep disorder",
    "Sleep_Disorder",
    "sleep_disorder",
    "target",
    "Target",
    "label",
    "Label",
    "class",
    "Class"
]

# Optional: columns that should never be used as features if they exist.
DROP_COLUMNS_IF_EXIST = [
    "Unnamed: 0",
    "index",
    "Index",
    "ID",
    "Id",
    "id"
]


# ===================== Helper Functions =====================
def load_dataset(data_path):
    """
    Automatically read CSV or Excel files based on the file extension.
    This means you only need to change DATA_PATH.
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(data_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(data_path, engine="openpyxl")
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Please use .csv, .xlsx, or .xls."
        )


def find_target_column(df, target_candidates):
    """
    Find the target column from possible names.
    """
    for col in target_candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No target column found. "
        f"Available columns are: {df.columns.tolist()}. "
        f"Expected one of: {target_candidates}"
    )


def clean_dataset_columns(df):
    """
    Clean column names and remove unnecessary index-like columns.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Drop columns like Unnamed: 0, index, ID if they exist.
    cols_to_drop = [
        col for col in df.columns
        if col in DROP_COLUMNS_IF_EXIST or col.startswith("Unnamed")
    ]

    if cols_to_drop:
        print("\nDropped unnecessary columns:")
        print(cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def make_onehot_encoder():
    """
    Create a OneHotEncoder that works with both new and older sklearn versions.
    """
    try:
        return OneHotEncoder(
            sparse_output=False,
            drop="first",
            handle_unknown="ignore"
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            drop="first",
            handle_unknown="ignore"
        )


def build_preprocessor(categorical_cols, numerical_cols):
    """
    Build preprocessing pipeline for any dataset column structure.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", make_onehot_encoder(), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def build_dt_pipeline(
    categorical_cols,
    numerical_cols,
    max_depth=3,
    min_samples_leaf=3,
    min_samples_split=2,
    criterion="gini"
):
    """
    Build a complete Decision Tree pipeline:
    preprocessing + DecisionTreeClassifier.
    """
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=SEED
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf)
    ])


def encoded_to_raw_feature(encoded_name, raw_features):
    """
    Map an encoded one-hot feature back to the original raw input variable.
    """
    for raw in sorted(raw_features, key=len, reverse=True):
        if encoded_name == raw or encoded_name.startswith(raw + "_"):
            return raw
    return encoded_name


def save_table_as_png(df, filename, title=None, max_rows=20):
    """
    Save a DataFrame as a PNG table.
    This is useful for inserting GridSearchCV results into a report.
    """
    df_to_show = df.head(max_rows).copy()

    fig_height = max(2.5, 0.35 * len(df_to_show) + 1.2)
    fig_width = max(12, 1.6 * len(df_to_show.columns))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=df_to_show.values,
        colLabels=df_to_show.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



def plot_train_test_accuracy_overlap(
    train_acc,
    test_acc,
    model_name="Decision Tree",
    save_path="decision_tree_train_test_accuracy_overlap.png"
):
    """
    Draw a bar chart similar to the Random Forest example:
    Training accuracy and testing accuracy are shown as overlapping bars.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Overlapping bars: train bar behind, test bar slightly shifted in front.
    ax.bar(
        [0.00],
        [train_acc],
        width=0.65,
        label="Train Accuracy",
        alpha=0.95
    )

    ax.bar(
        [0.28],
        [test_acc],
        width=0.65,
        label="Test Accuracy",
        alpha=0.95
    )

    ax.set_title(f"{model_name}: Training vs Testing Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)

    # Put the model name in the middle of the two overlapping bars.
    ax.set_xticks([0.14])
    ax.set_xticklabels([model_name])

    ax.legend(loc="upper right")

    # Leave space at the bottom for printed accuracy values.
    plt.subplots_adjust(bottom=0.22)

    fig.text(0.08, 0.08, f"Train Accuracy: {train_acc}", fontsize=10)
    fig.text(0.08, 0.04, f"Test Accuracy: {test_acc}", fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nTraining vs Testing Accuracy plot saved as: {save_path}")

def make_cv_splitter(y, max_splits=10):
    """
    Build a safe StratifiedKFold splitter.
    If the smallest class has fewer than 10 samples, reduce n_splits automatically.
    """
    min_class_count = y.value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples. "
            "Stratified train/test split and cross-validation cannot be performed safely."
        )

    n_splits = min(max_splits, int(min_class_count))
    n_splits = max(2, n_splits)

    if n_splits < max_splits:
        print(f"\nWarning: n_splits was reduced from {max_splits} to {n_splits} because one class has only {min_class_count} samples.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )


# ===================== 【1】Load Dataset =====================
df = load_dataset(DATA_PATH)
df = clean_dataset_columns(df)

print("当前数据集列名：")
print(df.columns.tolist())

target = find_target_column(df, TARGET_CANDIDATES)

# Automatically use all columns except the target column as input features.
feature_cols = [col for col in df.columns if col != target]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found. The dataset only contains the target column.")

# Keep only feature columns + target
df = df[feature_cols + [target]].copy()

# Drop missing values
before_drop = df.shape[0]
df = df.dropna()
after_drop = df.shape[0]

print("\n===================== Dataset Overview =====================")
print("Shape after dropping missing values:", df.shape)
print(f"Rows dropped due to missing values: {before_drop - after_drop}")

print("\nTarget column:")
print(target)

print("\nAutomatically selected input features:")
print(feature_cols)
print(f"Number of input features: {len(feature_cols)}")

print("\n睡眠障碍类别分布:")
print(df[target].value_counts())

print("\n前5行数据:")
print(df.head())


# ===================== 【2】Define X and y =====================
X = df[feature_cols].copy()
y = df[target].copy()

# More robust dtype selection:
# numerical columns = all numeric columns
# categorical columns = all remaining columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numerical_cols]

sleep_labels = sorted(y.unique().tolist())

print("\n===================== Features =====================")
print("Raw input variables:")
print(feature_cols)
print(f"Number of raw input variables: {len(feature_cols)}")

print("\nCategorical features:")
print(categorical_cols)

print("\nNumerical features:")
print(numerical_cols)

print("\nTarget classes:")
print(sleep_labels)


# ===================== 【3】80/20 Train-Test Split =====================
min_class_count = y.value_counts().min()

stratify_target = y if min_class_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_target
)

print("\n===================== 80/20 Train-Test Split =====================")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ===================== 【4】10-fold CV Setting =====================
cv10 = make_cv_splitter(y_train, max_splits=10)


# ===================== 【5】Baseline Decision Tree, depth=3 =====================
# This is kept only for comparison with the GridSearchCV best model.
baseline_pipeline = build_dt_pipeline(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    criterion="gini",
    max_depth=3,
    min_samples_leaf=3,
    min_samples_split=2
)

baseline_pipeline.fit(X_train, y_train)
baseline_pred_test = baseline_pipeline.predict(X_test)

baseline_test_report = classification_report(
    y_test,
    baseline_pred_test,
    labels=sleep_labels,
    target_names=sleep_labels,
    output_dict=True,
    zero_division=0
)

baseline_cv_res = cross_validate(
    baseline_pipeline,
    X_train,
    y_train,
    cv=cv10,
    scoring="accuracy"
)

baseline_cv_mean = baseline_cv_res["test_score"].mean()
baseline_cv_std = baseline_cv_res["test_score"].std()
baseline_stable = "Yes" if baseline_cv_std < 0.05 else "No — high variance"

print("\n===================== Baseline Decision Tree Results =====================")
print(pd.DataFrame([{
    "Model": "Baseline Decision Tree (depth=3)",
    "CV Mean Accuracy": round(baseline_cv_mean, 4),
    "CV Std": round(baseline_cv_std, 4),
    "Stable?": baseline_stable,
    "Final Test Accuracy": round(accuracy_score(y_test, baseline_pred_test), 4),
    "Final Test Macro F1": round(baseline_test_report["macro avg"]["f1-score"], 4),
    "Final Test Precision": round(baseline_test_report["macro avg"]["precision"], 4),
    "Final Test Recall": round(baseline_test_report["macro avg"]["recall"], 4),
}]).to_string(index=False))


# ===================== 【6】GridSearchCV: All Hyperparameter Combinations =====================
# This section meets the requirement:
# "Give me a table of all the hyperparameter value combinations and their accuracy using grid_cv_results".
#
# Note:
# n_estimators is NOT included because this is a single Decision Tree model,
# not a Random Forest model.
print("\n===================== GridSearchCV Hyperparameter Results =====================")

grid_pipe = build_dt_pipeline(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols
)

# Hyperparameter search space for Decision Tree
param_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [2, 3, 4, 5, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 3, 5, 10]
}

grid_cv = GridSearchCV(
    estimator=grid_pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv10,
    n_jobs=1,   # Use single process to avoid Windows Unicode path error
    return_train_score=True,
    refit=True  # This makes grid_cv.best_estimator_ fitted on the full training set
)

grid_cv.fit(X_train, y_train)

# Convert cv_results_ into a readable table
grid_cv_results = pd.DataFrame(grid_cv.cv_results_)

grid_cv_results_table = grid_cv_results[[
    "param_clf__criterion",
    "param_clf__max_depth",
    "param_clf__min_samples_split",
    "param_clf__min_samples_leaf",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "rank_test_score"
]].copy()

grid_cv_results_table = grid_cv_results_table.rename(columns={
    "param_clf__criterion": "criterion",
    "param_clf__max_depth": "max_depth",
    "param_clf__min_samples_split": "min_samples_split",
    "param_clf__min_samples_leaf": "min_samples_leaf",
    "mean_test_score": "CV Mean Accuracy",
    "std_test_score": "CV Std",
    "mean_train_score": "Train Mean Accuracy",
    "rank_test_score": "Rank"
})

grid_cv_results_table.insert(0, "Combination No.", range(1, len(grid_cv_results_table) + 1))

# Round values for report readability
grid_cv_results_table["CV Mean Accuracy"] = grid_cv_results_table["CV Mean Accuracy"].round(4)
grid_cv_results_table["CV Std"] = grid_cv_results_table["CV Std"].round(4)
grid_cv_results_table["Train Mean Accuracy"] = grid_cv_results_table["Train Mean Accuracy"].round(4)

# Sort by best cross-validation accuracy
grid_cv_results_table = grid_cv_results_table.sort_values(
    by=["Rank", "CV Mean Accuracy"],
    ascending=[True, False]
)

print("\nAll hyperparameter value combinations and their accuracy:")
print(grid_cv_results_table.to_string(index=False))

print("\nBest Hyperparameters:")
print(grid_cv.best_params_)

print("\nBest CV Accuracy:")
print(round(grid_cv.best_score_, 4))

# Save complete GridSearchCV results table
grid_cv_results_table.to_csv(
    "decision_tree_grid_cv_results.csv",
    index=False
)

grid_cv_results_table.to_html(
    "decision_tree_grid_cv_results.html",
    index=False
)

with open("decision_tree_grid_cv_results.txt", "w", encoding="utf-8") as f:
    f.write(grid_cv_results_table.to_string(index=False))

save_table_as_png(
    grid_cv_results_table,
    "decision_tree_grid_cv_results_top20.png",
    title="Top 20 GridSearchCV Hyperparameter Combinations",
    max_rows=20
)

print("\nGridSearchCV table files saved:")
print("decision_tree_grid_cv_results.csv")
print("decision_tree_grid_cv_results.html")
print("decision_tree_grid_cv_results.txt")
print("decision_tree_grid_cv_results_top20.png")


# ===================== 【7】Use Method 1: Adopt GridSearchCV Best Model =====================
# Method 1:
# Directly use grid_cv.best_estimator_ as the final Decision Tree pipeline.
# This means all later evaluation, confusion matrix, ROC curve, tree visualisation,
# and feature importance are based on the best GridSearchCV model.
dt_pipeline = grid_cv.best_estimator_

print("\n===================== Final Model Uses GridSearchCV Best Estimator =====================")
print("Final model parameters:")
print(dt_pipeline.named_steps["clf"].get_params())

y_pred_test = dt_pipeline.predict(X_test)


# ===================== 【8】Final Test Set Results for Best GridSearchCV Model =====================
test_report = classification_report(
    y_test,
    y_pred_test,
    labels=sleep_labels,
    target_names=sleep_labels,
    output_dict=True,
    zero_division=0
)

comparison = pd.DataFrame([{
    "Model": "Decision Tree - Best GridSearchCV",
    "Train/Test Split": "80/20",
    "Best CV Accuracy": round(grid_cv.best_score_, 4),
    "Final Test Accuracy": round(accuracy_score(y_test, y_pred_test), 4),
    "Final Test Macro F1": round(test_report["macro avg"]["f1-score"], 4),
    "Final Test Precision": round(test_report["macro avg"]["precision"], 4),
    "Final Test Recall": round(test_report["macro avg"]["recall"], 4),
}])

print("\n===================== Final Test Results: Best GridSearchCV Model =====================")
print(comparison.to_string(index=False))

print("\n===================== Test Classification Report: Best GridSearchCV Model =====================")
print(classification_report(
    y_test,
    y_pred_test,
    labels=sleep_labels,
    target_names=sleep_labels,
    zero_division=0
))



# ===================== 【8.1】Training vs Testing Accuracy Bar Chart =====================
# This plot is similar to the Random Forest training vs testing accuracy chart.
# It uses the final GridSearchCV best Decision Tree model.
train_accuracy = accuracy_score(y_train, dt_pipeline.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\n===================== Training vs Testing Accuracy =====================")
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

plot_train_test_accuracy_overlap(
    train_acc=train_accuracy,
    test_acc=test_accuracy,
    model_name="Decision Tree",
    save_path="decision_tree_train_test_accuracy_overlap.png"
)

# ===================== 【9】10-fold CV for the Final Best Model =====================
print("\n===================== 10-fold CV: Best GridSearchCV Model =====================")
print(f'{"Model":<36}  {"CV Mean Acc":>12}  {"CV Std":>8}  {"Stable?"}')
print("-" * 80)

cv_pipe = dt_pipeline

cv_res = cross_validate(
    cv_pipe,
    X_train,
    y_train,
    cv=cv10,
    scoring="accuracy"
)

cv_mean = cv_res["test_score"].mean()
cv_std = cv_res["test_score"].std()
stable = "Yes" if cv_std < 0.05 else "No — high variance"

print(f'{"Decision Tree - Best GridSearchCV":<36}  {cv_mean:>12.4f}  {cv_std:>8.4f}  {stable}')


# ===================== 【9.1】Validation Predictions from 10-fold CV =====================
# Every training sample is predicted once by a model that did not train on that fold.
y_val_pred = cross_val_predict(
    cv_pipe,
    X_train,
    y_train,
    cv=cv10
)


# ===================== 【9.2】Validation Classification Report =====================
print("\n===================== Validation Classification Report =====================")
print("This report is based on cross-validation predictions on the 80% training set.\n")

print(classification_report(
    y_train,
    y_val_pred,
    labels=sleep_labels,
    target_names=sleep_labels,
    zero_division=0
))

val_report = classification_report(
    y_train,
    y_val_pred,
    labels=sleep_labels,
    target_names=sleep_labels,
    output_dict=True,
    zero_division=0
)


# ===================== 【9.3】Validation Confusion Matrix =====================
val_cm = confusion_matrix(
    y_train,
    y_val_pred,
    labels=sleep_labels
)

val_disp = ConfusionMatrixDisplay(
    confusion_matrix=val_cm,
    display_labels=sleep_labels
)

val_disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix — Best GridSearchCV Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===================== 【9.4】Summary Table =====================
dt_summary = pd.DataFrame([
    {
        "Model": "Baseline Decision Tree (depth=3)",
        "Split": "80/20",
        "CV Method": f"{cv10.get_n_splits()}-fold CV on training set only",
        "CV Mean Accuracy": round(baseline_cv_mean, 4),
        "CV Std": round(baseline_cv_std, 4),
        "Stable?": baseline_stable,
        "Validation Accuracy": "",
        "Validation Macro F1": "",
        "Validation Precision": "",
        "Validation Recall": "",
        "Final Test Accuracy": round(accuracy_score(y_test, baseline_pred_test), 4),
        "Final Test Macro F1": round(baseline_test_report["macro avg"]["f1-score"], 4),
        "Final Test Precision": round(baseline_test_report["macro avg"]["precision"], 4),
        "Final Test Recall": round(baseline_test_report["macro avg"]["recall"], 4),
    },
    {
        "Model": "Decision Tree - Best GridSearchCV",
        "Split": "80/20",
        "CV Method": f"{cv10.get_n_splits()}-fold CV on training set only",
        "CV Mean Accuracy": round(cv_mean, 4),
        "CV Std": round(cv_std, 4),
        "Stable?": stable,
        "Validation Accuracy": round(accuracy_score(y_train, y_val_pred), 4),
        "Validation Macro F1": round(val_report["macro avg"]["f1-score"], 4),
        "Validation Precision": round(val_report["macro avg"]["precision"], 4),
        "Validation Recall": round(val_report["macro avg"]["recall"], 4),
        "Final Test Accuracy": round(accuracy_score(y_test, y_pred_test), 4),
        "Final Test Macro F1": round(test_report["macro avg"]["f1-score"], 4),
        "Final Test Precision": round(test_report["macro avg"]["precision"], 4),
        "Final Test Recall": round(test_report["macro avg"]["recall"], 4),
    }
])

print("\n===================== Decision Tree Summary Table =====================")
print(dt_summary.to_string(index=False))

dt_summary.to_csv("decision_tree_summary_table.csv", index=False)
print("\nSummary table saved as: decision_tree_summary_table.csv")


# ===================== 【10】Default Unconstrained Tree Analysis =====================
# This section compares a simple default tree against the final GridSearchCV model.
dt_default_pipe = build_dt_pipeline(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    criterion="gini",
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2
)

dt_default_pipe.fit(X_train, y_train)
default_pred = dt_default_pipe.predict(X_test)
default_clf = dt_default_pipe.named_steps["clf"]

best_clf = dt_pipeline.named_steps["clf"]

print("\n===================== Default Unconstrained Tree vs Best GridSearchCV Tree =====================")
print(f"默认未限制决策树准确率:  {accuracy_score(y_test, default_pred):.4f}")
print(f"默认未限制决策树最大深度: {default_clf.get_depth()}")
print(f"默认未限制决策树叶节点数: {default_clf.get_n_leaves()}")

print(f"\nGridSearchCV 最优树最大深度: {best_clf.get_depth()}")
print(f"GridSearchCV 最优树叶节点数: {best_clf.get_n_leaves()}")


# ===================== 【11】Test Confusion Matrix =====================
test_cm = confusion_matrix(
    y_test,
    y_pred_test,
    labels=sleep_labels
)

test_disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=sleep_labels
)

test_disp.plot(cmap="Blues")
plt.title("Test Confusion Matrix — Best GridSearchCV Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===================== 【12】Get Encoded Feature Names =====================
# After one-hot encoding categorical variables, the Decision Tree may see more encoded model features.
preprocessor = dt_pipeline.named_steps["preprocess"]
encoded_feature_names = preprocessor.get_feature_names_out().tolist()

X_train_enc = pd.DataFrame(
    preprocessor.transform(X_train),
    columns=encoded_feature_names,
    index=X_train.index
)

X_test_enc = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=encoded_feature_names,
    index=X_test.index
)

print("\n===================== Encoded Model Features =====================")
print(f"Raw input variables: {len(feature_cols)}")
print(f"Encoded model features after one-hot encoding: {len(encoded_feature_names)}")
print(encoded_feature_names)


# ===================== 【13】Decision Tree Visualisation =====================
plt.figure(figsize=(28, 12))
plot_tree(
    dt_pipeline.named_steps["clf"],
    feature_names=encoded_feature_names,
    class_names=sleep_labels,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Structure — Best GridSearchCV Model")
plt.tight_layout()
plt.show()


# ===================== 【14】Decision Tree Rules =====================
tree_rules = export_text(
    dt_pipeline.named_steps["clf"],
    feature_names=encoded_feature_names
)

print("\n===================== Decision Tree Rules: Best GridSearchCV Model =====================")
print(tree_rules)


# ===================== 【15】Raw Feature Importance =====================
# Use the final GridSearchCV best model to calculate feature importance.
final_clf = dt_pipeline.named_steps["clf"]

encoded_feat_imp_df = pd.DataFrame({
    "Encoded Feature": encoded_feature_names,
    "Raw Feature": [
        encoded_to_raw_feature(name, feature_cols)
        for name in encoded_feature_names
    ],
    "Importance": final_clf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n===================== Encoded Feature Importances — Best GridSearchCV Model =====================")
print(encoded_feat_imp_df.to_string(index=False))

# Aggregate encoded importance back to the original raw variables
raw_feat_imp_df = (
    encoded_feat_imp_df
    .groupby("Raw Feature", as_index=False)["Importance"]
    .sum()
)

# Make sure all raw variables appear, even if their importance is 0
raw_feat_imp_df = (
    pd.DataFrame({"Raw Feature": feature_cols})
    .merge(raw_feat_imp_df, on="Raw Feature", how="left")
    .fillna({"Importance": 0})
    .sort_values("Importance", ascending=False)
)

print("\n===================== Raw Feature Importances =====================")
print(raw_feat_imp_df.to_string(index=False))

raw_feat_imp_df.to_csv("decision_tree_raw_feature_importance.csv", index=False)
print("\nRaw feature importance table saved as: decision_tree_raw_feature_importance.csv")

plt.figure(figsize=(10, 5))
plt.bar(raw_feat_imp_df["Raw Feature"], raw_feat_imp_df["Importance"])
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.xlabel("Raw Input Variable")
plt.ylabel("Aggregated Importance Score")
plt.title("Feature Importances — Best GridSearchCV Model")
plt.tight_layout()
plt.show()


# ===================== 【16】Per-class + Macro-average ROC Curve =====================
try:
    y_score = dt_pipeline.predict_proba(X_test)
    class_order = list(dt_pipeline.named_steps["clf"].classes_)

    y_test_bin = label_binarize(y_test, classes=class_order)
    n_classes = len(class_order)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_order[i]} AUC = {roc_auc[i]:.3f}"
        )

    plt.plot(
        all_fpr,
        mean_tpr,
        linewidth=2.5,
        label=f"Macro-average AUC = {macro_auc:.3f}"
    )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Decision Tree ROC Curves — Best GridSearchCV Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\nROC curve could not be generated:")
    print(e)


# ===================== 【17】Feature Subset Experiments — Best GridSearchCV Decision Tree =====================
# These experiments use RAW VARIABLES, not encoded feature names:
# 1. All raw variables
# 2. Top variables selected from the feature importance ranking
# 3. Corr-reduced raw variables
print("\n===================== Feature Subset Experiments — Best GridSearchCV Decision Tree =====================")

all_features = feature_cols

# Use top variables ranked by aggregated raw importance.
# If there are at least 5 features, use top 5. Otherwise use all available features.
top_n = min(5, len(all_features))
top_features = raw_feat_imp_df["Raw Feature"].head(top_n).tolist()

print(f"Top {top_n} raw variables used:")
print(top_features)
print(f"Number of raw variables used: {len(top_features)}")

# Corr-reduced raw variables
# For correlation, convert categorical raw variables to one-hot internally,
# then map correlated encoded columns back to raw variables.
preprocessor_for_corr = build_preprocessor(categorical_cols, numerical_cols)
X_train_corr_enc = pd.DataFrame(
    preprocessor_for_corr.fit_transform(X_train),
    columns=preprocessor_for_corr.get_feature_names_out(),
    index=X_train.index
)

corr_matrix = X_train_corr_enc.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

encoded_corr_to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.6)
]

raw_corr_to_drop = sorted(set(
    encoded_to_raw_feature(col, feature_cols)
    for col in encoded_corr_to_drop
))

corr_reduced_features = [
    f for f in all_features
    if f not in raw_corr_to_drop
]

# Safety fallback: do not allow an empty feature set
if len(corr_reduced_features) == 0:
    corr_reduced_features = all_features.copy()

print("Highly correlated encoded features dropped:")
print(encoded_corr_to_drop)

print("Mapped raw variables dropped:")
print(raw_corr_to_drop)

print(f"Corr-reduced raw feature count: {len(corr_reduced_features)}")
print(corr_reduced_features)

feature_sets = {
    f"All {len(all_features)} raw variables": all_features,
    f"Top {top_n} raw variables": top_features,
    "Corr-reduced raw variables": corr_reduced_features
}

# Extract best hyperparameters for feature subset experiments
best_params = grid_cv.best_params_
best_criterion = best_params["clf__criterion"]
best_max_depth = best_params["clf__max_depth"]
best_min_samples_split = best_params["clf__min_samples_split"]
best_min_samples_leaf = best_params["clf__min_samples_leaf"]

dt_feature_results = []

for feat_label, feat_cols in feature_sets.items():
    print(f"\nRunning Decision Tree with feature set: {feat_label}")
    print(f"Number of raw variables used: {len(feat_cols)}")
    print("Raw variables used:")
    print(feat_cols)

    X_train_sub = X_train[feat_cols]
    X_test_sub = X_test[feat_cols]

    sub_numerical_cols = X_train_sub.select_dtypes(include=[np.number]).columns.tolist()
    sub_categorical_cols = [col for col in X_train_sub.columns if col not in sub_numerical_cols]

    # Use the same best hyperparameters selected by GridSearchCV
    pipe_sub = build_dt_pipeline(
        categorical_cols=sub_categorical_cols,
        numerical_cols=sub_numerical_cols,
        criterion=best_criterion,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf
    )

    cv_sub = cross_validate(
        pipe_sub,
        X_train_sub,
        y_train,
        cv=cv10,
        scoring="accuracy"
    )

    cv_sub_mean = cv_sub["test_score"].mean()
    cv_sub_std = cv_sub["test_score"].std()

    y_val_pred_sub = cross_val_predict(
        pipe_sub,
        X_train_sub,
        y_train,
        cv=cv10
    )

    val_report_sub = classification_report(
        y_train,
        y_val_pred_sub,
        labels=sleep_labels,
        target_names=sleep_labels,
        output_dict=True,
        zero_division=0
    )

    pipe_sub.fit(X_train_sub, y_train)
    y_pred_sub = pipe_sub.predict(X_test_sub)

    acc = accuracy_score(y_test, y_pred_sub)

    sub_report = classification_report(
        y_test,
        y_pred_sub,
        labels=sleep_labels,
        target_names=sleep_labels,
        output_dict=True,
        zero_division=0
    )

    dt_feature_results.append({
        "Model": "Decision Tree - Best GridSearchCV Params",
        "Feature set": feat_label,
        "Number of raw variables": len(feat_cols),

        "CV Mean Accuracy": round(cv_sub_mean, 4),
        "CV Std": round(cv_sub_std, 4),

        "Validation Accuracy": round(accuracy_score(y_train, y_val_pred_sub), 4),
        "Validation Macro F1": round(val_report_sub["macro avg"]["f1-score"], 4),

        "Final Test Accuracy": round(acc, 4),
        "Final Test Macro F1": round(sub_report["macro avg"]["f1-score"], 4),
        "Final Test Precision": round(sub_report["macro avg"]["precision"], 4),
        "Final Test Recall": round(sub_report["macro avg"]["recall"], 4),
    })

dt_feature_df = pd.DataFrame(dt_feature_results)

print("\nDecision Tree Feature Subset Results:")
print(dt_feature_df.to_string(index=False))

dt_feature_df.to_csv("decision_tree_feature_subset_results.csv", index=False)
print("\nFeature subset results saved as: decision_tree_feature_subset_results.csv")


# ===================== 【18】Pivot Tables for Report =====================
dt_pivot_test = dt_feature_df.pivot(
    index="Model",
    columns="Feature set",
    values="Final Test Accuracy"
)

print("\nFeature Subset Table Format — Final Test Accuracy:")
print(dt_pivot_test.to_string())

dt_pivot_cv = dt_feature_df.pivot(
    index="Model",
    columns="Feature set",
    values="CV Mean Accuracy"
)

print("\nFeature Subset Table Format — CV Mean Accuracy:")
print(dt_pivot_cv.to_string())

dt_pivot_val = dt_feature_df.pivot(
    index="Model",
    columns="Feature set",
    values="Validation Macro F1"
)

print("\nFeature Subset Table Format — Validation Macro F1:")
print(dt_pivot_val.to_string())


# ===================== 【19】Plot Feature Subset Comparison =====================
plt.figure(figsize=(9, 5))
plt.bar(dt_feature_df["Feature set"], dt_feature_df["Final Test Accuracy"])
plt.xlabel("Raw Variable Feature Set")
plt.ylabel("Final Test Accuracy")
plt.title("Decision Tree Accuracy Across Feature Sets — Best GridSearchCV Params")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(dt_feature_df["Feature set"], dt_feature_df["CV Mean Accuracy"])
plt.xlabel("Raw Variable Feature Set")
plt.ylabel("CV Mean Accuracy")
plt.title("Decision Tree Cross-validation Accuracy Across Feature Sets — Best GridSearchCV Params")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(dt_feature_df["Feature set"], dt_feature_df["Validation Macro F1"])
plt.xlabel("Raw Variable Feature Set")
plt.ylabel("Validation Macro F1")
plt.title("Decision Tree Validation Macro F1 Across Feature Sets — Best GridSearchCV Params")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


print("\n===================== Finished =====================")

# ===================== Notebook Runtime Timer =====================
end_time = time.perf_counter()
print(f"Total notebook runtime: {end_time - start_time:.2f} seconds")
