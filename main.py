import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.logistic_regression import train_logistic_regression, predict_logistic_regression
from models.kneighbors import train_kneighbors, predict_kneighbors
from models.random_forest import train_random_forest, predict_random_forest
from models.reports import evaluate_classification_report, plot_confusion_matrix
from models.xgboost import train_xgboost, predict_xgboost

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

# valorile 0 nu sunt valori valide pentru aceste caracteristici si le tratam ca date lipsa
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def ensure_outputs_dir():
    os.makedirs("outputs", exist_ok=True)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Lipsesc coloanele: {missing_cols}. Coloane gasite: {list(df.columns)}"
        )

    df = df.copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def fit_preprocess(X_train: pd.DataFrame):

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    return imputer, scaler, X_train_scaled


def transform_preprocess(imputer, scaler, X: pd.DataFrame):

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled


def get_model(model_name: str):
    model_name = model_name.strip().lower()

    if model_name == "logreg":
        return "Logistic Regression", train_logistic_regression, predict_logistic_regression
    if model_name == "knn":
        return "KNN", train_kneighbors, predict_kneighbors
    if model_name == "rf":
        return "Random Forest", train_random_forest, predict_random_forest
    if model_name == "xgb":
        return "XGBoost", train_xgboost, predict_xgboost

    raise ValueError("Model invalid. Alege din: logreg | knn | rf | xgb")


def available_models():
    models = ["logreg", "knn", "rf", "xgb"]
    return models


def compute_metrics(y_true, y_pred):
    # Pentru clasa pozitiva (1 = diabet)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }


def evaluate_one_model(csv_path: str, model_name: str, test_size: float = 0.2, seed: int = 42, save_cm: bool = True):
    df = load_and_clean(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    imputer, scaler, X_train_scaled = fit_preprocess(X_train)
    X_test_scaled = transform_preprocess(imputer, scaler, X_test)

    pretty_name, train_fn, pred_fn = get_model(model_name)
    model = train_fn(X_train_scaled, y_train)
    y_pred = pred_fn(model, X_test_scaled)

    acc, report, cm = evaluate_classification_report(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n=== Evaluare: {pretty_name} ===")
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

    if save_cm:
        ensure_outputs_dir()
        safe_name = pretty_name.lower().replace(" ", "_")
        cm_path = os.path.join("outputs", f"confusion_matrix_{safe_name}.png")
        try:
            plot_confusion_matrix(cm, pretty_name, save_path=cm_path)
        except TypeError:

            plot_confusion_matrix(cm, pretty_name)

    return pretty_name, metrics, report


def save_metrics_row(metrics_csv_path: str, row: dict):
    ensure_outputs_dir()
    file_exists = os.path.exists(metrics_csv_path)
    df_row = pd.DataFrame([row])
    df_row.to_csv(metrics_csv_path, mode="a", header=not file_exists, index=False)


def save_report_text(report_txt_path: str, title: str, report: str, metrics: dict):
    ensure_outputs_dir()
    with open(report_txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {title} =====\n")
        f.write("Metrics:\n")
        f.write(f"  Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall   : {metrics['recall']:.4f}\n")
        f.write(f"  F1-score : {metrics['f1']:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")


def read_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("Te rog introdu o valoare numerica valida.")


def read_person_from_keyboard() -> dict:
    print("\nIntrodu datele pacientului:\n")

    pregnancies = read_float("Pregnancies: ")
    glucose = read_float("Glucose: ")
    bloodpressure = read_float("Blood Pressure: ")
    skinthickness = read_float("Skin Thickness: ")
    insulin = read_float("Insulin: ")
    bmi = read_float("BMI: ")
    dpf = read_float("Diabetes Pedigree Function: ")
    age = read_float("Age: ")

    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }


def predict_person_interactive(csv_path: str, model_name: str):
    """
    Pentru demo: antrenam modelul pe tot dataset-ul,
    apoi cerem input de la tastatura si prezicem.
    """
    df = load_and_clean(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    imputer, scaler, X_scaled = fit_preprocess(X)

    pretty_name, train_fn, pred_fn = get_model(model_name)
    model = train_fn(X_scaled, y)

    person_dict = read_person_from_keyboard()
    person_df = pd.DataFrame([person_dict], columns=FEATURES)

    # aplicam aceeasi regula 0 -> NaN pe coloanele specifice
    person_df[ZERO_AS_MISSING] = person_df[ZERO_AS_MISSING].replace(0, np.nan)

    person_scaled = transform_preprocess(imputer, scaler, person_df)
    prediction = pred_fn(model, person_scaled)[0]

    print("\nREZULTAT PREDICȚIE")
    print("----------------------------")
    for k, v in person_dict.items():
        print(f"{k}: {v}")

    if int(prediction) == 1:
        print("\n PACIENTUL ARE DIABET (Outcome=1)")
    else:
        print("\n PACIENTUL NU ARE DIABET (Outcome=0)")

    print(f"\n(Model utilizat: {pretty_name})")


def choose_model_interactive() -> str:
    models = available_models()
    print("\nModele disponibile:", " | ".join(models))
    while True:
        m = input("Alege modelul: ").strip().lower()
        if m in models:
            return m
        print("Model invalid. Incearca din nou.")


def main():
    print("=== Diabetes Prediction App ===\n")
    csv_path = "data/diabetes.csv"

    ensure_outputs_dir()
    metrics_csv_path = os.path.join("outputs", "metrics_results.csv")
    report_txt_path = os.path.join("outputs", "metrics_reports.txt")

    while True:
        print("\n==============================")
        print("Meniu:")
        print("1 - Evaluare model (alegi un model)")
        print("2 - Evaluare TOATE modelele (salveaza rezultate)")
        print("3 - Predicție pacient (input de la tastatură)")
        print("0 - Exit")
        choice = input("Alege opțiunea: ").strip()

        if choice == "0":
            print("Ieșire din program")
            break

        elif choice == "1":
            model_name = choose_model_interactive()
            pretty_name, metrics, report = evaluate_one_model(
                csv_path=csv_path, model_name=model_name, test_size=0.2, seed=42, save_cm=True
            )

            # salvare rezultate
            row = {"model": pretty_name, **metrics}
            save_metrics_row(metrics_csv_path, row)
            save_report_text(report_txt_path, pretty_name, report, metrics)

            print(f"\nMetrici salvate în: {metrics_csv_path}")
            print(f"Raport text salvat în: {report_txt_path}")

        elif choice == "2":
            print("\nRulez evaluarea pentru toate modelele...")
            for m in available_models():
                pretty_name, metrics, report = evaluate_one_model(
                    csv_path=csv_path, model_name=m, test_size=0.2, seed=42, save_cm=True
                )
                row = {"model": pretty_name, **metrics}
                save_metrics_row(metrics_csv_path, row)
                save_report_text(report_txt_path, pretty_name, report, metrics)

            print(f"\nToate rezultatele au fost salvate în: {metrics_csv_path}")
            print(f"Rapoartele complete au fost salvate în: {report_txt_path}")

        elif choice == "3":
            model_name = choose_model_interactive()
            predict_person_interactive(csv_path=csv_path, model_name=model_name)

        else:
            print("Opțiune invalidă. Alege 0, 1, 2 sau 3.")


if __name__ == "__main__":
    main()
