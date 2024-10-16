import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
from cache.random_forest import get_path

def random_forest():
    # Carrega arquivo CSV
    file_path = (os.getenv('CSV_PATH'))
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    # Remove linhas com atributo alvo faltante
    data = data.dropna(subset=['Label'])

    # Faz o encode de valores categóricos no atributo alvo
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # Faz o encode de valores categoricos nas features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'Label':
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])

    # Tratamento de atributos vazios, infinities e NaNs nos atributos
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    data = data.astype(np.float64)

    # Separação entre atributos preditores e atributo alvo
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Criação dos folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []
    classification_reports = []
    #Divide os dados em folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2)
        rf_classifier.fit(X_train, y_train)
        # Save the model to a file
        model_path = os.path.join(get_path(), f'rf_classifier_fold_{fold}.joblib')
        joblib.dump(rf_classifier, model_path)

        train_predictions = rf_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_accuracies.append(train_accuracy)

        test_predictions = rf_classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_accuracies.append(test_accuracy)

         # Relatório de classificação
        report = classification_report(y_test, test_predictions, output_dict=True)
        classification_reports.append(report)

        print(f"Fold {len(train_accuracies)} - Acurácia de treinamento: {train_accuracy:.4f}, Acurácia de teste: {test_accuracy:.4f}")

    # Acurácia média de treinamento e de teste
    print(f"Acurácia média de treinamento: {np.mean(train_accuracies):.4f}")
    print(f"Acurácia média de teste: {np.mean(test_accuracies):.4f}")

    # Demais dados
    overall_report = {
        'precision': np.mean([report['weighted avg']['precision'] for report in classification_reports]),
        'recall': np.mean([report['weighted avg']['recall'] for report in classification_reports]),
        'f1-score': np.mean([report['weighted avg']['f1-score'] for report in classification_reports])
    }

    print("Relatório geral:")
    print(f"Precisão: {overall_report['precision']:.4f}")
    print(f"Revocação: {overall_report['recall']:.4f}")
    print(f"f1-score: {overall_report['f1-score']:.4f}")