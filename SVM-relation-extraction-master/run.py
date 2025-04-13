import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")


# Data loading and cleaning
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data, labels = [], []
    for i in range(0, len(lines), 4):
        if i + 1 >= len(lines):
            continue
        text, label = lines[i].strip(), lines[i + 1].strip()
        text = clean_text(text)
        data.append(text)
        labels.append(label)

    return data, labels


def clean_text(text):
    # Remove the marker
    text = re.sub(r"</?e[12]>", "", text)  # Remove <e1>、</e1>、<e2>、</e2>
    text = text.lower()  # Lowercase
    words = text.split()

    # Remove stop words and reduce word forms
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)


def train_svm(X_train, y_train):

    # GridSearchCV
    param_grid = {
        "svm__C": [0.1, 1, 10],  # Regularization intensity
        "svm__class_weight": [None, "balanced"],  # Deal with category imbalances
    }

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("svm", SVC(kernel="linear", random_state=42))
    ])

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)

    print(f"Optimal parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def main(train_file, test_file):
    print("Load Data...")
    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    print("Divide the data set...")
    all_data = train_data + test_data
    all_labels = train_labels + test_labels
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print("Train the SVM...")
    model = train_svm(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    result_file = "result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"The results have been saved to {result_file}")


if __name__ == "__main__":
    data_path = "data/SemEval2010_task8_all_data"
    train_file = os.path.join(data_path, "SemEval2010_task8_training/TRAIN_FILE.TXT")
    test_file = os.path.join(data_path, "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT")
    main(train_file, test_file)
