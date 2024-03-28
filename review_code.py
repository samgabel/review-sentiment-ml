import requests
import json
import textblob
import math
import time
import warnings
import sklearn.tree
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.metrics
import nltk

# punkt donwload and path set
nltk.download("punkt", download_dir="./venv/")
nltk.data.path.append("./venv/")
warnings.filterwarnings("ignore")


def load_data(url):
    response = requests.get(url)
    if response.ok:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to load data. Status code: {response.status_code}")


def preprocess_review(review):
    """Preprocess a review by converting it to lowercase."""
    return review.lower()


def identify_unique_words(review_string):
    """Identify unique words in a given string."""
    review_lower_blob = textblob.TextBlob(review_string)
    return set(review_lower_blob.words) # pyright: ignore[reportArgumentType] -> this works properly and has been tested

def generate_relevance_scores(data, unique_words):
    """Generate relevance scores based on unique words."""
    relevance_scores = []
    for word in unique_words:
        A = B = C = D = 0
        for line in data:
            review_lower = preprocess_review(line["Review"].lower())
            safety = line["Safety hazard"]

            if word in review_lower and safety == 1:
                A += 1
            elif word in review_lower and safety == 0:
                B += 1
            elif word not in review_lower and safety == 1:
                C += 1
            elif word not in review_lower and safety == 0:
                D += 1

        try:
            score = ((math.sqrt(A + B + C + D)) * (A * D - C * B)) / (
                (math.sqrt((A + B) * (C + D)))
            )
        except ZeroDivisionError:
            score = 0

        relevance_scores.append((word, score))

    return relevance_scores


def format_2d_list(data, relevant_words):
    """Format 2D list for machine learning."""
    x = []
    y = []

    for line in data:
        review_lower = preprocess_review(line["Review"].lower())
        safety = line["Safety hazard"]

        inner_list = [1 if word in review_lower else 0 for word in relevant_words]

        x.append(inner_list)
        y.append(safety)

    return x, y


def train_machine_learning_models(x, y):
    """Train machine learning models and return accuracies."""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.2
    )

    # Decision Tree
    dt_clf = sklearn.tree.DecisionTreeClassifier()
    dt_clf.fit(x_train, y_train)
    dt_accuracy = dt_clf.score(x_test, y_test)

    # K-Nearest Neighbors
    kn_clf = sklearn.neighbors.KNeighborsClassifier(2)
    kn_clf.fit(x_train, y_train)
    kn_accuracy = kn_clf.score(x_test, y_test)

    # Neural Network
    nn_clf = sklearn.neural_network.MLPClassifier()
    nn_clf.fit(x_train, y_train)
    nn_accuracy = nn_clf.score(x_test, y_test)

    return dt_accuracy, kn_accuracy, nn_accuracy


def determine_best_model(dt_accuracy, kn_accuracy, nn_accuracy):
    """Determine the best model and return its name."""
    if dt_accuracy > kn_accuracy and dt_accuracy > nn_accuracy:
        return "Decision Tree"
    elif kn_accuracy > dt_accuracy and kn_accuracy > nn_accuracy:
        return "K-Nearest Neighbor"
    else:
        return "Neural Network"


if __name__ == "__main__":
    print("Loading data...")
    start_time = time.time()

    try:
        data = load_data("https://dgoldberg.sdsu.edu/515/appliance_reviews.json")

        review_string = " ".join(preprocess_review(line["Review"]) for line in data)
        unique_words = identify_unique_words(review_string)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Completed in", time_elapsed, "seconds.", "\n")

        print("Generating relevance scores...")
        start_time = time.time()

        relevance_scores = generate_relevance_scores(data, unique_words)
        relevant_words = [word for word, score in relevance_scores if score >= 3800]

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Completed in", time_elapsed, "seconds.", "\n")

        print("Formatting 2D List...")
        start_time = time.time()

        x, y = format_2d_list(data, relevant_words)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Completed in", time_elapsed, "seconds.", "\n")

        print("Training machine learning models...")
        start_time = time.time()

        dt_accuracy, kn_accuracy, nn_accuracy = train_machine_learning_models(x, y)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Completed in", time_elapsed, "seconds.", "\n")

        print("Decision Tree accuracy:", dt_accuracy)
        print("K-Nearest Neighbor accuracy:", kn_accuracy)
        print("Neural Network accuracy:", nn_accuracy)

        best_model = determine_best_model(
            dt_accuracy, kn_accuracy, nn_accuracy
            )
        print(f"{best_model} model performed best.")

    except Exception as e:
        print(f"Error: {e}")
