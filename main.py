import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# R1: Bejaia Region Dataset
# R2: Sidi-Bel Abbes Region Dataset
# R3: Portuguese Forest Fire Dataset

def read(file):  # returns array of arrays, each inner array is a row of data
    data = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header row
        for row in reader:
            line = []
            for x in row[3:13]:  # data points are floats
                line.append(float(x))
            if type(row[13]) == str and row[13].strip() == "fire":
                line.append(1)
            if type(row[13]) == str and row[13].strip() == "not fire":
                line.append(0)
            data.append(line)
        return np.array(data, dtype='object')


def adapt_alg(file):  # adapt algerian dataset to portuguese dataset format for testing (no 11: BUI and 12: FWI entries)
    data = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header row
        for row in reader:
            line = []
            for x in row[3:11]:  # data points are floats
                line.append(float(x))
            if type(row[13]) == str and row[13].strip() == "fire":
                line.append(1)
            if type(row[13]) == str and row[13].strip() == "not fire":
                line.append(0)
            data.append(line)
        return np.array(data, dtype='object')


def adapt_port(file):  # adapt portuguese dataset to algerian dataset format for testing
    data = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header row
        for row in reader:
            line = [row[8], row[9], row[10], row[11], row[4], row[5], row[6], row[7]]
            line = [float(x) for x in line]
            if float(row[12]) != 0:
                line.append(1)
            else:
                line.append(0)
            data.append(line)
        return np.array(data, dtype='object')


def get_ranges(data):
    range_mins, range_maxes = [], []
    for i in range(len(data[0]) - 1):
        range_mins.append(min([x[i] for x in data]))
        range_maxes.append(max([x[i] for x in data]))
    return range_mins, range_maxes


def get_spread(data):
    means, stds = [], []
    for i in range(len(data[0]) - 1):
        means.append(np.mean([x[i] for x in data]))
        stds.append(np.std([x[i] for x in data]))
    return means, stds


def normalize(training_data, data):  # min max normalization
    range_mins, range_maxes = get_ranges(training_data)
    shape = np.shape(data)
    normalized = np.zeros(shape=shape)
    for i in range(len(data)):  # for row in data
        for j in range(len(data[0]) - 1):  # for feature in row, but not the class
            normalized[i][j] = (data[i][j] - range_mins[j]) / (range_maxes[j] - range_mins[j])
        np.append(normalized, data[i][-1])
    return normalized


# data can be either training or testing data; ensures testing data uses training data mean and std for standardization
def z_score(training_data, data):
    shape = np.shape(data)
    z_scores = np.zeros(shape=shape)
    means, stds = get_spread(training_data)
    for i in range(len(data)):  # for row in data
        for j in range(len(data[0]) - 1):  # for feature in row, but not the class
            z_scores[i][j] = (data[i][j] - means[j]) / stds[j]
        np.append(z_scores, data[i][-1])
    return z_scores


def standardize(x_train, x_test):  # same as z-score but with skitlearn
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def dp(v1, v2):  # finds dot product of two vectors
    dot_product = 0
    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]
    return dot_product


def sigmoid(z):  # returns sigmoid of a number
    return 1 / (1 + np.exp(-z))


def get_y(data):  # extract class from data
    shape = len(data)
    y = np.empty(shape=shape)
    for i, row in enumerate(data):
        y[i] = row[-1]
    return y


def get_x(data):  # extract features from data
    return np.copy(data[:, :-1])


# data analysis
def hist(data, range_min, range_max, variable, title, y_lim):
    bins = np.linspace(np.floor(range_min), np.ceil(range_max), 10)
    plt.hist(data, bins=bins, align='left', alpha=0.7, rwidth=0.8)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.ylim(0, y_lim)
    plt.title(title)
    plt.savefig(variable + title + ".png")
    plt.clf()
    # plt.show()


def compile_hist(data, labels, y_lim):  # compile histogram for regions 1 and 2
    no_fire, fire = [], []
    range_mins, range_maxes = get_ranges(data)
    for i in range(len(data[0]) - 1):
        no_fire.append([row[i] for row in data if row[-1] == 0])
        fire.append([row[i] for row in data if row[-1] == 1])
    for i in range(len(fire)):
        hist(fire[i], range_mins[i], range_maxes[i], labels[i], "Given Fire", y_lim)
        hist(no_fire[i], range_mins[i], range_maxes[i], labels[i], "Given No Fire", y_lim)


def prob_fire(y_data):
    fires = 0
    for y in y_data:
        if y != 0:
            fires += 1
    return fires / len(y_data)


def confusion_matrix(truth, pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(truth)):
        if truth[i] == 1 and pred[i] == 1:
            tp += 1
        if truth[i] == 0 and pred[i] == 0:
            tn += 1
        if truth[i] == 0 and pred[i] == 1:
            fp += 1
        if truth[i] == 1 and pred[i] == 0:
            fn += 1
    return tp, fp, fn, tn


# logistic regression
def train(training_data, num_steps, step_size):
    num_features = len(training_data[0]) - 1  # number of entries per row in data, minus the class
    num_data = len(training_data)  # number of total rows in data
    params = [0] * num_features  # initialize thetas as all zeros
    for s in range(num_steps):
        gradient = [0] * num_features
        for i in range(num_data):
            for j in range(num_features):  # for each entry per row (index i, j in data matrix)
                gradient[j] += training_data[i][j] * (training_data[i][-1] - sigmoid(dp(params, training_data[i][:-1])))
        for j in range(len(params)):
            params[j] += step_size * gradient[j]
    return params


def logistic_regression(testing_data, params):
    probs = []
    for i in range(len(testing_data)):  # exclude y
        probs.append(sigmoid(dp(params, testing_data[i][:-1])))
    return probs


def predict(probs):
    predictions = []
    for i in range(len(probs)):
        if probs[i] > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def test(testing_data, predictions):
    correct = 0
    for i in range(len(testing_data)):
        if testing_data[i][-1] == predictions[i]:
            correct += 1
    return correct / len(testing_data)


def combine_matrices(matrix1, matrix2):
    return np.concatenate((matrix1, matrix2), axis=0)


def run_lr_analysis(x_train, y_train, x_test, y_test):
    x_train, x_test = standardize(x_train, x_test)

    # my logistic regression
    training = np.c_[x_train, y_train]
    testing = np.c_[x_test, y_test]
    params = train(training, 500, 0.00625)
    probs = logistic_regression(testing, params)
    pred = predict(probs)
    print("Homemade LR:", test(testing, pred))

    # skitlearn logistic regression
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred_lr = logreg.predict(x_test)
    logistic_accuracy = accuracy_score(y_test, y_pred_lr)
    print("Skitlearn LR:", logistic_accuracy)

    # lasso
    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    lasso.fit(x_train, y_train)
    y_pred_lasso = lasso.predict(x_test)
    lasso_accuracy = accuracy_score(y_test, y_pred_lasso)
    print("Lasso:", lasso_accuracy)

    print("my lr", confusion_matrix(y_test, pred))
    print("sklearn lr", confusion_matrix(y_test, y_pred_lr))
    print("lasso", confusion_matrix(y_test, y_pred_lasso))


def run_rf_analysis(x_train, y_train, x_test, y_test):
    # random forests
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("Random Forests:", rf_accuracy)

    print("rf", confusion_matrix(y_test, y_pred_rf))

    # feature importances
    feature_importances = rf.feature_importances_
    threshold_value = 0.1
    sfm = SelectFromModel(rf, threshold=threshold_value)
    sfm.fit(x_train, y_train)
    print("feature importances: ", feature_importances)


def analyze(x_train, y_train, x_test, y_test):
    run_lr_analysis(x_train, y_train, x_test, y_test)
    run_rf_analysis(x_train, y_train, x_test, y_test)


def del_unimportant_features(x_data1, x_data2, indices_to_del):
    return np.delete(x_data1, indices_to_del, axis=1), np.delete(x_data2, indices_to_del, axis=1)


def main():
    r1_and_r2_labels = ["Temperature", "Relative Humidity", "Wind Speed", "Rain", "Fine Fuel Moisture Code (FFMC)",
                        "Duff Moisture Code (DMC)", "Drought Code (DC)", "Initial Spread Index (ISI)",
                        "Buildup Index (BUI)",
                        "Fire Weather Index (FWI)"]
    r3_labels = ["Temperature", "Relative Humidity", "Wind Speed", "Rain", "Fine Fuel Moisture Code (FFMC)",
                 "Duff Moisture Code (DMC)", "Drought Code (DC)", "Initial Spread Index (ISI)"]
    # data
    r1_data, r2_data = read("Algerian_forest_r1.csv"), read("Algerian_forest_r2.csv")
    y_r1, y_r2 = get_y(r1_data), get_y(r2_data)
    x_r1, x_r2 = get_x(r1_data), get_x(r2_data)

    # data without BUI and FWI features (for training region 3)
    new_r1_data, new_r2_data, new_r3_data = adapt_alg("Algerian_forest_r1.csv"), adapt_alg(
        "Algerian_forest_r2.csv"), adapt_port("Portuguese_forest_r3.csv")
    new_y_r1, new_y_r2, new_y_r3 = get_y(new_r1_data), get_y(new_r2_data), get_y(new_r3_data)
    new_x_r1, new_x_r2, new_x_r3 = get_x(new_r1_data), get_x(new_r2_data), get_x(new_r3_data)

    # compile_hist(new_r3_data, r3_labels, 100)

    print("Train: Algerian Region 1, Test: Algerian Region 2")
    analyze(x_r1, y_r1, x_r2, y_r2)

    print("Train: Algerian Region 1, Test Algerian Region 2 (without unimportant features)")
    x1, x2 = del_unimportant_features(x_r1, x_r2, [1, 2, 0])
    analyze(x1, y_r1, x2, y_r2)

    print("Train: Algerian Region 1, Test: Portuguese Region")
    analyze(new_x_r1, new_y_r1, new_x_r3, new_y_r3)

    print("Train: Algerian Region 1, Test Portuguese Region 3 (without unimportant features)")
    new_x1, new_x3 = del_unimportant_features(new_x_r1, new_x_r3, [2])
    analyze(new_x1, new_y_r1, new_x3, new_y_r3)

    print("r1 probability of fire: ", prob_fire(new_y_r1))
    print("r2 probability of fire: ", prob_fire(new_y_r2))
    print("r3 probability of fire: ", prob_fire(new_y_r3))


if __name__ == "__main__":
    main()
