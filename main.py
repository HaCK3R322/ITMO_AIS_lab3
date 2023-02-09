import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt


def info(attributes, target):
    h = 0
    for k in target.unique():
        p = len(attributes.loc[attributes.index.intersection(target.index[target == k])]) / len(attributes)
        h -= p * np.log2(p)

    return h


def info_x(attributes, target, x):
    unique_values_of_x = attributes[x].unique()

    h = 0
    for value in unique_values_of_x:
        h += len(attributes[attributes[x] == value]) / len(attributes) \
             * info(attributes[attributes[x] == value], target.loc[attributes.index[attributes[x] == value]])

    return h


def split_info_x(attributes, target, x):
    unique_values_of_x = attributes[x].unique()

    h = 0
    for value in unique_values_of_x:
        p = len(attributes[attributes[x] == value]) / len(attributes)
        h -= p * np.log2(p)

    return h


def gain_ratio(attributes, target, x):
    if split_info_x(attributes, target, x) == 0:
        return 0
    return (info(attributes, target) - info_x(attributes, target, x)) / split_info_x(attributes, target, x)


def check_attributes_correctness(attributes, target):
    connected = attributes.join(target)

    if len(attributes.drop_duplicates()) != len(connected.drop_duplicates()):
        raise ValueError(f'Attributes are not correct: some unique rows contains different target values\n'
                         f'Number of unique rows: {len(attributes.drop_duplicates())}\n'
                         f'Number of unique rows with target: {len(connected.drop_duplicates())}')


def build_tree(attributes, target):
    classes = set(target)
    if len(classes) == 1:
        return classes.pop()

    gain_ratios = [gain_ratio(attributes, target, attr) for attr in attributes]
    best_gain_ratio = max(gain_ratios)
    best_attribute = attributes.columns[np.argmax(gain_ratios)]

    tree = {best_attribute: {}}
    for value in set(attributes[best_attribute]):
        examples_indexes = attributes.index[attributes[best_attribute] == value]

        new_attributes = attributes.loc[examples_indexes]
        new_target = target.loc[examples_indexes]

        subtree = build_tree(new_attributes, new_target)
        tree[best_attribute][value] = subtree

    return tree


def predict(tree, row, default_prediction=None):
    if type(tree) == int:
        return tree

    for key in tree:
        try:
            value = row[key]
        except ValueError:
            return None
        try:
            t = tree[key][value]
        except KeyError:
            print(f'Prediction for row {row.values} is not found so default prediction ({default_prediction}) is used')
            return default_prediction

        result = predict(t, row, default_prediction)
        if result is not None:
            return result
    return None


def main():
    df = pd.read_csv('DATA.csv', sep=';')

    # Напишу на русском так быстрее: по заданию просили взять только корень_кв(колво атрибутов) случайных аттрибутов
    # Однако, в таком случае дерево будет строиться не совсем корректно, так как в некоторых случаях
    # мы будем иметь одинаковые наборы предикторов с разными целевыми переменными (т.е. на целевую переменную
    # влияют предикотры, которые мы не учитываем)
    # В таком случае будет выбор: либо присваивать класс который встречается чаще при одинаковых предикторах,
    # либо выбирать убирать из выборки такие строки.
    # Так как оба этих варианта не очень хороши, я решил отказаться от ограничения по предикторам и выбрал минимально
    # необходимое количество предикторов (просто увеличивал число лучших по приросту информации предикторов
    # до тех пор, пока данные не станут корректными)

    attributes_to_choose = ['26', '7', '3', '17', '25', '13', '11', '28', '18', '9', '4', '8', '24', '27', '22', '20',
                            '21', '30', '2', '29']

    attributes = df[df.columns[df.columns.isin(attributes_to_choose)]]
    # normilize grades with rule: 1,2 - 0 (bad) and 3,4,5 - 1 (good)
    target = df['GRADE'].map(lambda x: 0 if x <= 2 else 1)
    check_attributes_correctness(attributes, target)

    # Split data into 80% train and 20% test
    msk = np.random.default_rng(228).random(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    # Split attributes and target for training and testing data
    train_attributes = train[train.columns[train.columns.isin(attributes_to_choose)]]
    train_target = train['GRADE'].map(lambda x: 0 if x <= 2 else 1)

    test_attributes = test[test.columns[test.columns.isin(attributes_to_choose)]]
    test_target = test['GRADE'].map(lambda x: 0 if x <= 2 else 1)

    # Build the decision tree using the training data
    tree = build_tree(train_attributes, train_target)

    # Use the decision tree to make predictions on the test data
    most_common_class = train_target.value_counts().index[0]
    print(f'Most common class in train data is {most_common_class}')
    predictions = []
    for i in range(len(test)):
        prediction = predict(tree, test_attributes.iloc[i, :], default_prediction=most_common_class)
        predictions.append(prediction)

    # Convert predictions to a numpy array
    predictions = np.array(predictions)

    # Convert test target to a numpy array
    test_target = np.array(test_target)

    # Calculate accuracy
    accuracy = np.sum(predictions == test_target) / len(test_target)

    # Calculate precision
    precision = np.sum((predictions == 1) & (test_target == 1)) / np.sum(predictions == 1)

    # Calculate recall
    recall = np.sum((predictions == 1) & (test_target == 1)) / np.sum(test_target == 1)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)

    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

    fpr, tpr, thresholds = roc_curve(test_target, predictions)

    # Calculate the AUC-ROC
    auc_roc = roc_auc_score(test_target, predictions)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Calculate the AUC-PR
    auc_pr = average_precision_score(test_target, predictions)
    precision, recall, thresholds_pr = precision_recall_curve(test_target, predictions)

    # plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % auc_pr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()


def test_function():
    # Маленькая тестовая функция, на которой можно наглядно увидеть работоспособность алооритма

    attributes = pd.DataFrame({
        'A': [1, 1, 2, 2, 2, 3, 3, 3, 3],
        'B': [2, 2, 3, 3, 3, 1, 1, 2, 2],
        'C': [1, 2, 1, 2, 1, 2, 1, 2, 1]
    })

    target = pd.DataFrame({
        'target': [0, 0, 1, 0, 1, 1, 0, 1, 0]
    })

    tree = build_tree(attributes, target['target'])
    print(tree)

    print('Predicted target:', predict(tree, attributes.iloc[2]))


if __name__ == '__main__':
    main()
    # test_function()
