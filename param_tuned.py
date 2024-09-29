import time
import sys
import itertools
import matplotlib.pyplot as plt
from preprocess_data import prepare_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pickle
from predict import predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, target_names, title, cmap=None, normalize=False):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def create_model(train_x, test_y):
    num_classes = test_y.shape[1]

    model = Sequential()

    model.add(Dense(15, input_dim=train_x.shape[1], activation='relu'))

    model.add(Dense(75, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['mse', 'accuracy'])
    
    return model

global num_classes
global train_x_shape
def build_model(hidden_layer_sizes=(10,), activation='relu', optimizer='adam',alpha=0.0001,**kwargs):
    global train_x_shape, num_classes
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layer_sizes[0], input_dim=train_x_shape, activation=activation))

    # Adding additional hidden layers
    for size in hidden_layer_sizes[1:]:
        model.add(Dense(size, activation=activation,))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model 


def multilayer_perceptron():
    """
    This function performs multiclass classification with multilayer_perceptron
    """
    train_x, test_x, train_y, test_y = prepare_data()

    labels = test_y

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    
    
    global train_x_shape, num_classes # Ensure num_classes is defined globally for use in create_model
    num_classes = test_y.shape[1]
    train_x_shape = train_x.shape[1]
    # Wrap the Keras model using KerasClassifier
    model = KerasClassifier(build_fn=build_model, epochs=50, batch_size=10, verbose=0, activation='relu',alpha=0.0001,early_stopping=True,hidden_layer_sizes=(10,),learning_rate='constant',learning_rate_init=0.001,max_iter=200,solver='lbfgs',validation_fraction=.1)

    start = time.time()
    model.fit(train_x, train_y, epochs=50, batch_size=10, shuffle=False)
    end = time.time()
    
    y_pred = model.predict(test_x)
    predictions = np.argmax(y_pred, axis=1)

    time_ = end - start
    accuracy = accuracy_score(test_y, y_pred) * 100

    print("### MLP ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))
    
    # Define the parameter grid for tuning MLP
    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],  # Number of neurons in each hidden layer
        'activation': ['identity', 'logistic', 'tanh', 'relu'],  # Activation function for the hidden layer
         'alpha': [ 0.001, 0.1, 1.0],  # L2 penalty (regularization term)
        'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedule for weight updates
        'learning_rate_init': [0.001,  0.1],  # Initial learning rate
        'max_iter': [200, 30, 500],  # Maximum number of iterations
        }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)

    # Fit the model with GridSearchCV
    start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/mlp.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as mlp.pkl")

    # Predictions using the best model
    y_pred = best_clf.predict(test_x)

    # Calculate accuracy
    tuned_accuracy = 100 * accuracy_score(np.argmax(test_y, axis=1), y_pred)

    tuned_time = end - start
    print("### Param Tuned MLP ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)

    return tuned_time, tuned_accuracy


def xgboost():
    """
    This function performs classification with XGBoost
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))
    
   
    clf = XGBClassifier(n_estimators=20)
    
    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()
    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### XGB ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))
    
    # Define the parameter grid for tuning XGBoost
    param_grid = {
        'n_estimators': [50,  200, 500],  # Number of boosting rounds
        'learning_rate': [0.01,  0.1,  0.3],  # Step size shrinkage
        'max_depth': [3, 6,  9],  # Maximum depth of a tree
        'min_child_weight': [1,  3,  5],  # Minimum sum of instance weight (hessian) needed in a child
        'gamma': [0,  0.2,  0.4],  # Minimum loss reduction required to make a further partition
        'subsample': [0.6, 0.8, 1.0],  # Fraction of samples to be used for each tree
        'scale_pos_weight': [1, 3, 5],  # Controls the balance of positive and negative weights
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)

    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/xgb.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as xgb.pkl")

    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned Linear Gradient Boosting ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)

    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    return tuned_time, tuned_accuracy


def logistic_regression():
    """
    This function performs classification with logistic regression.
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))

    clf = LogisticRegression()
    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()

    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Logistic Regression ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    # Define the parameter grid for tuning
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
        'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs'],  # Optimization algorithms
        'max_iter': [100, 200, 300, 400],  # Maximum iterations
        'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria
        'class_weight': [None, 'balanced'],  # Handling class imbalance
        'fit_intercept': [True, False],  # Whether to include an intercept term
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)

    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as logistic_regression.pkl")

    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned Logistic Regression ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)

    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    return tuned_time, tuned_accuracy


def show_feature_importances(clf):
    importance = clf.feature_importances_
    
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.xlabel('Features')
    plt.ylabel('Feature importance factor')
    plt.title('Random Forest:Features importance')
    plt.show()
    
def random_forest():
    """
    This function performs classification with random forest.
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))
        
    clf = RandomForestClassifier()
    
    start = time.time()
    clf.fit(train_x, train_y)    
    end = time.time()
    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Random Forest ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))
    print("F1-score = ",f1_score(test_y, y_pred, average='macro')*100)

    # Define the parameter grid for tuning Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],  # Number of trees in the forest
        'max_features': ['sqrt', 'log2'],  # Number of features to consider at each split
        'max_depth': [ 10, 20, 30, 40, 50],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'class_weight': [None, 'balanced'],  # Weights associated with classes
        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)
    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/randomforest.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as randomforest.pkl")
    
    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned Random Forest ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)
    
    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    
    return tuned_time, tuned_accuracy

def k_neighbors():
    """
    This function performs classification with k neighbors algorithm.
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))
          
    clf = KNeighborsClassifier(n_neighbors=3)
    
    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()
    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### KNN ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    
    # Define the parameter grid for tuning Random Forest
    
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'p': [1, 2],
    }
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)
    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/knn.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as knn.pkl")
    
    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned K Nearest Neighbors ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)
    
    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    
    return tuned_time, tuned_accuracy




def gradient_boosting():
    """
    This function performs classification with Gradient Boosting.
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))
        
    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=75)

    start = time.time()
    clf.fit(train_x, train_y)    
    end = time.time()

    y_pred = clf.predict(test_x)
    
    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Gradient Boosting ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    # Define the parameter grid for tuning Gradient Boosting
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],  # Number of boosting stages to be run
        'learning_rate': [ 0.05, 0.1, 0.2],  # Step size shrinkage used in update to prevent overfitting
        'max_depth': [3,  5, 7, 9,  11],  # Maximum depth of the individual estimators
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'max_features': [ 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    }


    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)
    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/gb.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as gb.pkl")
    
    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned Gradient Boosting ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)
    
    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    
    
    conf_matrix = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm=conf_matrix, target_names=['Trojan Free','Trojan Infected'], title='Tuned Gradient Boosting:Confusion matrix')
    
    return tuned_time, tuned_accuracy
    


def support_vector_machine():
    """
    This function performs classification with support vector machine
    """
    train_x, test_x, train_y, test_y = prepare_data()
    train_y = train_y.reshape((train_y.shape[0], ))
  
    clf = SVC(kernel="rbf", C=10, gamma=1)

    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()

    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)
    
    print("### SVM ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("For C : ", 10, ", Gamma: ", 1, ", kernel = rbf",
          " => Accuracy = %.2f" % (accuracy))
        

    # Define the parameter grid for tuning SVM
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],  # Regularization parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type to be used in the algorithm
        'degree': [2, 3, 4, 5],  # Degree of the polynomial kernel function (only relevant for 'poly')
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        'class_weight': [None, 'balanced'],  # Weights associated with classes
        'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1)
    tuned_start = time.time()
    grid_search.fit(train_x, train_y)  # Fit the model with the best parameters
    tuned_end = time.time()

    # Save the best model
    best_clf = grid_search.best_estimator_

    with open('param_tuned_models/svm.pkl', 'wb') as f:
        pickle.dump(best_clf, f)
    print("Saved model to disk as svm.pkl")
    
    y_pred = best_clf.predict(test_x)

    tuned_time = tuned_end - tuned_start
    tuned_accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### Param tuned Support Vector Machine ###\n")
    print("Training lasted %.2f seconds" % tuned_time)
    print("Accuracy = %.2f" % (tuned_accuracy))
    print("Best parameters: ", grid_search.best_params_)
    
    print("### Difference ###\n")
    print("extra time in training(tuned-orig): %.2f" % (tuned_time - time_))
    print("Difference in accuracy(tuned-orig): %.2f" % (tuned_accuracy - accuracy))
    
    
    return tuned_time, tuned_accuracy


# main
if __name__ == '__main__':
    # Define the user's preferred method
    if sys.argv[1] == 'svm':
        svm_time, svm_accuracy = support_vector_machine()
    elif sys.argv[1] == 'random_forest':
        rf_time, rf_accuracy = random_forest()
    elif sys.argv[1] == 'mlp':
        mlp_time, mlp_accuracy = multilayer_perceptron()
    elif sys.argv[1] == 'gradient_boosting':
        grad_time, grad_accuracy = gradient_boosting()
    elif sys.argv[1] == 'k_neighbors':
        k_time, k_accuracy = k_neighbors()
    elif sys.argv[1] == 'logistic_regression':
        log_time, log_accuracy = logistic_regression()
    elif sys.argv[1] == 'xgboost':
        xg_time, xg_accuracy = xgboost()
    elif sys.argv[1] == 'comparative':
        svm_time, svm_accuracy = support_vector_machine()
        rf_time, rf_accuracy = random_forest()
        mlp_time, mlp_accuracy = multilayer_perceptron()
        grad_time, grad_accuracy = gradient_boosting()
        k_time, k_accuracy = k_neighbors()
        log_time, log_accuracy = logistic_regression()
        xg_time, xg_accuracy = xgboost()

        accuracy = [svm_accuracy, rf_accuracy, mlp_accuracy, grad_accuracy,
                    k_accuracy, log_accuracy, xg_accuracy]
        time_ = [svm_time, rf_time, mlp_time, grad_time, k_time, log_time, xg_time]

        plt.ylim(0, 100)
        plt.xlabel("accuracy ")
        plt.title("Comparison of performance")
        l1, l2, l3, l4, l5, l6, l8 = plt.bar(["SVM-acc", "RF-acc", "MLP-acc",
                                                  "GB-acc", "K-acc", "log-acc",
                                                  "xg-acc"],
                                                 accuracy)
        
        plt.xticks(rotation=45)
        
        l1.set_facecolor('r')
        l2.set_facecolor('r')
        l3.set_facecolor('r')
        l4.set_facecolor('r')
        l5.set_facecolor('r')
        l6.set_facecolor('r')
        l8.set_facecolor('r')
        
        plt.show()
        plt.close('all')
        plt.ylim(0, 10)
        plt.xlabel("execution time")
        plt.title("Comparison of performance")
        c1, c2, c3, c4, c5, c6, c8 = plt.bar(["SVM-time", "RF-time", "MLP-time",
                                                  "GB-time", "K-time", "log-time",
                                                  "xg-time"],
                                                 time_)
        c1.set_facecolor('b')
        c2.set_facecolor('b')
        c3.set_facecolor('b')
        c4.set_facecolor('b')
        c5.set_facecolor('b')
        c6.set_facecolor('b')
        c8.set_facecolor('b')
        plt.xticks(rotation=45)
        plt.show()        
    elif sys.argv[1] == 'help':
        with open('help.txt', 'r') as f:
            print(f.read())
    elif sys.argv[1] == 'predict':
        predict(sys.argv[2])
    else:
        print("None algorithm was given from input")
        exit
    