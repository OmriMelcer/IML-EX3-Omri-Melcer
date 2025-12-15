import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import Ridge_Regression
import helpers as hlp


def read_files ():
    #get the data 
    train = pd.read_csv("datasets/train.csv")
    test = pd.read_csv("datasets/test.csv")
    validation = pd.read_csv("datasets/validation.csv")
    X_train = train[['long','lat']]
    Y_train = train['country']
    X_validation = validation[['long','lat']]
    Y_validation = validation['country']
    X_test = test[['long','lat']]
    Y_test = test['country']    
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def turn_to_minus_one_and_one(Y):
    return 2 * (Y - 0.5)

def calculate_accuracy(Y_true, Y_pred):
    correct = np.sum(Y_true == Y_pred)
    total = len(Y_true)
    return correct / total

def plot_accuracy_vs_lambdas(lambs:dict[float:dict[str:float]]):
    lambdas_nums = sorted(list(lambs.keys()))
    train_accuracies = [lambs[lamb]['train'] for lamb in lambdas_nums]
    validation_accuracies = [lambs[lamb]['validation'] for lamb in lambdas_nums]
    test_accuracies = [lambs[lamb]['test'] for lamb in lambdas_nums]
    
    plt.figure()
    plt.plot(lambdas_nums, train_accuracies, label='Train Accuracy', marker='o', color='blue')
    plt.plot(lambdas_nums, validation_accuracies, label='Validation Accuracy', marker='o', color='green')
    plt.plot(lambdas_nums, test_accuracies, label='Test Accuracy', marker='o', color='red')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.close()

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_files()
    #create model
    lambs ={0.:{'train':0,'validation':0, 'test':0},2.:{'train':0,'validation':0, 'test':0},4.:{'train':0,'validation':0, 'test':0},6.:{'train':0,'validation':0, 'test':0},8.:{'train':0,'validation':0, 'test':0},10.:{'train':0,'validation':0, 'test':0}}
    Y_train = turn_to_minus_one_and_one(Y_train)
    Y_validation = turn_to_minus_one_and_one(Y_validation)
    Y_test = turn_to_minus_one_and_one(Y_test)
    predictions_per_lamb_on_test = {}
    for lamb in lambs.keys():
        model = Ridge_Regression(lamb)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_validation)
        accuracy = calculate_accuracy(Y_validation, Y_pred)
        lambs[lamb]['validation'] = accuracy
        Y_pred_train = model.predict(X_train)
        accuracy_train = calculate_accuracy(Y_train, Y_pred_train)
        lambs[lamb]['train'] = accuracy_train
        Y_pred_test = model.predict(X_test)
        accuracy_test = calculate_accuracy(Y_test, Y_pred_test)
        lambs[lamb]['test'] = accuracy_test
        print(f"Lambda: {lamb}, Train Accuracy: {accuracy_train}, Validation Accuracy: {accuracy}, Test Accuracy: {accuracy_test}")
        predictions_per_lamb_on_test[lamb] = Y_pred_test
    plot_accuracy_vs_lambdas(lambs)
    val_results_per_lamb = {lamb: lambs[lamb]['validation'] for lamb in lambs.keys()}
    best_lambda = max(val_results_per_lamb, key=val_results_per_lamb.get)
    worst_lambda = min(val_results_per_lamb, key=val_results_per_lamb.get)
    print (f"worst lambda for validation {worst_lambda}, with accuracy {val_results_per_lamb[worst_lambda]}")
    print (f"best lambda for validation {best_lambda}, with accuracy {val_results_per_lamb[best_lambda]}")
    best_model = Ridge_Regression(best_lambda)
    best_model.fit(X_train, Y_train)
    worst_model = Ridge_Regression(worst_lambda)
    worst_model.fit(X_train, Y_train)
    hlp.plot_decision_boundaries(model=best_model, X=X_train.to_numpy(),y=Y_train.to_numpy()
                                 ,title=f"best lambda {best_lambda} with accuracy {lambs[best_lambda]}")
    hlp.plot_decision_boundaries(model=worst_model, X=X_train.to_numpy(),y=Y_train.to_numpy(), title=f"worst lambda {worst_lambda} with accuracy {lambs[worst_lambda]}")





if __name__ == "__main__":
    main()
