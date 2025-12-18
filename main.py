import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import Ridge_Regression, Logistic_Regression, SimpleSet
import helpers as hlp
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier


def read_files (path_train="datasets/train.csv", path_test = "datasets/test.csv", path_validation = "datasets/validation.csv"):
    #get the data 
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    validation = pd.read_csv(path_validation)
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
                                 ,title=f"best lambda {float(best_lambda)} with accuracy {float(lambs[best_lambda]['test'])}",file_name="best_model")
    hlp.plot_decision_boundaries(model=worst_model, X=X_train.to_numpy(),y=Y_train.to_numpy(), title=f"worst lambda {float(worst_lambda)} with accuracy {float(lambs[worst_lambda]['test'])}", file_name="worst_model")
    create_grad_decent_with_np_on_specific_function()
    
    # Logistic Regression Part
    print("\n--- Logistic Regression ---")
    # Reload data to get 0/1 labels (read_files returns 0/1, but main transformed them earlier)
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_files()
    train_logistic_regression(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)

    # multy-class regression. 
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_files(path_train="datasets/train_multiclass.csv", path_test="datasets/test_multiclass.csv", path_validation='datasets/validation_multiclass.csv')
    learning_rates = [0.01, 0.001,0.003]
    epochs = 30
    decay = 0.3
    train_logistic_regression(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, learning_rates=learning_rates, epochs=epochs, decay=decay)

    decision_tree_tasks()



def train_logistic_regression(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, learning_rates =[0.1, 0.01, 0.001], batch_size =32, epochs=10, decay=1.0):
    classes_k = pd.unique(Y_train).max() + 1
    train_ds = SimpleSet(X_train.to_numpy(),Y_train.to_numpy())
    validation_ds = SimpleSet(X_validation.to_numpy(),Y_validation.to_numpy())
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    validation_loader = DataLoader(validation_ds,batch_size=batch_size, shuffle=False)
    X_test_t = torch.from_numpy(X_test.to_numpy()).float()
    Y_test_t = torch.from_numpy(Y_test.to_numpy()).long()
    best_loss= float('inf')
    best_loss_lr=-1
    best_acc_tst = -1
    best_acc=-1
    best_epoch = -1
    best_model_state_dict=None
    var_list = ['train_loss', 'validation_loss', 'test_loss','train_acc', 'validation_acc','test_acc']
    complete_output = {lr:{i : None for i in range(epochs)} for lr in learning_rates}
    for leraning_rate in learning_rates:
        model = Logistic_Regression(X_train.shape[1],classes_k)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=leraning_rate )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay)
        for i in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss = criterion(y_pred,batch_y)
                loss.backward()
                optimizer.step()
            lr_scheduler.step() 
            model.eval()
            with torch.no_grad():
                validation_pred = model(validation_ds.X)
                training_pred = model(train_ds.X)
                test_pred = model (X_test_t)
                train_loss = criterion(training_pred, train_ds.Y)
                validation_loss = criterion(validation_pred, validation_ds.Y)
                test_loss = criterion(test_pred, Y_test_t)
                train_acc, validation_acc, test_acc = take_accuracy(training_pred,validation_pred,test_pred, train_ds.Y, validation_ds.Y,Y_test_t)
                if (validation_loss < best_loss ):
                    best_loss = validation_loss
                    best_acc = validation_acc
                    best_epoch = i 
                    best_loss_lr = leraning_rate
                    best_acc_tst = test_acc
                    best_model_state_dict = model.state_dict()
                
                # Capture locals to avoid scope issues in comprehension
                current_locals = locals()
                full_data_epoch = {k: current_locals[k].item() if hasattr(current_locals[k], 'item') else current_locals[k] for k in var_list}
                complete_output[leraning_rate][i]=full_data_epoch
            model.train()

    best_model = Logistic_Regression (X_train.shape[1],classes_k)
    best_model.load_state_dict(best_model_state_dict)
    best_model.eval()

    #print nice output to the terminal of all the results, of the data of the best model and so on and so forth
    print(f"\n{'='*20} Training Complete {'='*20}")
    print(f"Best Model found at Epoch {best_epoch+1} with LR {best_loss_lr}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"best model validation accuracy:{best_acc:.4f}")
    print(f"best model test accuracy: {best_acc_tst:.4f}")
    # Retrieve history for the best LR
    best_history = complete_output[best_loss_lr]
    
    # Extract lists for plotting
    epochs_range = range(1, epochs + 1)
    train_losses = [best_history[i]['train_loss'] for i in range(epochs)]
    val_losses = [best_history[i]['validation_loss'] for i in range(epochs)]
    test_losses = [best_history[i]['test_loss'] for i in range(epochs)]
    
    train_accs = [best_history[i]['train_acc'] for i in range(epochs)]
    val_accs = [best_history[i]['validation_acc'] for i in range(epochs)]
    test_accs = [best_history[i]['test_acc'] for i in range(epochs)]

    #make a plot of the all the sets loss as function of the epochs of the winning lr 
    plt.figure(figsize=(14, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.plot(epochs_range, test_losses, label='Test Loss', marker='o')
    plt.title(f'Loss vs Epochs (LR={best_loss_lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # make the same for accuracy for all the data in a matter you see fit
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='s')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy', marker='s')
    plt.plot(epochs_range, test_accs, label='Test Accuracy', marker='s')
    plt.title(f'Accuracy vs Epochs (LR={best_loss_lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_metrics.png')
    plt.close()

    # New Plot: Accuracy vs Learning Rate
    best_val_accs = []
    best_test_accs = []
    sorted_lrs = sorted(learning_rates)
    
    for lr in sorted_lrs:
        # Find the epoch with the best validation loss for this LR
        best_epoch_lr = -1
        min_val_loss = float('inf')
        
        for epoch, metrics in complete_output[lr].items():
            if metrics and metrics['validation_loss'] < min_val_loss:
                min_val_loss = metrics['validation_loss']
                best_epoch_lr = epoch
        
        # Get the accuracies at that best epoch
        if best_epoch_lr != -1:
            best_metrics = complete_output[lr][best_epoch_lr]
            best_val_accs.append(best_metrics['validation_acc'])
            best_test_accs.append(best_metrics['test_acc'])
        else:
            best_val_accs.append(0)
            best_test_accs.append(0)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lrs, best_val_accs, marker='o', label='Validation Accuracy')
    plt.plot(sorted_lrs, best_test_accs, marker='s', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate (Best Model per LR)')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_accuracy_plot.png')
    plt.close()

    # Report best model
    max_val_acc = -1
    best_idx = -1
    for i, val_acc in enumerate(best_val_accs):
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_idx = i
            
    if best_idx != -1:
        print(f"\nBest Model found at Learning Rate: {sorted_lrs[best_idx]}")
        print(f"Validation Accuracy: {best_val_accs[best_idx]:.4f}")
        print(f"Test Accuracy: {best_test_accs[best_idx]:.4f}")

    print("\n" + "="*20 + " Full Training Results " + "="*20)
    all_results = []
    for lr, epochs_data in complete_output.items():
        for epoch, metrics in epochs_data.items():
            if metrics:
                row = {'Learning Rate': lr, 'Epoch': epoch}
                # Round values for nicer display
                row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
                all_results.append(row)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        # Define column order
        cols = ['Learning Rate', 'Epoch', 'train_loss', 'validation_loss', 'test_loss', 'train_acc', 'validation_acc', 'test_acc']
        # Ensure columns exist
        cols = [c for c in cols if c in df_results.columns]
        print(df_results[cols].to_string(index=False))
    
    hlp.plot_decision_boundaries(best_model, X_test.to_numpy(), Y_test.to_numpy(), title=f"epoch {best_epoch+1}, best accuracy in validation set {best_acc}")


def take_accuracy (training_pred,validation_pred,test_pred, train_Y, validation_Y, Y_test_t):
    training_pred = torch.argmax(training_pred, dim=-1)
    validation_pred = torch.argmax(validation_pred, dim=-1)
    test_pred = torch.argmax(test_pred,dim=-1)
    train_acc = ((train_Y==training_pred).float().mean().item()) 
    validation_acc = ((validation_Y==validation_pred).float().mean().item()) 
    test_acc = ((Y_test_t==test_pred).float().mean().item()) 
    return train_acc ,validation_acc, test_acc
                                        
        

def decision_tree_tasks():
    print("\n" + "="*20 + " Decision Tree Tasks " + "="*20)
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_files(path_train="datasets/train_multiclass.csv", path_test="datasets/test_multiclass.csv", path_validation='datasets/validation_multiclass.csv')
    
    # Task 3: max_depth=2
    print("\nTask 3: Decision Tree with max_depth=2")
    dt_2 = DecisionTreeClassifier(max_depth=2, random_state=0)
    dt_2.fit(X_train, Y_train)
    
    train_acc_2 = dt_2.score(X_train, Y_train)
    val_acc_2 = dt_2.score(X_validation, Y_validation)
    test_acc_2 = dt_2.score(X_test, Y_test)
    
    print(f"Train Accuracy: {train_acc_2:.4f}")
    print(f"Validation Accuracy: {val_acc_2:.4f}")
    print(f"Test Accuracy: {test_acc_2:.4f}")
    
    hlp.plot_decision_boundaries(dt_2, X_train.to_numpy(), Y_train.to_numpy(), title="Decision Tree (depth=2)", file_name="dt_depth_2")
    
    # Task 4: max_depth=10
    print("\nTask 4: Decision Tree with max_depth=10")
    dt_10 = DecisionTreeClassifier(max_depth=10, random_state=0)
    dt_10.fit(X_train, Y_train)
    
    train_acc_10 = dt_10.score(X_train, Y_train)
    val_acc_10 = dt_10.score(X_validation, Y_validation)
    test_acc_10 = dt_10.score(X_test, Y_test)
    
    print(f"Train Accuracy: {train_acc_10:.4f}")
    print(f"Validation Accuracy: {val_acc_10:.4f}")
    print(f"Test Accuracy: {test_acc_10:.4f}")
    
    hlp.plot_decision_boundaries(dt_10, X_train.to_numpy(), Y_train.to_numpy(), title="Decision Tree (depth=10)", file_name="dt_depth_10")





def create_grad_decent_with_np_on_specific_function():
    #f(x,y) = (x-3)^2 + (y-5)^2
    #eta =0.1
    #plot the vector through the iterations where  x is x axis  y is y exis and the iterations have different colors
    
    # Vectorized initialization
    w = np.array([0.0, 0.0])  # Starting point [x, y]
    target = np.array([3.0, 5.0])
    eta = 0.1
    epochs = 1000
    
    trajectory = [w.copy()]
    
    # Gradient Descent Loop
    for _ in range(epochs):
        # Gradient of (x-3)^2 + (y-5)^2 is [2(x-3), 2(y-5)]
        grad = 2 * (w - target)
        w -= eta * grad
        trajectory.append(w.copy())
    
    trajectory = np.array(trajectory)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Background contour
    x_vals = np.linspace(min(trajectory[:,0])-1, max(trajectory[:,0])+1, 100)
    y_vals = np.linspace(min(trajectory[:,1])-1, max(trajectory[:,1])+1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = (X - 3)**2 + (Y - 5)**2
    plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
    
    # Trajectory
    # Color points by iteration index, using a colormap that transitions (e.g., light to dark)
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.arange(len(trajectory)), cmap='Reds', s=20, edgecolors='k', linewidths=0.5)
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='gray')
    
    plt.colorbar(label='Iteration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Trajectory')
    plt.grid(True)
    plt.savefig('gradient_descent.png')
    plt.close()

    # Find the point with the lowest function value in the trajectory
    # f(w) = ||w - target||^2
    diff = trajectory - target
    function_values = np.sum(diff**2, axis=1)
    best_idx = np.argmin(function_values)
    best_point = trajectory[best_idx]
    val = function_values[best_idx]
    
    print(f"Best point found (at iteration {best_idx}): x={best_point[0]}, y={best_point[1]}, f(x,y)={val}")


if __name__ == "__main__":
    main()
