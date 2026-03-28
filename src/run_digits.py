import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SoftmaxRegression,OneHiddenLayerNN
from src.utils import train_model,evaluate
import matplotlib.pyplot as plt

def load_digits_data(root_folder):

    #creating path

    data_file=os.path.join(root_folder,"data","digits_data.npz")
    split_file=os.path.join(root_folder,"data","digits_split_indices.npz")

    #Loading data
    digits_data=np.load(data_file)
    split_inform=np.load(split_file)
    X_data=digits_data["X"]
    y_labels=digits_data["y"]

    #Getting indexs
    train_indx=split_inform["train_idx"]
    val_indx=split_inform["val_idx"]
    test_indx=split_inform["test_idx"]

    #Splitting data
    train_dataset=(X_data[train_indx],y_labels[train_indx])
    val_dataset=(X_data[val_indx],y_labels[val_indx])
    test_dataset=(X_data[test_indx],y_labels[test_indx])

    return train_dataset,val_dataset,test_dataset

def evaluate_multiple_runs(model_cls,model_params,X_train,y_train,X_val,y_val,
    X_test,y_test,num_runs=5,lr=0.05):
    print(f"\nRunning experiments with {num_runs} different seeds for {model_cls.__name__}")
    acc_list=[]
    ce_list=[]

    for i in range(num_runs):
        np.random.seed(i)
        model=model_cls(**model_params)

        #model training

        train_model(model,X_train,y_train,X_val,y_val,epochs=200,batch_size=64,lr=lr,use_best_val=True)

        #testing
        ce_val,acc_val=evaluate(model,X_test,y_test)

        acc_list.append(acc_val)
        ce_list.append(ce_val)

        print(f"[Run {i}] acc={acc_val:.4f}, ce={ce_val:.4f}")

    #Calculations
    acc_mean=np.mean(acc_list)
    acc_std=np.std(acc_list,ddof=1)
    acc_ci=2.776*acc_std /np.sqrt(num_runs)


    ce_mean=np.mean(ce_list)
    ce_std=np.std(ce_list,ddof=1)
    ce_ci=2.776*ce_std /np.sqrt(num_runs)


    print('\nSummary')
    print(f"{model_cls.__name__}->Acc:{acc_mean:.4f}±{acc_ci:.4f}")
    print(f"{model_cls.__name__}->CE:{ce_mean:.4f}±{ce_ci:.4f}")

def run_experiments_on_digits():
    project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    (X_train,y_train),(X_val,y_val),(X_test,y_test)=load_digits_data(project_root)


    input_size=X_train.shape[1]
    n_classes=10

    print("\nModel Comparison (Repeated Runs)")

    # Repeated-seed statistics for Softmax and NN

    evaluate_multiple_runs(SoftmaxRegression,
    {"input_dim":input_size,"num_classes":n_classes,"l2_reg":1e-4},
    X_train,y_train,X_val,y_val,X_test,y_test,num_runs=5,lr=0.1)


    evaluate_multiple_runs(OneHiddenLayerNN,
    {"input_dim":input_size,"hidden_dim":32,"num_classes":n_classes,"l2_reg":1e-4},
    X_train,y_train,X_val,y_val,X_test,y_test,num_runs=5,lr=0.1)

    #Learning Rate ablation on digits (for NN)
    print("\nLearning Rate Experiment(NN)")

    lr_values=[0.005, 0.05, 0.2]
    fig_path=os.path.join(project_root,"figures")
    os.makedirs(fig_path,exist_ok=True)


    plt.figure(figsize=(10,6))

    for i in lr_values:
        np.random.seed(42)
        model=OneHiddenLayerNN(input_size,32,n_classes)

        history=train_model(model,X_train,y_train,X_val,y_val,epochs=100,batch_size=64,
        lr=i,return_history=True,use_best_val=False)

        val_losses=history["val_loss"]
        plt.plot(val_losses,label=f"lr={i}")

        print(f"lr={i}->final val loss: {val_losses[-1]:.4f}")

    plt.title("Validation Loss vs Epochs (Different Learning Rates)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()

    save_file=os.path.join(fig_path,"lr_ablation.png")
    plt.savefig(save_file)
    plt.close()

    print(f"\nFigure saved to:{save_file}")

if __name__ == "__main__":
    run_experiments_on_digits()