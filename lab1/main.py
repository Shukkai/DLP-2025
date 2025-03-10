from data import generate_linear, generate_XOR_easy
from model import Linear_Model, Conv_Model, Linear_wo_active_Model
from plotting import show_loss, show_result
import numpy as np

def train_test(experiment, layer_type = "linear", epochs = 100001,hidden_size=4, active = "sigmoid", optim = "sgd", lr = 1e-4):
    print("current data: "+experiment)
    if experiment == "linear":
        X, y = generate_linear(100)
    else:
        X, y =generate_XOR_easy()
    if layer_type == "linear":
        model = Linear_Model(X, y, lr = lr,hidden_size=hidden_size, activate = active, optim= optim)
    elif layer_type == "conv":
        model = Conv_Model(X,y,lr=lr,conv_filters=hidden_size, activate_conv=active,optim=optim)
    else:
        model = Linear_wo_active_Model(X, y)
    losses = []
    print("####training####")
    for epoch in range(epochs):
        loss = model.backward()
        losses.append(loss)
        if epoch % 5000 == 0:
            print(f"epoch {epoch} loss : {loss:.10f}")
    show_loss(losses, experiment+"_"+active+"_"+optim+"_loss.png")
    print("####testing####")
    if experiment == "linear":
        X_test, y_test = generate_linear(100)
    else:
        X_test, y_test = generate_XOR_easy()
    pred_Y = model.forward(X_test)
    pred_label = np.where(pred_Y > 0.5, 1, 0)
    for i in range(X_test.shape[0]):
        print(f"Iter: {i} |\t Ground truth: {y_test[i][0]} |\t Predict: {pred_Y[i][0]:.10f}")
    print(f"loss={losses[-1]:.10f} accuracy={np.mean(pred_label == y_test)*100}%")
    show_result(X_test, y_test, pred_label)

if __name__ == "__main__":
    """basic"""
    train_test("linear")
    train_test("xor")

    """no activation function"""
    # train_test("linear", layer_type="none",lr=1e-6,hidden_size=2)
    # train_test("xor",layer_type="none",lr=1e-6,hidden_size=2)

    """different activation"""
    # train_test("linear", layer_type="linear", active="sigmoid")
    # train_test("linear", layer_type="linear", active="relu")
    # train_test("linear", layer_type="linear", active="tan")
    # train_test("xor",active="sigmoid")
    # train_test("xor",active="tan")
    # train_test("xor",active="relu")

    """different optimizer"""
    # train_test("linear",optim="sgd")
    # train_test("linear", optim="ada")
    # train_test("linear", optim="mom")
    # train_test("xor",optim="sgd")
    # train_test("xor",optim="ada")
    # train_test("xor",optim="mom")

    """different lr"""
    # train_test("linear",lr=1e-3)
    # train_test("linear",lr=1e-6)
    # train_test("xor",lr= 1e-3)
    # train_test("xor",lr= 1e-6)

    """different hidden units"""
    # train_test("linear",hidden_size=4)
    # train_test("linear",hidden_size=10)
    # train_test("xor",hidden_size=4)
    # train_test("xor",hidden_size=10)

    """conv layer"""
    # train_test("linear",layer_type="conv",lr=1e-2)
    # train_test("xor",layer_type="conv",lr=1e-1)