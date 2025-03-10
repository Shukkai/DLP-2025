from data import generate_linear, generate_XOR_easy
from model import Linear_Model, Conv_Model, Linear_wo_active_Model
from plotting import show_loss, show_result
import numpy as np

def train_test(experiment, layer_type = "linear", epochs = 100001, active = "sigmoid", optim = "sgd", lr = 1e-6):
    print(experiment+" "+ layer_type)
    if experiment == "linear":
        X, y = generate_linear(100)
    else:
        X, y =generate_XOR_easy()
    if layer_type == "linear":
        model = Linear_Model(X, y, lr = lr, activate = active, optim= optim)
    elif layer_type == "conv":
        model = Conv_Model(X,y,lr=lr,activate=active,optim=optim)
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
    # print("####testing####")
    if experiment == "linear":
        X_test, y_test = generate_linear(100)
    else:
        X_test, y_test = generate_XOR_easy()
    pred_Y = model.forward(X_test)
    pred_label = np.where(pred_Y > 0.5, 1, 0)
    for i in range(X_test.shape[0]):
        print(f"Iter: {i} |\t Ground truth: {y_test[i][0]} |\t Predict: {pred_Y[i][0]:.10f}")
    print(f"loss={losses[-1]:.10f} accuracy={np.mean(pred_label == y_test)*100}%")
    # show_result(X_test, y_test, pred_label)

if __name__ == "__main__":
    train_test("linear", layer_type="linear", active="sigmoid", lr = 1e-6)
    # train_test("linear", layer_type="none", lr = 1e-6)
    # train_test("linear", layer_type="linear", optim="ada", lr = 1e-6)
    # train_test("linear", layer_type="linear", optim="sgd", lr = 1e-6)
    # train_test("linear", layer_type="linear", optim="sgd", lr = 1e-3)
    # train_test("xor",optim="sgd",lr= 1e-3)
    train_test("xor",optim="sgd",lr= 1e-6)
    # train_test("xor",optim="sgd",lr= 0.01)
    # train_test("xor",optim="ada",lr= 0.01)
    # train_test("xor",optim="mom",lr= 0.01)
    # train_test("linear", layer_type="linear", active="sigmoid", lr = 1e-6)
    # train_test("linear", layer_type="linear", active="relu", optim="sigmoid", lr = 1e-6)
    # train_test("xor",layer_type="none",lr=1e-3)
    # train_test("xor",active="sigmoid",lr= 0.01)
    # train_test("xor",active="tan",lr= 1e-6)