import matplotlib.pyplot as plt

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Groung truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()

def show_loss(losses, filename="loss_plot.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss", color='b', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # Save the figure
    # plt.savefig(filename, dpi=300, bbox_inches='tight')  
    # print(f"Plot saved as {filename}")
    plt.show()
