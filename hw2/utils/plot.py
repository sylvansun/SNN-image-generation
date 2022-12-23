import matplotlib.pyplot as plt


def read_file_to_list(file_path):
    with open(file_path, "r") as f:
        return [float(x) for x in f.read().strip("[]").split(", ")]


def smooth_list(list, weight):
    result = [list[0]]
    for i in range(1, len(list)):
        result.append(list[i] * weight + result[i - 1] * (1 - weight))
    return result


snn = read_file_to_list("docs/acc_snn.txt")
snn = smooth_list(snn, 0.8)
cnn = read_file_to_list("docs/acc_cnn.txt")
cnn = smooth_list(cnn, 0.8)
plt.plot(snn, label="SNN")
plt.plot(cnn, label="CNN")
plt.legend()
plt.xlim(-0.2, 19.2)
plt.ylim(0.15, 0.9)
plt.savefig("docs/acc_snn_cnn.png")
