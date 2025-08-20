from scipy.stats import ttest_rel


def result_test():
    avg_f1_2019 = [
        0.9402,
        0.9473,
        0.9313,
        0.9414,
        0.9427,
        0.9382,
        0.9569,
        0.9595,
        0.9385,
    ]
    avg_acc_2019 = [
        0.9366,
        0.9445,
        0.9263,
        0.9380,
        0.9405,
        0.9343,
        0.9560,
        0.9581,
        0.9346,
    ]
    prox_f1_2019 = [
        0.9581,
        0.9661,
        0.9464,
        0.9485,
        0.9501,
        0.9595,
        0.9584,
        0.9680,
        0.9556,
    ]
    prox_acc_2019 = [
        0.9564,
        0.9653,
        0.9436,
        0.9459,
        0.9477,
        0.9579,
        0.9568,
        0.9670,
        0.9537,
    ]
    adam_f1_2019 = [
        0.8606,
        0.8576,
        0.9389,
        0.8472,
        0.8987,
        0.9452,
        0.8415,
        0.8930,
        0.9520,
    ]
    adam_acc_2019 = [
        0.8416,
        0.8375,
        0.9363,
        0.8260,
        0.8901,
        0.9422,
        0.8218,
        0.8808,
        0.9511,
    ]
    print(adam_acc_2019[2::3])

    avg_f1_2017 = [
        0.9124,
        0.8917,
        0.8690,
        0.9025,
        0.8923,
        0.8704,
        0.8982,
        0.8839,
        0.8702,
    ]
    avg_acc_2017 = [
        0.9071,
        0.8863,
        0.8689,
        0.8964,
        0.8831,
        0.8644,
        0.8922,
        0.8749,
        0.8602,
    ]
    prox_f1_2017 = [
        0.9139,
        0.9073,
        0.8860,
        0.9147,
        0.9009,
        0.8765,
        0.9044,
        0.8940,
        0.8756,
    ]
    prox_acc_2017 = [
        0.9104,
        0.9029,
        0.8776,
        0.9097,
        0.8970,
        0.8733,
        0.8996,
        0.8894,
        0.8649,
    ]
    adam_f1_2017 = [
        0.8082,
        0.8302,
        0.8693,
        0.8042,
        0.8188,
        0.8844,
        0.8006,
        0.7736,
        0.8941,
    ]
    adam_acc_2017 = [
        0.8246,
        0.8312,
        0.8753,
        0.8179,
        0.8110,
        0.8780,
        0.7842,
        0.7480,
        0.8901,
    ]

    print("FedProx CIC-DDoS2019 F1-Score:", ttest_rel(prox_f1_2019, avg_f1_2019))
    print()
    print("FedProx CIC-DDoS2019 Accuracy:", ttest_rel(prox_acc_2019, avg_acc_2019))
    print()
    print("FedProx CIC-IDS2017 F1-Score:", ttest_rel(prox_f1_2017, avg_f1_2017))
    print()
    print("FedProx CIC-IDS2017 Accuracy:", ttest_rel(prox_acc_2017, avg_acc_2017))
    print()
    print("FedAdam CIC-DDoS2019 F1-Score:", ttest_rel(adam_f1_2019, avg_f1_2019))
    print()
    print("FedAdam CIC-DDoS2019 Accuracy:", ttest_rel(adam_acc_2019, avg_acc_2019))
    print()
    print("FedAdam CIC-DDoS2017 F1-Score:", ttest_rel(adam_f1_2017, avg_f1_2017))
    print()
    print("FedAdam CIC-DDoS2017 Accuracy:", ttest_rel(adam_acc_2017, avg_acc_2017))
    print()


if __name__ == "__main__":
    result_test()
