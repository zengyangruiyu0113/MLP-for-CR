from scipy.stats import gaussian_kde
from scipy.stats import poisson
import keras.optimizers
from keras.models import Sequential,load_model
from keras.layers.core import Dense,Activation
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import keras.backend as K
import tkinter as tk
from tkinter import filedialog

class DataDenoising:
    def __init__(self):
        self.data = {}  # Store data
        self.data_kinds_label = {}  # Store data types
        self.model = None
        self.history =None

    def read_excel(self, path):
        df = pd.read_excel(path)
        self.data["真实数据"] = df.values
        return df

    def save_excel(self, data):
        path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        data.to_excel(path, index=False)

    def df_one_hot_encoded(self, data, columns_name):
        return pd.get_dummies(data, columns=[columns_name])

    def numpy_one_hot_encoded(self, data):
        unique_categories = np.unique(data)
        one_hot_encoded = np.zeros((data.shape[0], unique_categories.shape[0]))
        for i, category in enumerate(data):
            index = np.where(unique_categories == category)[0][0]
            one_hot_encoded[i, index] = 1
        return one_hot_encoded

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def make_noise_data(self, data_scale, data_size):
        self.set_random_seed(320)
        values = np.array(data_scale)
        # Random Data
        random_indices = np.random.randint(0, len(values), data_size)
        self.data["随机数据"] = values[random_indices]
        # Longstring
        random_indices2 = np.random.randint(0, len(values), data_size[0])
        random_array2 = values[random_indices2].reshape((data_size[0], 1))
        self.data["相同数据"] = np.repeat(random_array2, repeats=data_size[1], axis=1)
        # Pattern Data 1
        quotient, remainder = divmod(data_size[1], len(data_scale))
        list1 = data_scale * quotient + data_scale[:remainder]
        self.data["规律数据1"] = np.tile(np.array(list1), (data_size[0] // 4, 1))
        # Pattern Data 2
        data_scale_reverse = list(reversed(data_scale))
        quotient2, remainder2 = divmod(quotient, 2)
        if remainder2 == 0:
            list2 = (data_scale + data_scale_reverse) * quotient2 + data_scale[:remainder]
        else:
            list2 = (data_scale + data_scale_reverse) * quotient2 + data_scale + data_scale_reverse[:remainder]
        self.data["规律数据2"] = np.tile(np.array(list2), (data_size[0] // 4, 1))

    def merge_data(self):
        df_list = []
        for n, (k, v) in enumerate(self.data.items()):
            df = pd.DataFrame(data=v)
            df["数据类型"] = n
            self.data_kinds_label[n] = k
            df_list.append(df)
        self.merged_df = pd.concat(df_list, axis=0) if df_list else pd.DataFrame()

    def save_to_excel(self, filename):
        df_random = pd.DataFrame(self.data["随机数据"], columns=[f'随机数据_{i+1}' for i in range(self.data["随机数据"].shape[1])])
        df_same = pd.DataFrame(self.data["相同数据"], columns=[f'相同数据_{i+1}' for i in range(self.data["相同数据"].shape[1])])
        df_pattern1 = pd.DataFrame(self.data["规律数据1"], columns=[f'规律数据1_{i+1}' for i in range(self.data["规律数据1"].shape[1])])
        df_pattern2 = pd.DataFrame(self.data["规律数据2"], columns=[f'规律数据2_{i+1}' for i in range(self.data["规律数据2"].shape[1])])
        df = pd.concat([df_random, df_same, df_pattern1, df_pattern2], axis=1)
        df.to_excel(filename, index=False)

    def train_test_data(self, itemsnum=20, k=5):
        data = self.merged_df.to_numpy()
        np.random.shuffle(data)
        x_data = data[:, :itemsnum]
        for i in range(itemsnum):
            x_data[:, i] = (x_data[:, i] - np.mean(x_data[:, i])) / np.std(x_data[:, i])

        y_data = self.numpy_one_hot_encoded(data[:, itemsnum:])
        # kf = KFold(n_splits=k)
        # accuracies = []
        #
        # for train_index, test_index in kf.split(x_data):
        #     self.X_train, self.X_test = x_data[train_index], x_data[test_index]
        #     self.T_train, self.T_test = y_data[train_index], y_data[test_index]
        #     self.build_model(itemsnum=itemsnum, middle_struction=(30, 30, 30, 30), epochs=30)
        #     accuracy = self.evaluate_model()
        #     accuracies.append(accuracy)
        #
        # print("Cross-validation accuracies:", accuracies)
        # print("Average accuracy:", np.mean(accuracies))
        kf = KFold(n_splits=k)

        # 假设这里只需要第一个分组作为训练和测试，不循环使用所有分组
        train_index, test_index = next(kf.split(x_data))  # 仅取第一个分组
        self.X_train, self.X_test = x_data[train_index], x_data[test_index]
        self.T_train, self.T_test = y_data[train_index], y_data[test_index]

    def build_model(self, itemsnum, middle_struction, activation="sigmoid", epochs=1000, batch_size=100, seed=None):
        if seed:
            np.random.seed(seed)

        self.model = Sequential()
        input_dim = itemsnum
        for neuro_num in middle_struction:
            self.model.add(Dense(neuro_num, input_dim=input_dim, activation=activation, kernel_initializer="uniform"))
            input_dim = neuro_num

        self.model.add(Dense(len(self.data_kinds_label), activation='softmax', kernel_initializer='uniform'))
        sgd = keras.optimizers.SGD(learning_rate=0.1)
        self.model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.T_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(self.X_test, self.T_test))

        self.Y_test = self.model.predict(self.X_test)
        self.metrics()

    def evaluate_model(self):
        score = self.model.evaluate(self.X_test, self.T_test, verbose=0)
        return score[1]

    def model_predict(self, array_data):
        results = self.model.predict(array_data)
        kinds = np.argmax(results, axis=1)
        kinds = [self.data_kinds_label[x] for x in kinds]
        return kinds

    def feature_importance_show(self):
        self.score_list = []
        for char in range(self.X_test.shape[1]):
            data = self.X_test.copy()
            np.random.shuffle(data[:, char])
            score_shuffle = self.model.evaluate(data, self.T_test, verbose=0)
            self.score_list.append(score_shuffle[0])

    def precision(self):
        tp = K.sum(K.round(K.clip(self.T_test * self.Y_test, 0, 1)))
        pp = K.sum(K.round(K.clip(self.Y_test, 0, 1)))
        self.precision_value = tp / (pp + K.epsilon())
        return self.precision_value

    def recall(self):
        tp = K.sum(K.round(K.clip(self.T_test * self.Y_test, 0, 1)))
        pp = K.sum(K.round(K.clip(self.T_test, 0, 1)))
        self.recall_value = tp / (pp + K.epsilon())
        return self.recall_value

    def f1_score(self):
        self.f1_value = 2 * ((self.precision_value * self.recall_value) / (self.precision_value + self.recall_value + K.epsilon()))
        return self.f1_value

    def metrics(self):
        self.X_test = self.X_test.astype(np.float32)
        self.T_test = self.T_test.astype(np.float32)
        loss, acc = self.model.evaluate(self.X_test, self.T_test)
        self.precision()
        self.recall()
        self.f1_score()
        self.metrics_values = {"loss": loss,
            "acc": acc,
            "precision": self.precision_value.numpy(),
            "recall": self.recall_value.numpy(),
            "F1": self.f1_value.numpy()}
        print(self.metrics_values)
        return self.metrics_values

    def show_cross_validation_history(self, histories):
        num_folds = len(histories)
        plt.figure(figsize=(12, 8))

        for i in range(num_folds):
            fold_history = histories[i]

            # Loss Plot
            plt.subplot(2, num_folds, i + 1)
            plt.plot(fold_history.history["loss"], "black", label="training")
            plt.plot(fold_history.history["val_loss"], "cornflowerblue", label="validation")
            plt.title(f"Fold {i + 1} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            # Accuracy Plot
            plt.subplot(2, num_folds, num_folds + i + 1)
            plt.plot(fold_history.history["accuracy"], "black", label="training")
            plt.plot(fold_history.history["val_accuracy"], "cornflowerblue", label="validation")
            plt.title(f"Fold {i + 1} Accuracy")
            plt.xlabel("Epoch")

    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def show_training_history(self):
        if self.history is not None:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history["loss"], "black", label="training")
            plt.plot(self.history.history["val_loss"], "cornflowerblue", label="validation")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.history.history["accuracy"], "black", label="training")
            plt.plot(self.history.history["val_accuracy"], "cornflowerblue", label="validation")
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.tight_layout()
            plt.show()
    def get_weights(self):
        self.layers_weights = [layer.get_weights() for layer in self.model.layers]

    def save_model(self):
        path = filedialog.aveasfilename(defaultextension=".h5")
        self.model.save(path)

    def load_model(self):
        path = filedialog.askopenfilename()
        self.model = load_model(path)

class Data_analysis:
    def __init__(self, data):
        self.data = data
    def compute_std(self,data):#计算标准差
        return np.std(data,axis=1)
    def compute_dis(self,data,items):#计算正向题目与反向题目的距离，距离越大，数据越可靠
        dis = np.zeros((len(data),))
        for item in items:
            dis+=((data[:,item]-data[:,item+1])**2)**0.5
        return dis
    def make_hist_plot(self,data_list,distribution_function_type):
        #创建一个包含两个子图的画布
        fig, ax = plt.subplots()
        num=0
        for data in data_list:
            num+=1
            color=(random.random(),random.random(),random.random())
            # 绘制数据1的直方图
            if distribution_function_type=="高斯":
                # 根据数据生成高斯核密度函数
                ax.hist(data, bins=30,color=color, alpha=0.7, label='Data' + str(num))
                kde = gaussian_kde(data)
                # 绘制数据1的拟合曲线
                x_vals = sorted(set(data))
                ax.plot(x_vals, kde(x_vals), color=color, linestyle='--', label='Gaussian Distribution'+ str(num))
            elif distribution_function_type=="泊松":
                values, counts = np.unique(data, return_counts=True)
                freq = counts / len(data)
                ax.bar(values, freq, color=color, alpha=0.7, label='Data Frequency'+ str(num))
                mu=np.mean(data)
                poisson_dist = poisson(mu=mu)
                pmf_values = poisson_dist.pmf(values)
                ax.plot(values, pmf_values, color=color, marker='o', linestyle='-', label='Poisson Distribution'+ str(num))

        # 添加标题和标签
        ax.set_title('Distribution of Data')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        # 添加图例
        ax.legend()
        # 显示图形
        plt.show()


#使用方法
random.seed(123)
project=DataDenoising()
project.read_excel(path="Depression.xlsx")
project.set_random_seed(320)
project.make_noise_data(data_size=(10000,20),data_scale=[0,1,2,3])#生成噪声数据
print(project.data)
project.save_to_excel("noise&true_depression.xlsx")
project.merge_data()
project.train_test_data(itemsnum=20)#生成模型训练数据集和测试集
project.build_model(middle_struction=(30,30,30,30),itemsnum=20,activation="relu",epochs=30)#建模
project.model.save("Data_denoising_depression_new3.h5")
project.show_training_history()#显示训练过程结果图片
