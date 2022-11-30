from operator import abs
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot
from tkinter import *
from tkinter.ttk import *

form = Tk()
label1 = StringVar()
label2 = StringVar()
label3 = StringVar()
label4 = StringVar()
label5 = StringVar()
spinData1 = DoubleVar()
spinData2 = IntVar()
radio_var1 = IntVar()
radio_var2 = IntVar()
text1 = IntVar()
text2 = StringVar()
lR = ''
epoch_num = ''
use_bias = True
activation_fun = ''
hidden_num = ''
neurons = []
encodes = {'Adelie': [0, 0, 1], 'Gentoo': [0, 1, 0], 'Chinstrap': [1, 0, 0]}
train_labels = []
test_labels = []
train_data = []
test_data = []
weights = []
delta = []
layers_output = []


# call back when press the run button
def run():
    user_inputs()
    # print(lR)
    # print(epoch_num)
    # print(use_bias)
    # print(activation_fun)
    # print(hidden_num)
    # print(neurons)


# take user values
def user_inputs():
    global lR, epoch_num, use_bias, activation_fun, hidden_num, neurons
    lR = spinData1.get()
    epoch_num = spinData2.get()
    hidden_num = text1.get()
    string_value = text2.get()
    neurons = [int(num) for num in string_value.split()]
    use_bias = True
    if radio_var1.get() == 1:
        use_bias = True
    elif radio_var1.get() == 2:
        use_bias = False

    if radio_var2.get() == 1:
        activation_fun = "Sigmoid"
    elif radio_var2.get() == 2:
        activation_fun = "Tangent"


# call all function that create elements in Gui
def gui():
    form.geometry("450x450")
    form.title("Form")
    create_label()
    create_spinbox()
    create_radio()
    create_button()
    create_text_box()
    form.mainloop()


# create Radio buttons in gui
def create_radio():
    r1 = Radiobutton(form, text="bias", width=120, variable=radio_var1, value=1)
    r1.place(x=120, y=290)

    r2 = Radiobutton(form, text="no bias", width=120, variable=radio_var1, value=2)
    r2.place(x=300, y=290)

    r3 = Radiobutton(form, text="Sigmoid", width=120, variable=radio_var2, value=1)
    r3.place(x=120, y=320)

    r4 = Radiobutton(form, text="Tangent", width=120, variable=radio_var2, value=2)
    r4.place(x=300, y=320)


# create spinbox in gui
def create_spinbox():
    spin1 = Spinbox(form, from_=0, to=1, increment=0.1, width=5, textvariable=spinData1)
    spin1.place(x=120, y=220)

    spin2 = Spinbox(form, from_=1, to=5000, width=5, textvariable=spinData2)
    spin2.place(x=350, y=220)


# create  run_button in gui
def create_text_box():
    text_box = Entry(form, width=20, textvariable=text1)
    text_box.place(x=30, y=60)

    text_box2 = Entry(form, width=20, textvariable=text2)
    text_box2.place(x=30, y=150)


# create  run_button in gui
def create_button():
    btn = Button(form, text="Run", command=run)
    btn.place(x=200, y=370)


# create labels in gui
def create_label():
    hidden_layer_label = Label(form, textvariable=label1)
    label1.set("Enter number of hidden layers")
    hidden_layer_label.place(x=20, y=30)

    num_of_neurons_label = Label(form, textvariable=label2)
    label2.set("Enter number of neurons'make space between each number'")
    num_of_neurons_label.place(x=20, y=120)

    lr_label = Label(form, textvariable=label3)
    label3.set("learning rate")
    lr_label.place(x=20, y=220)

    epoch_label = Label(form, textvariable=label5)
    label5.set("epochs number")
    epoch_label.place(x=250, y=220)


def data_preprocessing():
    global train_data, test_data, train_labels, test_labels

    dataSet = pd.read_csv('penguins.csv')

    # find important columns name which contain  numeric values
    numbers_cols = dataSet.select_dtypes(include=np.number).columns.to_list()

    # find important columns name which contain nun numeric values & convert it's type to string
    non_integer_cols = dataSet.select_dtypes(include=['object']).columns.to_list()
    dataSet[non_integer_cols] = dataSet[non_integer_cols].astype('string')

    # split dataSet based on specie
    adelie = dataSet.iloc[0:50, :]
    gentoo = dataSet.iloc[50: 100, :]
    chinstrap = dataSet.iloc[100: 150, :]

    nan_val_in_Adelie = {}
    nan_val_in_Gentoo = {}
    nan_val_in_Chinstrap = {}

    # find values for 'nan' with median in integer cols & with most repeated value in 'gender' col.
    # for integer col
    for col in numbers_cols:
        nan_val_in_Adelie[col] = adelie[col].median()
        nan_val_in_Gentoo[col] = gentoo[col].median()
        nan_val_in_Chinstrap[col] = chinstrap[col].median()

    # for gender
    nan_val_in_Adelie['gender'] = adelie['gender'].mode()[0]
    nan_val_in_Gentoo['gender'] = gentoo['gender'].mode()[0]
    nan_val_in_Chinstrap['gender'] = chinstrap['gender'].mode()[0]

    # replace nan
    # in adelie
    adelie = adelie.fillna(value=nan_val_in_Adelie)
    # in gentoo
    gentoo = gentoo.fillna(value=nan_val_in_Gentoo)
    # in Chinstrap
    chinstrap = chinstrap.fillna(value=nan_val_in_Chinstrap)

    # Encoding gender column
    genders = ['male', 'female']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(genders)
    adelie[adelie.columns[4]] = label_encoder.transform(adelie['gender'])
    gentoo[gentoo.columns[4]] = label_encoder.transform(gentoo['gender'])
    chinstrap[chinstrap.columns[4]] = label_encoder.transform(chinstrap['gender'])

    # dataSet shuffling
    adelie = adelie.sample(frac=1).reset_index(drop=True)
    gentoo = gentoo.sample(frac=1).reset_index(drop=True)
    chinstrap = chinstrap.sample(frac=1).reset_index(drop=True)

    # split dataSet into train dataSet and test dataSet
    Adelie_train = adelie.iloc[:30, :]
    Adelie_test = adelie.iloc[30:, :].reset_index(drop=True)
    Gentoo_train = gentoo.iloc[:30, :]
    Gentoo_test = gentoo.iloc[30:, :].reset_index(drop=True)
    Chinstrap_train = chinstrap.iloc[:30, :]
    Chinstrap_test = chinstrap.iloc[30:, :].reset_index(drop=True)

    # concatenate the data set
    train_frames = [Adelie_train, Gentoo_train, Chinstrap_train]
    test_frames = [Adelie_test, Gentoo_test, Chinstrap_test]

    train_data = pd.concat(train_frames).reset_index(drop=True)
    test_data = pd.concat(test_frames).reset_index(drop=True)

    # data shuffling
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # create labels list after encode species column
    train_labels_list = []
    test_labels_list = []

    for idx in range(len(train_data)):
        specie = train_data.iloc[idx, 0]
        train_labels_list.append(encodes[specie])

    for idx in range(len(test_data)):
        specie = test_data.iloc[idx, 0]
        test_labels_list.append(encodes[specie])

    # create train_labels & test_labels dataframes
    train_labels = pd.DataFrame(data={'species': train_labels_list})
    test_labels = pd.DataFrame(data={'species': test_labels_list})

    # remove labels from X data
    train_data.pop('species')
    test_data.pop('species')

    # normalize training data
    train_data = preprocessing.normalize(train_data)


def initialize_Model_Dfs():
    user_inputs()
    global weights, layers_output,delta
    # weight & bias
    if use_bias:
        for layerNum in range(hidden_num + 1):
            if layerNum == 0:
                weights.append(np.random.rand(neurons[layerNum], 6))
            elif layerNum == hidden_num:
                weights.append(np.random.rand(3, (neurons[layerNum-1] + 1)))
            else:
                weights.append(np.random.rand(neurons[layerNum],
                                              (neurons[layerNum - 1] + 1)))

    else:
        for layerNum in range(hidden_num + 1):
            if layerNum == 0:
                weights.append(np.random.rand(neurons[layerNum], 5))
            elif layerNum == hidden_num:
                weights.append(np.random.rand(3, neurons[layerNum - 1]))
            else:
                weights.append(np.random.rand(neurons[layerNum], neurons[layerNum - 1]))

    delta = [[] for layerNum in range(hidden_num + 1)]
    layers_output = [[] for layerNum in range(hidden_num + 1)]
    print(delta)


# main
data_preprocessing()
gui()
initialize_Model_Dfs()
