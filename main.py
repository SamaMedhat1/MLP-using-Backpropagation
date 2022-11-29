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


# main
gui()