from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import copy
from PIL import Image
import os
import random
import time
import math


imagesize = (120, 120)
peak = 100
gusti = ["margherita", "crudo", "funghi", "salame", "rucola", "4formaggi", "americana"]

def buildnet():
    inputs = len(gusti)
    outputs = imagesize[0] * imagesize[1] * 3 # R G B
    hiddens = (120 * 3)  # lol, I have no idea
    return buildNetwork(inputs, hiddens, outputs)

def getSwitchTuple(index, lengt, disturb=0):
    ret = []
    for i in range(lengt):
        if i == index:
            ret.append((1.0 + disturb) * peak)
        else:
            ret.append(0.0)
    return tuple(ret)

def buildtrainset():
    inputs = len(gusti)
    outputs = imagesize[0] * imagesize[1] * 3
    ds = SupervisedDataSet(inputs, outputs)
    for gusto in gusti:
        indice = gusti.index(gusto)
        pizzaset = os.listdir("./pizze/" + gusto + "/")
        print("Training set for gusto: %s (%s)" % (gusto, ",".join(map(str, getSwitchTuple(indice, inputs)))))
        for pizzaname in pizzaset:
            pizza = "./pizze/" + gusto + "/" + pizzaname
            print("   Training with %s" % pizza, end=" ")
            ds.addSample(getSwitchTuple(indice, inputs, disturb=random.uniform(-0.3, 0.3)), processImg(pizza))
            print("done")
    return ds

def outimage(outtuple, name):
    img = Image.new('RGB', imagesize, "white")
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            tup_index = (i*img.size[0] + j) * 3
            pixels[i,j] = (int(outtuple[tup_index]), int(outtuple[tup_index + 1]), int(outtuple[tup_index + 2]))
    img.save(name)
    #img.show()

def calcETA(timestep, remaining):
    totsec = timestep * remaining
    totmin = math.floor(totsec / 60)
    remsec = totsec - (totmin * 60)
    return totmin, remsec

def letsrock(rounds=25):
    minimum = 999999999999
    bestnet = None
    print("Initializing neural network...")
    net = buildnet()
    print("Building training set...")
    trset = buildtrainset()

    trainer = BackpropTrainer(net, trset)

    started = time.time()
    for i in range(rounds):
        print("training: %d%%... " % ((i*100) / rounds), end="")
        err = trainer.train()
        timestep = (time.time() - started) / (i+1)
        min, sec = calcETA(timestep, rounds - i - 1)
        if err < minimum:
            minimum = err
            bestnet = copy.deepcopy(net)
        print("error: %.05f  - ETA: %02d:%02d" % (err, min, sec), end="\r")
    #trainer.trainUntilConvergence(verbose=True)
    print("training: complete!                           ")
    return bestnet

def fullShow():
    net = letsrock()
    for gusto in gusti:
        print("Creating pizza, gusto: %s" % gusto)
        indice = gusti.index(gusto)
        activ = getSwitchTuple(indice, len(gusti))
        name = "oven/" + gusto + ".jpg"
        rgb = net.activate(activ)
        datum = list(rgb)
        outimage(datum, name)

def processImg(filename):
    img = Image.open(filename)
    img = img.resize(imagesize, Image.ANTIALIAS)
    rgb_img = img.convert('RGB')
    pixels = []
    for x in range(imagesize[0]):
        for y in range(imagesize[1]):
            tup = tuple(rgb_img.getpixel((x, y)))
            pixels.extend(tup)
    return tuple(pixels)


if __name__ == "__main__":
    fullShow()