import pandas as pd
import numpy as np

w1=0
w2=0
b=0
lr=0.1
epochs=20

def predict(x1,x2):
    z=w1*x1+w2*x2+b
    if z>=0:
        return 1
    else:
        return 0
    


data = [ 
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [1,1,1]
]

# data = [
#     [0,0,0],
#     [0,1,1],
#     [1,0,1],
#     [1,1,1]
# ]



# data = [
#     [0,0,1],
#     [0,1,1],
#     [1,0,1],
#     [1,1,0]
# ]


# data = [
#     [0,0,1],
#     [0,1,0],
#     [1,0,0],
#     [1,1,0]
# ]


# data = [
#     [0,0,0],
#     [0,1,1],
#     [1,0,1],
#     [1,1,0]
# ]



df=pd.DataFrame(data, columns=["x1", "x2", "y"])

def train(df):
    global w1, w2, b

    for _ in range(epochs):
        for i in range(len(df)):
            x1=df.loc[i,"x1"]
            x2=df.loc[i,"x2"]
            y=df.loc[i,"y"]


            y_hat=predict(x1,x2)
            error=y-y_hat

            w1=w1+lr*error*x1
            w2=w2+lr*error*x1
            b=b+lr*error


train(df)


print("final results:", w1,w2)
print("final bias:",b)

print("\nPredictions:")
for i in range(len(df)):
    x1 = df.loc[i, "x1"]
    x2 = df.loc[i, "x2"]
    print(x1, x2, "->", predict(x1, x2), "(expected:", df.loc[i,'y'], ")")
