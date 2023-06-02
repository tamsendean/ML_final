import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d


def read_data():
    df=pd.read_csv("end_times.csv",sep=",",header=0)
    #header=0[default]:column names are inferred from the first line of the file
    x=np.array(df[['Day', 'Ride Time (secs)', 'Average Heart Rate (bpm)']])
    y=np.array(df['Average Speed (mph)'])
    y=np.reshape(y,(-1,1))
    return(x,y)

def normalise_data(x,y):
    m=len(x)
    X_norm=np.zeros((m,np.size(x,1)))
    mu=np.zeros((1,np.size(x,1))) #find the size of x along the coloumn(axis=1)
    sigma=np.zeros((1,np.size(x,1)))
    #print(np.shape(mu))
    mu[0,0]=np.mean(x[:,0])
    mu[0,1]=np.mean(x[:,1])
    mu[0,2]=np.mean(x[:,2])
    sigma[0,0]=np.std(x[:,0])
    sigma[0,1]=np.std(x[:,1])
    sigma[0,2]=np.std(x[:,2])
    X_norm[:,0]=(x[:,0]-mu[0,0])/sigma[0,0]
    X_norm[:,1]=(x[:,1]-mu[0,1])/sigma[0,1]
    X_norm[:,2]=(x[:,2]-mu[0,2])/sigma[0,2]
    y_norm=(y-np.mean(y))/np.std(y)
    print(format(np.mean(X_norm[:,0]),".6f"),format(np.mean(X_norm[:,1]),".6f"),format(np.mean(y_norm),".6f"))
    print(format(np.std(X_norm[:,0]),".6f"),format(np.std(X_norm[:,1]),".6f"),format(np.std(y_norm),".6f"))
    return(X_norm,y_norm)

def plot_data(x,y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,0], x[:,1], y, c=y, cmap='viridis')
    ax.set_xlabel('Ride Time (secs)')
    ax.set_ylabel('Average Heart Rate (bpm)')
    ax.set_zlabel('Average Speed (mph)')
    return ax

def cost_function(X,y,theta,m):
    J=0
    k=np.matmul(X,theta)
    k=np.reshape(k,(-1,1))
    #print(np.shape(k))
    sub=k-y
    #print(y)
    J=np.sum(np.square(sub))
    return J/(2*m)
    

def grad_descent(X,y,theta,m,n_iters):
    alpha=0.1
    for i in range(0,n_iters):
            dJ0=0
            dJ1=0
            dJ2=0
            dJ3=0
            k=np.matmul(X,theta)
            k=np.reshape(k,(-1,1))
            sub= k - y                  #basically we are doing (pred-y),in vectorised form (x*theta-y)
            dJ0=(np.sum(sub))/m

            x1=X[:,1]
            x1=np.reshape(x1,(-1,1))
            x1=np.transpose(x1)
            #print(np.shape(X[:,1]),np.shape(x1))
            dJ1=(x1 @ sub)/m

            x2=X[:,2]
            x2=np.reshape(x2,(-1,1))
            x2=np.transpose(x2)
            dJ2=(x2 @ sub)/m

            x3=X[:,3]
            x3=np.reshape(x3,(-1,1))
            x3=np.transpose(x3)
            dJ3=(x3 @ sub)/m

            theta[0]=theta[0]-alpha*dJ0
            theta[1]=theta[1]-alpha*dJ1
            theta[2]=theta[2]-alpha*dJ2
            theta[3]=theta[3]-alpha*dJ3

            J_new=cost_function(X,y,theta,m)
            #print("for theta:",format(theta[0,0],".6f"),format(theta[1,0],".6f"),format(theta[2,0],".6f"))
            #print("cost obtained:",J_new)
    return (J_new,theta)

def main():
    #Read data
    x,y=read_data()
    m=len(x)
    theta=np.zeros((4,1))
    #print(x,y)
    print("Input shape:{} Output shape: {}".format(np.shape(x),np.shape(y)))
    #Normalise data
    x_norm,y_norm=normalise_data(x,y)
    #Plot the data
    ax=plot_data(x,y)
    plt.show()
    X=np.concatenate((np.ones((m,1)),x_norm),axis=1)
    #Find cost of intial theta values
    J=cost_function(X,y_norm,theta,m)
    print("Cost Obtained:",J)
    #Apply Gradient Descent to fit the model to the data and find cost of obtained theta values
    J_new,theta=grad_descent(X,y_norm,theta,m,n_iters=2000)
    print("final cost: ", J_new)

    # Cartesian indexing
    x1_plot=np.linspace(-3,3,50)
    x2_plot=np.linspace(-3,3,50)
    x1_plot,x2_plot=np.meshgrid(x1_plot,x2_plot)
    y_plot=theta[0]+theta[1]*x1_plot +theta[2]*x2_plot   
    contours = plt.contour(x1_plot, x2_plot, y_plot, 30, colors='black')
    print('model parameters:',theta,sep = '\n')
    
    # prompts user for inputs
    print('======================')
    print('Predictions for Speed')
    day = int(input("Please enter the # of days after 8/07: "))
    ride_time = int(input("Please enter the # of seconds of the ride: "))
    hr = int(input("Please enter the heart rate: "))
    print("Predicted speed: ", (theta[0]+theta[1]*day+theta[2]*ride_time+theta[3]*hr)/10)

    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(y_plot, extent=[0, 3, 0, 3], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__=="__main__":
    main()