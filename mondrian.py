from scipy.io import loadmat
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.patches as pch

COLORLIST=['','white','r','gold','b','black']

# The neo-plastic paintings of Piet Mondrian (1872–1944) are simple but subtle abstract compositions. 
# Each composition consists solely of horizontal and vertical black lines and adjacent rectangles of 
# uniform red, yellow, blue and black on a white background. Such a series, with no complex colors 
# like green or brown, no curves, no brightness changes, no explicit depth rendering, may be suitable 
# for analyzing with classical machine learning methods of its composition and trying to create new painting 
# of Mondrian style.

# Albert LI

annots = loadmat('C:\Study-cityu\y1-semA\machine learning\Mondrian sklearn\MondrianData\MondriansAndTransatlantics.mat')
fig, ax = plt.subplots()

kmeans=KMeans(n_clusters=3, random_state=9, n_init=10)
vRegressor=KNeighborsRegressor(n_neighbors=10,weights='distance')
hRegressor=KNeighborsRegressor(n_neighbors=10,weights='distance')
colorClassifier=KNeighborsClassifier(n_neighbors=3,weights='distance')

# 'labels', 'names', 'reps'; 45 Images
# 0-height, 1-width, 2-vertical points, 3-vext, 4-vthick, 5-horizontal points, 6-hext, 7-hthick, 8-rect, 9-colors

# 1. Interpretation of dataset, visualization of images
def getData(index):
    painting=annots['reps'][0][index]
    print(annots['names'][0][index][0])
    height=painting[0][0][0]
    width=painting[1][0][0]
    vpoints=painting[2][0]
    vext=painting[3]
    vthick=[]
    for i in range(len(painting[4])):
        vthick.append(painting[4][i][0])
    vthick=np.array(vthick)
    hpoints=painting[5][0]
    hext=painting[6]
    hthick=[]
    for i in range(len(painting[7])):
        hthick.append(painting[7][i][0])
    hthick=np.array(hthick)
    rectangles=painting[8]
    for i in range(len(rectangles)):
        for j in range(4):
            rectangles[i][j]=int(rectangles[i][j])-1
    colors=painting[9][0]
    return height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors

def draw(height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors):
    plt.axis("equal")
    plt.xlim(0,width)
    plt.ylim(0,height)
    plt.gca().invert_yaxis()
    plt.axis('off')

    for i in range(len(colors)):
        rdata=rectangles[i]
        x=vpoints[rdata[0]]+vthick[rdata[0]]
        w=vpoints[rdata[1]]-x
        y=hpoints[rdata[2]]+hthick[rdata[2]]
        h=hpoints[rdata[3]]-y
        c=COLORLIST[colors[i]]
        rect=pch.Rectangle(xy=(x,y),width=w,height=h,color=c)
        ax.add_patch(rect)
    
    for i in range(1,len(vext)-1):
        ldata=vext[i]
        for j in range(2):
            ldata[j]=int(ldata[j])-1
        x=vpoints[i]
        w=vthick[i]
        y=hpoints[ldata[0]]
        h=hpoints[ldata[1]]-y
        rect=pch.Rectangle(xy=(x,y),width=w,height=h,color='black')
        ax.add_patch(rect)
    
    for i in range(1,len(hext)-1):
        ldata=hext[i]
        if ldata[0]==ldata[1]:
            continue
        for j in range(2):
            ldata[j]=int(ldata[j])-1
        x=vpoints[ldata[0]]
        w=vpoints[ldata[1]]-x
        y=hpoints[i]
        h=hthick[i]
        rect=pch.Rectangle(xy=(x,y),width=w,height=h,color='black')
        ax.add_patch(rect)


# 2. Attempt to use Kmeans clustering to find the focus of paintings
def kmeansDraw(height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors):
    rectpoints=[]
    rectdata=[]

    for i in range(len(colors)):
        rdata=rectangles[i]
        x=float((vpoints[rdata[0]]+vthick[rdata[0]]+vpoints[rdata[1]])/2)
        y=float((hpoints[rdata[2]]+hthick[rdata[2]]+hpoints[rdata[3]])/2)
        rectpoints.append([x,y])
        x=x*512/height
        y=y*512/height
        r,g,b=255,255,255
        if colors[i]==2:
            r,g,b=255,0,0
        elif colors[i]==3:
            r,g,b=255,215,0
        elif colors[i]==4:
            r,g,b=0,0,255
        elif colors[i]==5:
            r,g,b=0,0,0
        rectdata.append([x,y,r,g,b])
    rectdata=np.array(rectdata)

    kmeans.fit(rectdata)
    labels=kmeans.labels_
    dots=kmeans.cluster_centers_

    for i in range(len(colors)):
        plotx=rectpoints[i][0]
        ploty=rectpoints[i][1]
        plt.annotate(str(labels[i]),xy=(plotx,ploty),color='green')

    for dot in dots:
        plotx=float(dot[0]*height/512)
        ploty=float(dot[1]*height/512)
        cir=pch.Circle(xy=(plotx,ploty),radius=10,color='greenyellow')
        ax.add_patch(cir)

# Kmeans test, with samples 40, 23, 0

# imgindex=40
# height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors=getData(imgindex)
# draw(height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors)
# kmeansDraw(height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors)
# plt.show()


# 3 & 4. Using K-neighbors regression and classification to generate a new painting
def train():
    vx,vy,hx,hy,nbx,nby=[],[],[],[],[],[]

    for p in range(45):
        height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors=getData(p)

        # Samples for K Neighbors Regression
        ratio=np.float32(width/height)
        vpointsf=vpoints.astype(np.float32)/width
        hpointsf=hpoints.astype(np.float32)/height
        vthickf=vthick.astype(np.float32)/width
        hthickf=hthick.astype(np.float32)/height
        vcount=len(vpointsf)-2
        hcount=len(hpointsf)-2

        for i in range(1,len(vext)-1):
            vx.append([ratio,vcount,hcount,i])
            line=[]
            line.append(vpointsf[i])
            line.append(vthickf[i])
            line.append(hpointsf[vext[i][0]-1])
            line.append(hpointsf[vext[i][1]-1])
            vy.append(line)

        for i in range(1,len(hext)-1):
            hx.append([ratio,vcount,hcount,i])
            line=[]
            line.append(hpointsf[i])
            line.append(hthickf[i])
            line.append(vpointsf[hext[i][0]-1])
            line.append(vpointsf[hext[i][1]-1])
            hy.append(line)
        
        # Samples for K Neighbors Classification
        for rect in rectangles:
            w=np.float32(vpointsf[rect[1]]-vpointsf[rect[0]]-vthickf[rect[0]])
            h=np.float32(hpointsf[rect[3]]-hpointsf[rect[2]]-hthickf[rect[2]])
            x=np.float32(vpointsf[rect[1]]+vpointsf[rect[0]]+vthickf[rect[0]])/2
            y=np.float32(hpointsf[rect[3]]+hpointsf[rect[2]]+hthickf[rect[2]])/2
            nbx.append([ratio,vcount,hcount,w,h,x,y])
        nby=nby+list(colors)
    
    vx=np.array(vx,dtype=np.float32)
    vy=np.array(vy,dtype=np.float32)
    hx=np.array(hx,dtype=np.float32)
    hy=np.array(hy,dtype=np.float32)
    vRegressor.fit(vx,vy)
    hRegressor.fit(hx,hy)
    print("Vertical Lines: "+str(len(hx))) # 287
    print("Horizontal Lines: "+str(len(hy))) # 287

    nbx=np.array(nbx,dtype=np.float32)
    nby=np.array(nby,dtype=np.float32)
    colorClassifier.fit(nbx, nby)
    print("Color Blocks: "+str(len(nby))) # 1013



def findnearest(array,value):
    i=np.searchsorted(array,value,side='left')
    if i==0:
        return 0
    if i==len(array) or abs(value-array[i-1])<abs(value-array[i]):
        return i-1
    return i

def dotconnection(vext,hext,x,y):
    up=vext[x-1][0]<y and vext[x-1][1]>=y
    down=vext[x-1][0]<=y and vext[x-1][1]>y
    left=hext[y-1][0]<x and hext[y-1][1]>=x
    right=hext[y-1][0]<=x and hext[y-1][1]>x
    return up, down, left, right

def desingular(vext,hext,x,y):
    up, down, left, right=dotconnection(vext,hext,x,y)
    count=int(up)+int(down)+int(left)+int(right)
    if count==0 or count>2 or (up and down) or (left and right):
        return False
    if up:
        vext[x-1][1]-=1
    if down:
        vext[x-1][0]+=1
    if left:
        hext[y-1][1]-=1
    if right:
        hext[y-1][0]+=1
    return True


def predict(height,width,vcount,hcount):
    ratio=np.float32(width/height)
    longest=[0,0,1]
    vx,hx,nbx=[],[],[]
    
    for i in range(vcount):
        vx.append([ratio,vcount,hcount,i])
    vx=np.array(vx,dtype=np.float32)
    vy=vRegressor.predict(vx)
    vy=vy.T
    vpoints=[1]+list(vy[0]*width)+[width]
    vpoints.sort()
    vthick=[0]+list(vy[1]*width)+[0]

    for i in range(hcount):
        hx.append([ratio,vcount,hcount,i])
    hx=np.array(hx,dtype=np.float32)
    hy=hRegressor.predict(hx)
    hy=hy.T
    hpoints=[1]+list(hy[0]*height)+[height]
    hpoints.sort()
    hthick=[0]+list(hy[1]*height)+[0]

    vext=[[1,hcount+2]]
    vext1=vy[2]*height
    vext2=vy[3]*height
    for i in range(vcount):
        vext.append([])
        vext[-1].append(findnearest(hpoints,vext1[i])+1)
        vext[-1].append(findnearest(hpoints,vext2[i])+1)
        h=hpoints[vext[-1][1]-1]-hpoints[vext[-1][0]-1]
        if h>longest[0] or (h==longest[0] and random.random()<0.5):
            longest=[h,0,i+1]
    vext.append([1,hcount+2])
    
    hext=[[1,vcount+2]]
    hext1=hy[2]*width
    hext2=hy[3]*width
    for i in range(hcount):
        hext.append([])
        hext[-1].append(findnearest(vpoints,hext1[i])+1)
        hext[-1].append(findnearest(vpoints,hext2[i])+1)
        v=vpoints[hext[-1][1]-1]-vpoints[hext[-1][0]-1]
        if v>longest[0] or (v==longest[0] and random.random()<0.5):
            longest=[v,1,i+1]
    hext.append([1,vcount+2])

    # Ensure at least one through line
    if longest[1]==0:
        vext[longest[2]]=[1,hcount+2]
    else:
        hext[longest[2]]=[1,vcount+2]
    
    # Apply de-singularization
    for k in range(10):
        singulars=0
        for i in range(2,len(vext)):
            for j in range(2,len(hext)):
                if desingular(vext,hext,i,j):
                    singulars+=1
        if singulars==0:
            break
    
    # Generate rectangle list
    rectangles=[]
    for i in range(1,len(vext)):
        for j in range(1,len(hext)):
            up, down, left, right=dotconnection(vext,hext,i,j)
            if not (down and right):
                continue
            rect=[i-1,i,j-1,j]
            for k in range(i+1,len(vext)):
                _up, _down, _left, _right=dotconnection(vext,hext,k,j)
                if _down:
                    break
                rect[1]+=1
            for l in range(j+1,len(hext)):
                _up, _down, _left, _right=dotconnection(vext,hext,i,l)
                if(_right):
                    break
                rect[3]+=1
            rectangles.append(rect)
            w=np.float32(vpoints[rect[1]]-vpoints[rect[0]]-vthick[rect[0]])/width
            h=np.float32(hpoints[rect[3]]-hpoints[rect[2]]-hthick[rect[2]])/height
            x=np.float32(vpoints[rect[1]]+vpoints[rect[0]]+vthick[rect[0]])/width/2
            y=np.float32(hpoints[rect[3]]+hpoints[rect[2]]+hthick[rect[2]])/height/2
            nbx.append([ratio,vcount,hcount,w,h,x,y])
    nbx=np.array(nbx,dtype=np.float32)
    nby=colorClassifier.predict(nbx).astype(int)
    colors=list(nby)

    return height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors


# Image Generation Test
# 500,500,1,3
# 800,900,2,9
# 500,600,3,5
# 720,1280,7,7
# 800,450,2,4

train()
height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors = predict(600,400,5,4)
draw(height, width, vpoints, vext, vthick, hpoints, hext, hthick, rectangles, colors)
plt.show()
