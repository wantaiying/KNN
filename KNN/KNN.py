import numpy as np
#本次实验中距离选择欧式距离，（2次方那个）,k值取3，分类通过距离最近的那个来判断类别 
#第一步生成测试数据
def data_create():
    x_data=np.linspace(1,5,10)
    x_data=x_data[:,np.newaxis]

    noise=np.random.normal(0,0.5,x_data.shape)
    y_data=2*x_data+noise

    data=np.hstack((x_data,y_data))
    #此处label为直接确定好的，如果需要无监督判断，可以使用kmeans聚类,判断
    label=np.array([1.0,2.0,3,4,5,6,7,8,9,10])

    return data,label

data,label=data_create()
#print("测试数据为:")
#print(data,label)


#第二步，训练模型
def classfy(input_data,data,k):
    DataSetSize = data.shape[0]#判断行数即数据数
    diffmat = np.tile(input_data, (DataSetSize, 1)) - data  #计算每个点与要求点[0,0]的x，y差值，用于后面求距离

    SqDiffMat = diffmat**2                              #平方,内部每个数都会自动平方

    SqDistances = SqDiffMat.sum(axis=1)     #求距离
    Distance = SqDistances**0.5
    print("距离算出来了")
    print(Distance)
    temp=list(Distance)#这里比较熟悉数组的操作，所以转换回数组
    dic={}
    for num in range(k):#找最近的k个
        min=99999
        now_min=99
        for i in range(len(temp)):
            if temp[i]<min:
                min=temp[i]
                now_min=i
                #每次循环找到当前最小的
        dic[now_min]=temp[now_min]
        temp[now_min]=99999

    
    #根据分类策略进行判断,分类策略，如果三个中有多个属于一个类别，则分为这个类别，否则就和最近的一个类别
    Keys=[]#存类别
    category=[]#存各个类别的数量
    for i in range(k):
        category.append(0)
    for key in dic.keys(): 
        Keys.append(label[key])
    for i in range(k):#判断是否存在重复类别
        
        for x in range(k):
            if Keys[i]==Keys[x]:
                category[i]+=1
    max=1
    now_max=0
    #找到重复次数最多的类别，有多个类别重复次数相同，取最前面的，因为距离短
    for i in range(k):
        if category[i]>max:
            max=category[i]
            now_max=i
   
    #最终可以宣布类别了
    if max>1:
        print("这个点属于%s"% str(Keys[now_max]))
    else:
        print("这个点属于%s"% str(Keys[0]))




            
       
   

classfy([3,2],data,3)



