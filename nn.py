import tensorflow as tf 
import numpy as np
import os.path


class mlp(object):

    def parmsFun(self): 
        params_w = { 
                     'h1' : tf.Variable (tf.random_normal ([inLyrSize    , hidLyrSize[0]])),
                     'h2' : tf.Variable (tf.random_normal ([hidLyrSize[0], hidLyrSize[1]])),
                     'out': tf.Variable (tf.random_normal ([hidLyrSize[1], outLyrSize]   ))  
                   }
        
        params_b = { 
                     'b1' : tf.Variable (tf.random_normal([hidLyrSize[0]])),
                     'b2' : tf.Variable (tf.random_normal([hidLyrSize[1]])),
                     'out': tf.Variable (tf.random_normal([outLyrSize]   ))  
                   }

        return params_w,params_b
        
    def scoreFun(self):
        
        in_to_hid1 = tf.add(tf.matmul(self.x_, self.params_w_['h1']), self.params_b_['b1'])
        in_to_hid1 = tf.nn.relu(in_to_hid1)
    
        hid1_to_hid2 = tf.add(tf.matmul(in_to_hid1, self.params_w_['h2']), self.params_b_['b2'])
        hid1_to_hid2 = tf.nn.relu(hid1_to_hid2)
    
        outLyr = tf.matmul(hid1_to_hid2, self.params_w_['out']) + self.params_b_['out']

        return outLyr
        
    def costFun(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.score_, self.y_)) 
        return cost
        
    def updateFun(self):
        update = tf.train.AdamOptimizer(learning_rate = self.lr_).minimize(self.cost_) 
        return update   
    
    def perfFun(self):
        correct_pred = tf.equal(tf.argmax(self.score_,1), tf.argmax(tf_y,1))
        return(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))    
        
    def __init__(self,tf_x,tf_y, inLyrSize, hidLyrSize, outLyrSize,lr):
        
        self.inLyrSize  = inLyrSize
        self.hidLyrSize = hidLyrSize
        self.outLyrSize = outLyrSize

        self.x_  = tf_x
        self.y_  = tf_y
        self.lr_ = lr
        
        [self.params_w_, self.params_b_] = mlp.parmsFun(self)
        self.score_                      = mlp.scoreFun(self)    
        self.cost_                       = mlp.costFun(self)
        self.update_                     = mlp.updateFun(self)
        self.perf_                       = mlp.perfFun(self)


########################################################## 
 

lr         = 0.01
trEpochs   = 1
batch_size = 100

hidLyrSize = [256, 256]
inLyrSize  = 104
outLyrSize = 2

tf_x = tf.placeholder("float", [None, inLyrSize])
tf_y = tf.placeholder("float", [None, outLyrSize])

mlp_class = mlp(tf_x,tf_y, inLyrSize, hidLyrSize, outLyrSize,lr)
initVar = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initVar) 
    c =[]
    print("##########################################################")   
    for epoch_i in range(trEpochs):
    
        avgOfCost = 0.
        index = 0
        indexval = 0
        
        # -----read data, convert to float------
        with open('out.csv') as f:
            input = f.readlines() 
        input = [x.strip() for x in input] 
        for i in range(len(input)):
            input[i] = [x.strip() for x in input[i].split(',')] 
        for i in range(len(input)):
            input[i] = map(float, input[i])
        
        trainData = []
        testData = []
        
        for i in range(11000):
            trainData.append(input[i])
        for i in range(11000,len(input)):
            testData.append(input[i])   
            
        # --------------------------------------
        
        for batch_i in range(2000):  # 5000  
            trData_i, trLabel_temp, batch = [], [], []
            
            for i in range(batch_size):
                batch.append(trainData[i+index])
                
            index += batch_size
            if index > ( len(trainData) - batch_size + 1 ):
                index = 0
                
            for item in batch:     
                trData_i .append( item[:104] )
                trLabel_temp.append( item[105] )
                
            trLabel_i = np.zeros((100,2))
            for i in range(batch_size): 
                if trLabel_temp[i] == 0:
                    trLabel_i[i][0] = 1
                else:
                    trLabel_i[i][1] = 1
                    
            trData_i  = np.reshape( trData_i , ( -1, 104 ) )
            trLabel_i = np.reshape( trLabel_i, ( -1, 2   ) )
            
            _,cost_i = sess.run([mlp_class.update_, mlp_class.cost_], feed_dict = {tf_x:trData_i, tf_y:trLabel_i})
            
# ---------------------------------------------------------------------------------------------------------------------

            if batch_i % 10 == 0: 
                training_accuracy = sess.run(mlp_class.perf_, feed_dict={tf_x: trData_i,    tf_y: trLabel_i })
                trData_ival, trLabel_tempval, batchval = [], [], []
                
                for i in range(batch_size):
                    batchval.append(testData[i+indexval])
                    
                indexval += batch_size
                if indexval > ( len(testData) - batch_size + 1 ):
                    indexval = 0
                    
                for item in batchval:     
                    trData_ival .append( item[:104] )
                    trLabel_tempval.append( item[105] )
                    
                trLabel_ival = np.zeros((100,2))
                for i in range(batch_size): 
                    if trLabel_tempval[i] == 0:
                        trLabel_ival[i][0] = 1
                    else:
                        trLabel_ival[i][1] = 1
                        
                trData_ival  = np.reshape( trData_ival , ( -1, 104 ) )
                trLabel_ival = np.reshape( trLabel_ival, ( -1, 2   ) ) 
                vald_accuracy = sess.run(mlp_class.perf_, feed_dict={tf_x: trData_ival,    tf_y: trLabel_ival }) 
                print str(batch_i) + " , trAccu: " + str(training_accuracy) + " , valdAccu: " + str(vald_accuracy)

        # -----------------------------------------------
        
        if os.path.exists('test.csv'): 
        
            predicted = open("predicted.txt" , 'w')
            
            with open('test.csv') as f:
                input = f.readlines() 
            input = [x.strip() for x in input] 
            for i in range(len(input)):
                input[i] = [x.strip() for x in input[i].split(',')] 
            for i in range(len(input)):
                input[i] = map(float, input[i])
                
            trData_i,  batch = [], []      
            for item in input:     
                trData_i .append( item[:104] ) 
                
            trLabel_i = np.zeros((100,2)) 
            
            trData_i  = np.reshape( trData_i , ( -1, 104 ) )
            trLabel_i = np.reshape( trLabel_i, ( -1, 2   ) )
            
            score_i = sess.run(mlp_class.score_, feed_dict={tf_x: trData_i,  tf_y: trLabel_i })
            
            for i in range (score_i.shape[0]):
                if score_i[i][0]  >  score_i[i][1] : 
                    predicted.write( "0" + "\n")
                else:   
                    predicted.write( "1" + "\n")
                    
        # -----------------------------------------------
        
#########################################################################################################











