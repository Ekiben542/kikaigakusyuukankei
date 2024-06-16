import numpy as np
import random
import matplotlib.pyplot as plt

class Deepnetwork:
    def __init__(self,input_layer,hidden_layer,output_layer,learningrate,Huber):
        self.count_output=output_layer
        self.weight=[np.random.normal(0.0,pow(hidden_layer[0],-0.5),(hidden_layer[0],input_layer))]
        for i in range(len(hidden_layer)-1):
            self.weight.append(np.random.normal(0.0,pow(hidden_layer[i+1],-0.5),(hidden_layer[i+1],hidden_layer[i])))
        self.weight.append(np.random.normal(0.0,pow(self.count_output,-0.5),(self.count_output,hidden_layer[-1])))
        self.lr=learningrate
        self.count_weight=len(self.weight)-1
        self.Huber=Huber
    def caluculation(self,inputer):
      self.state=[inputer]
      for i in range(self.count_weight+1):
        self.save=np.dot(self.weight[i],self.state[i])
        self.state.append(np.where(self.save>0,self.save,0.01*self.save))

      return self.state[-1]

    def learning(self,error):
        self.error=[error]
        for i in range(self.count_weight):
          self.error.append(np.dot(self.weight[self.count_weight-i].T,self.error[i]))
        for i in range(self.count_weight+1):
          self.the_error=self.error[i]
          self.the_Huber=self.Huber[self.count_weight-i]
          self.weight[self.count_weight-i]+=self.lr*np.dot((np.where(np.abs(self.error[i])>self.the_Huber,-1.0,self.the_error)*np.where(np.dot(self.weight[self.count_weight-i],self.state[self.count_weight-i])>0,1.0,0.01)).reshape(-1,1),np.array(self.state[self.count_weight-i]).reshape(1,-1))

    def get_weight(self):
        return self.weight
    def change_weight(self,weight):
        self.weight=weight

class Qlearning:
    def __init__(self,lr,dr,gamma,c_state,c_action,Huber):
        self.lr=lr
        self.dr=dr
        self.gamma=gamma
        self.c_action=c_action
        self.q_network=Deepnetwork(c_state,(25,),c_action,lr,Huber)
        self.t_network=Deepnetwork(c_state,(25,),c_action,lr,Huber)
        self.memory=[]
    def act(self,state):
        self.state=state
        if self.gamma>np.random.uniform():
            self.action=np.random.randint(0,self.c_action)
        else:
          self.action=np.argmax(self.t_network.caluculation(state))
        return self.action

    def note(self,next_state,reward,is_no_next,data_amount_limit):
      self.reward=reward
      self.near_memory=[self.state,self.action,self.reward,next_state,is_no_next]
      if not self.near_memory in self.memory:
        self.memory.append(self.near_memory)
        self.c_memory=len(self.memory)
        if (not data_amount_limit==-1) and self.c_memory==data_amount_limit+1:
          del self.memory[0]

    def learn(self,batchsize,changer):
      if batchsize==-1:
        self.the_action=self.near_memory[1]
        self.error=np.repeat(0.0,self.c_action)
        self.error[self.the_action]=self.near_memory[2]+np.where(self.near_memory[4]==True,0.0,self.dr*max(self.t_network.caluculation(self.near_memory[3])))-self.q_network.caluculation(self.near_memory[0])[self.the_action]
        self.q_network.learning(self.error)
      else:
        if batchsize>self.c_memory:
          self.c_sample=self.c_memory
        else:
          self.c_sample=batchsize
        self.sample=random.sample(self.memory,self.c_sample)
        self.c_the_action=np.repeat(0.0,self.c_action)
        self.sum=[]
        for i in range(self.c_sample):
          self.the_action=self.sample[i][1]
          self.error=np.repeat(0.0,self.c_action)
          self.error[self.the_action]=self.sample[i][2]+np.where(self.sample[i][4]==True,0.0,self.dr*max(self.t_network.caluculation(self.sample[i][3])))-self.q_network.caluculation(self.sample[i][0])[self.the_action]
          self.sum.append(self.error)
        self.sum=sum(self.sum)
        self.sum/=np.repeat(batchsize,self.c_action)
        self.q_network.learning(self.sum)
      if changer==-1:
          self.t_network.change_weight(self.q_network.get_weight())
      else:
           self.q_weight=self.q_network.get_weight()
           self.t_weight=self.t_network.get_weight()
           for i in range(len(self.q_weight)):
              self.t_weight[i]=changer*(self.q_weight[i]-self.t_weight[i])
              self.t_network.change_weight(self.t_weight)
    def change_gamma(self,gamma):
      self.gamma=gamma
    def show_memory(self):
      return self.memory
    def show_t_network_weight(self):
      return self.t_network.get_weight()
    def change_t_network_weight(self,weight):
      return self.t_network.change_weight(weight)
    def show_memory(self):
        return self.memory
    def change_memory(self,data,is_replace):
        if is_replace:
            self.memory=data
        else:
            for i in data:
                self.memory.append(i)