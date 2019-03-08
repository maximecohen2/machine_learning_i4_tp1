import random
#samples inside


class samples:
    def __init__(self,nb_samples):
        self.samples_in=[]
        self.samples_out=[]
        self.nb_samples=nb_samples
        self.samples=[]
        self.samples_temp=[]
        self.samples_list=[]
        self.samples_temp_list=[]
        
    def create_samples(self):

##        print "Nombre d'exemple cree  pour le carre",self.nb_samples

        while len(self.samples_in)<(self.nb_samples):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)
            a,b = 0,0
            if (x>0.5 and x <1):
                b=x
            if (x>-1 and x<-0.5):
                b=x
            if(y>0.5 and y <1) :
                a=y
            if (y>-1 and y<-0.5):
                a=y
                            
            if a != 0 and b!=0:
                self.samples_in.append([b,a])
                self.samples_temp_list.append([b,a])


        for i in range (self.nb_samples):
            tuple_in = ((self.samples_in[i][0],self.samples_in[i][1]),1)
            self.samples_in[i]= tuple_in
            self.samples_list.append(tuple_in[0][0])
            

##        print "#---------------------------------------------------------#"
            
        #samples outside
##        print "Points HORS du carre : "
             
        while len(self.samples_out)<(self.nb_samples):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)
            if (x>=1 and x <=1.5) or (x>=-1.5 and x<=-1):
                if (y>=1 and y <=1.5) or (y>=-1.5 and y<=-1):

                    self.samples_out.append([x,y])
                    self.samples_temp_list.append([x,y])
        
        for i in range (self.nb_samples):
            tuple_out = ((self.samples_out[i][0],self.samples_out[i][1]),-1)
            self.samples_out[i]= tuple_out       
            
        self.samples_temp=self.samples_in+self.samples_out
        for i in range(self.nb_samples):
            index=random.randint(0,len(self.samples_temp)-1)
            temp=self.samples_temp.pop(index)
            self.samples.append(temp)

        self.samples_list=self.samples_in+self.samples_out

        for i in range (len(self.samples_temp_list)):
            self.samples_list[i]=self.samples_temp_list[i]

##        print "#---------------------------------------------------------#"
exercice=samples(10)
exercice.create_samples()
#print 'Exercice dans le carre : '



