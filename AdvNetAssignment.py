import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

#filters the way using an exponential moving average
def FilterWave(data, FilterWindow):
    CumSum = np.cumsum(data)
    CumSum[FilterWindow:] = CumSum[FilterWindow:] - CumSum[:-FilterWindow]
    return CumSum[FilterWindow - 1:] / FilterWindow


class Queue:
    def __init__(self):
        #things we track
        self.TotalArrivals = 0  # number of packets that have arrived
        self.ServicingTotal = 0  #Total Time server is sending packets
        self.ServerBusy = 0  #Boolean for if the server currently has a packet
        self.PacketsInQueue = 0  #Amount of packets in the system. Queue + server
        self.WaitTotal = 0.0  # total time spent by packets waiting to send
        self.TotalSent = 0  # Total number of packets successfully sent
        self.TotalDropped = 0  # Total number of packets dropped

        #RED Variables
        self.B = 10  # queue capacity
        self.W = 0.1 #Weight for Red Sim
        self.MinTH = 2.0 #minimum threshold for RED
        self.MaxP = .1 #max dropping prob when q is MaxTH
        self.MaxTH = 0.0 #Max threshold for RED
        self.Avg = float(0.0) #Average Queue size
        self.PastAvg = 0.0 #Average Queue Size on arrival of previous packet
        self.Pa = 0.0 #probability to drop an arriving packet

        #timing Variables
        self.NextArrival = self.GenArrival()  # time when the next packet will arrive. Uses Poisson
        self.NextDeparture = float('inf')  # Time when the next packet will depart. Uses Exponential. Inf at start
        self.CurrTime = 0.0  # Current number of time advances

    # generate a rv using poisson methods with lambda = .95
    def GenArrival(self):
        New = np.random.poisson(.95)
        return (New)

        # generate a rv using exponential methods with mu = 1

    def GenService(self):
        New = np.random.exponential(scale=(1/1))
        return (New)

        # find PA for RED



    def FindPA(self):
        return ((self.MaxP * ((self.Avg - self.MinTH) / (self.MaxTH - self.MinTH))))

        # return AVG for RED

    def FindAvg(self):
        return (((1 - self.W) * self.PastAvg) + (self.W * self.PacketsInQueue))

        # for quadratic line of best fit

    def Quadratic(self, x, a, b, c):
        return a * x ** 2 + b * x + c

    # increase the time step
    def FindNextTime(self):
        #Find when the next event happens
        NextEvent = min(self.NextArrival, self.NextDeparture)

        #increase the wait time
        self.WaitTotal += (self.PacketsInQueue * (NextEvent - self.CurrTime))

        #advance time to next event
        self.CurrTime = NextEvent

        #if the next event is an arrival or a departure, do those
        if self.NextArrival < self.NextDeparture:
            self.NewArrival()
        else:
            self.SendPacket()

    #A new arrival event occurs
    def NewArrival(self):
        # track arrivals
        self.TotalArrivals += 1

        # perform RED avg
        self.PastAvg = self.Avg
        self.Avg = self.FindAvg()

        # Use probability to decide to add or dop packet
        # if very empty q, add packet
        if float(self.Avg) <= float(self.MinTH):
            self.AddPacket()
        elif (self.Avg > self.MinTH) & (self.Avg < float(self.MaxTH)):  # if relatively full q, find a prob to add
            self.PA = self.FindPA()  # generate probability

            # generate a uniform rv, if greater than, we drop the packet. This method for deciding when to drop
            # was found here: https://www.geeksforgeeks.org/random-early-detection-red-queue-discipline/
            if self.Pa > np.random.uniform(0.0, 1.0):
                self.DropPacket()
            else:
                self.AddPacket()
        elif self.Avg >= self.MaxTH:  # if full q, just drop
            self.DropPacket()

    #drops an arriving packet
    def DropPacket(self):
        #find next arrival and increase amount dropped
        self.TotalDropped += 1
        self.NextArrival = self.CurrTime + self.GenArrival()

    #add a packet to queue or server
    def AddPacket(self):
        self.NextArrival = self.CurrTime + self.GenArrival() #find next arrival
        if self.PacketsInQueue == 0: #if empty queue
            if self.ServerBusy == 1:  #put the packet in the q
                self.PacketsInQueue += 1
            elif self.ServerBusy == 0:  #Server is not busy
                self.dep = self.GenService() #find service time for new packet
                self.ServicingTotal += self.dep  # track how long server is sending packets
                self.NextDeparture = self.CurrTime + self.dep
                self.PacketsInQueue += 1 #notify that  server busy and add a packet to system
                self.ServerBusy = 1
        elif (self.PacketsInQueue < self.B) & (self.PacketsInQueue != 0):  #if q is neither empty nor full
            self.PacketsInQueue += 1 #place in queue

#sends a packet to next node
    def SendPacket(self):
        self.TotalSent += 1 #increase the amount of packets sucesfully sent
        if self.PacketsInQueue > 0: #if packets in q
            self.PacketsInQueue -= 1 # decrease amount in q
            self.dep = self.GenService() #find next departure time
            self.ServicingTotal += self.dep
            self.NextDeparture = self.CurrTime + self.dep
        else:
            self.NextDeparture = float('inf') #if queue and server empty, no more departures
            self.ServerBusy = 0

    def Question1(self):
        #this is the amount of time intervals per MaxTH value
        size = 500000

        #create empty arrays to hold values
        Thruput = []
        ArrivalRate = []
        LossRate = []
        MaxTH = [2,3,4,5,6,7,8,9,10]
        SystemTime = []
        QueueingDelay = []

        for x in range(300000):
            self.FindNextTime()

        for j in range(9):
            #increase maxth every loop starting at 1 and ending at 10
            self.MaxTH = MaxTH[j]

            #50,000 packets
            for i in range(size):
                #advance time continuously
                self.FindNextTime()

                #check for no divide by zeroes errors
                if self.CurrTime != 0:
                    #record current average arrival rate
                    ArrivalRate.append(float(self.TotalArrivals/self.CurrTime))
                if self.TotalSent != 0:
                    #record average current loss rate, queue delay, and system time
                    LossRate.append(float(self.TotalDropped/self.TotalArrivals))
                    QueueingDelay.append((self.WaitTotal / self.TotalSent))
                    SystemTime.append((self.WaitTotal + self.ServicingTotal) / self.TotalSent)
                if self.TotalSent != 0 and self.TotalArrivals != 0:
                    #record current thruput
                    Thruput.append((self.TotalSent / self.CurrTime))

        #start overall figure
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.title("Buffer Size = 10 (Maxp = .1 and MinTH = 2", fontsize = 18)

        MaxTH = np.linspace(2, 10, len(Thruput))

        # dictionary of lists
        # dict = {'MaxTH': MaxTH, 'Thru': Thruput}
        #
        # df = pd.DataFrame(dict)
        #
        # # saving the dataframe
        # df.to_csv('GFG.csv')

        #add to subplot so it is all in one page
        plt.subplot(2,2,1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.plot(MaxTH, Thruput, label='Data points', color='blue') #raw data
        #MaxTH.tofile('MaxTH.csv', sep=',')
        #Thruput.tofile('thruput.csv',sep=',')
        FilterWindow = 350  #plot the filtered wave after getting rid of garbage values
        Smoothedthruput = FilterWave(Thruput, FilterWindow)

        # Plot the smoothed data for thruput
        #plt.plot(MaxTH[FilterWindow - 1:], Smoothedthruput, label='Smoothed Data', color='green')
        plt.xlabel('Max Threshold (Packets)', fontsize=18)
        plt.ylabel('Throughput (Packets/Second)', fontsize=18, labelpad = 5)
        #plt.legend()

        MaxTH = np.linspace(2, 10, len(LossRate))

        #add to figure
        plt.subplot(2, 2, 2)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.plot(MaxTH, LossRate, label='Data points', color='blue') #raw data
        FilterWindow = 150  #plot the filtered wave after getting rid of garbage values
        SmoothedLossRate = FilterWave(LossRate, FilterWindow)

        # Plot the smoothed data for loss rate
        #plt.plot(MaxTH[FilterWindow - 1:], SmoothedLossRate, label='Smoothed Data', color='green')
        plt.xlabel('Max Threshold (Packets)', fontsize=18)
        plt.ylabel('Average LossRate', fontsize=18, labelpad = 5)
        ##plt.legend()

        MaxTH = np.linspace(2, 10, len(QueueingDelay))

        #add to figure
        plt.subplot(2, 2, 3)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.plot(MaxTH, QueueingDelay, label='Data points', color='blue') #Raw Data
        FilterWindow = 150  #plot the filtered wave after getting rid of garbage values
        SmoothedQueueingDelay = FilterWave(QueueingDelay, FilterWindow)

        # Plot the smoothed data for q delay
        #plt.plot(MaxTH[FilterWindow - 1:], SmoothedQueueingDelay, label='Smoothed Data', color='green')
        plt.xlabel('Max Threshold (Packets)', fontsize=18)
        plt.ylabel('Queue Delay (Seconds)', fontsize=18, labelpad = 5)
        ##plt.legend()

        MaxTH = np.linspace(2, 10, len(SystemTime))

        #add to figure
        plt.subplot(2, 2, 4)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.plot(MaxTH, SystemTime, label='Data points', color='blue') #raw Data
        FilterWindow = 150  #plot the filtered wave after getting rid of garbage values
        SmoothedSystemTime = FilterWave(SystemTime, FilterWindow)

        # Plot the smoothed data for system time
        #plt.plot(MaxTH[FilterWindow - 1:], SmoothedSystemTime, label='Smoothed Data', color='green')
        plt.xlabel('Max Threshold (Packets)', fontsize=18)
        plt.ylabel('System Time (Seconds)', fontsize=18, labelpad = 5)
        ##plt.legend()


        #show figure
        plt.show()

    def Question2(self):
        # this is the amount of time intervals per MaxTH value
        size = 5000

        self.MaxTH = 7  # lock MaxTH

        # create empty arrays to hold values
        Thruput = []
        ArrivalRate = []
        LossRate = []
        MaxP = np.linspace(0, 1,  1000)  # holds all values between 2 and 10
        SystemTime = []
        QueueingDelay = []
        for x in range(300000):
            self.FindNextTime()
        for j in range(1000):
            # increase MaxP every loop starting at 1 and ending at 10
            self.MaxP = MaxP[j]

            # 50,000 packets
            for i in range(size):
                # advance time continuously
                self.FindNextTime()

                # check for no divide by zeroes errors
                if self.CurrTime != 0:
                    # record current arrival rate
                    ArrivalRate.append(float(self.TotalArrivals / self.CurrTime))
                if self.TotalSent != 0:
                    # record current loss rate, queue delay, and system time
                    LossRate.append(float(self.TotalDropped / self.TotalSent))
                    QueueingDelay.append((self.WaitTotal / self.TotalSent))
                    SystemTime.append((self.WaitTotal + self.ServicingTotal) / self.TotalSent)
                if self.TotalSent != 0 and self.TotalArrivals != 0:
                    # record current thruput
                    Thruput.append((self.TotalSent / self.CurrTime))

        # start overall figure
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.title("Buffer Size = 10 (MaxTH = 7 and MinTH = 2)", fontsize = 18)

        MaxP = np.linspace(0, 1,  len(Thruput))
        # add to subplot so it is all in one page
        plt.subplot(2, 2, 1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        #plot smoothed wave for thruput
        # FilterWindow = 200
        # Smoothedthruput = FilterWave(Thruput, FilterWindow)
        # plt.plot(MaxP[FilterWindow - 1:], Smoothedthruput, label='Smoothed Data', color='green')

        SmoothedThruput = savgol_filter(Thruput, window_length=1000, polyorder=2)
        plt.plot(MaxP, SmoothedThruput, label='Smoothed Data', color='red')


        #plt.plot(MaxP, Thruput, label='Data points', color='blue')  # raw data
        plt.xlabel('Max Probability (MaxP)', fontsize=18)
        plt.ylabel('Throughput (Packets/Second)', fontsize=16, labelpad = 5)
        #plt.legend()

        MaxP = np.linspace(0, 1,  len(LossRate))
        # add to figure
        plt.subplot(2, 2, 2)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        #plt.plot(MaxP, LossRate, label='Data points', color='blue')  # raw data

        # Plot the smoothed data for loss rate
        # SmoothedLossRate = FilterWave(LossRate, FilterWindow)
        # plt.plot(MaxP[FilterWindow - 1:], SmoothedLossRate, label='Smoothed Data', color='green')

        SmoothedLossRate = savgol_filter(LossRate, window_length=1000, polyorder=2)
        plt.plot(MaxP, SmoothedLossRate, label='Smoothed Data', color='red')

        plt.xlabel('Max Probability (MaxP)', fontsize=18)
        plt.ylabel('Average LossRate', fontsize=18, labelpad = 5)
        #plt.legend()


        MaxP = np.linspace(0, 1,  len(QueueingDelay))
        # add to figure
        plt.subplot(2, 2, 3)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # Plot the smoothed data for queue delay
        # SmoothedQueueDelay = FilterWave(QueueingDelay, FilterWindow)
        # plt.plot(MaxP[FilterWindow - 1:], SmoothedQueueDelay, label='Smoothed Data', color='green')

        SmoothedQueueDelay = savgol_filter(QueueingDelay, window_length=1000, polyorder=2)
        plt.plot(MaxP, SmoothedQueueDelay, label='Smoothed Data', color='red')


        #plt.plot(MaxP, QueueingDelay, label='Data points', color='blue')  # Raw Data
        plt.xlabel('Max Probability (MaxP)', fontsize=18)
        plt.ylabel('Queue Delay (Seconds)', fontsize=18, labelpad = 5)
        #plt.legend()

        MaxP = np.linspace(0, 1,  len(SystemTime))
        # add to figure
        plt.subplot(2, 2, 4)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # Plot the smoothed data for system time
        # SmoothedSystemTime = FilterWave(SystemTime, FilterWindow)
        # plt.plot(MaxP[FilterWindow - 1:], SmoothedSystemTime, label='Smoothed Data', color='green')

        SmoothedSystemTime= savgol_filter(SystemTime, window_length=1000, polyorder=2)
        plt.plot(MaxP, SmoothedSystemTime, label='Smoothed Data', color='red')

        #plt.plot(MaxP, SystemTime, label='Data points', color='blue')  # raw Data
        plt.xlabel('Max Probability (MaxP)', fontsize=18)
        plt.ylabel('System Time (Seconds)', fontsize=18, labelpad = 5)
        #plt.legend()
        # show figure
        plt.tight_layout()
        plt.show()


#actual script starts here. Make queue objects, and answer questions
Queue1 = Queue()
Queue1.Question1()

Queue2 = Queue()
Queue2.Question2()