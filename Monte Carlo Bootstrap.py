# Monte Carlo Bootstrap resampling
# Pembagian perpotongan


import numpy as np
import math,random,sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.set_printoptions(threshold=sys.maxsize)
#Getting the Data From External File (.txt)
Production = np.loadtxt ("TM-05.txt",dtype=float,delimiter='\t',skiprows=1)
cek = np.loadtxt("Cek Data TM-05.txt",dtype=float,delimiter='\t',skiprows=1)

# print (Production)
# print ('')

panjang = len(Production)
panjang_cek = len(cek)

print('Panjang data sebanyak : ',panjang)
print ('')
#Data Input N

base = int(input('Input size of resampling : '))
data_sim = int(input('Berapa lama data yang akan dipakai ? (Month)'))
banyak_simulasi = int(input('Banyaknya simulasi : '))
forecast = int(input ('Jumlah Timestep untuk forecast : '))

#Function for hyperbolic decline
def Hyperbolic (t , b , d, rate ):
    return rate/((1+d*b*t)**(1/b))
def Gompertz(t,b,d,rate):

    return float(rate*d*b*math.exp(-d*t)*math.exp(-b*math.exp(-d*t)))
f2 = np.vectorize(Gompertz)


#Size of data resample
array_size =np.shape(Production)
a = array_size[0]
coef = data_sim/base
size = base * math.floor(coef)
beda = panjang - size

#Experimental data points (Rate - Time)
yData = Production[:panjang]
yData_calc = Production[panjang-data_sim:panjang]
xData = np.arange (panjang, dtype = int)
xData_calc = np.arange(data_sim)
xData_calc_plot = np.arange (panjang-data_sim,panjang)
xData_cek = np.arange(panjang+1,panjang+panjang_cek+1)
print(yData_calc)
#Forecast data
xForecast = np.arange(panjang,panjang+forecast)

#Curve fitting
plt.plot(xData, yData, 'bo', label = 'experimental data')
fit = curve_fit(f2, xData_calc, yData_calc,p0=[1,1,1],bounds=((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf)))

popt, pcov = fit
plt.plot (xData_calc_plot, f2(xData_calc, *popt ), 'g--',label ='Data fit' )
# plt.show()

print(f2(xData_calc, *popt ))
#Residual
residual = abs(yData_calc-f2(xData_calc, *popt ))
print('Residual :')
print(residual)


#Bootsraping (Creating Resampling Data)
resample = np.empty([0,0])
result_parameter = np.empty([0,0])
print(np.shape(result_parameter))
awal = f2(xData_calc, *popt )
print(awal)
print(data_sim-base)
calc_length = len(yData_calc)

for i in range (banyak_simulasi):
    for j in range (math.floor(coef)):
        random_number = random.randint(1,data_sim-base)
        tambahan = residual[random_number:random_number+base]
        resample = np.append(resample,tambahan)

    sample_data = awal + resample
    fit_sample = curve_fit(f2, xData_calc, sample_data,p0=[1,1,1],bounds=((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf)))
    popt_sample, pcov_sample = fit_sample

    result_parameter = np.append(result_parameter,popt_sample)
    print('Sample Data ke - ',i)
    print(sample_data)
    resample = np.empty([0, 0])

result_parameter = np.resize(result_parameter,(banyak_simulasi,3))


#List Parameter

list_rate_initial = result_parameter[:,[2]]
list_b_initial = result_parameter[:,[1]]
list_decline_initial = result_parameter[:,[0]]

list_rate_initial = np.sort(list_rate_initial.T)
list_b_initial = np.sort(list_b_initial.T)
list_decline_initial = np.sort(list_decline_initial.T)

print(list_rate_initial)
print(list_b_initial)
print(list_decline_initial)


upper_prop = np.zeros(3)
most_prop = np.zeros(3)
low_bound= np.zeros(3)

#Generating P-10,P-50 and P-90 for Initial Pressure
upper_prop[2] = list_rate_initial[0][int(math.floor(banyak_simulasi*0.9-1))]
most_prop[2] = list_rate_initial[0][int(math.floor(banyak_simulasi*0.5-1))]
low_bound[2] = list_rate_initial[0][int(math.floor(banyak_simulasi*0.1-1))]

#Generating P-10,P-50 and P-90 for b
upper_prop[1] = list_b_initial[0][int(math.floor(banyak_simulasi*0.9-1))]
most_prop[1] = list_b_initial[0][int(math.floor(banyak_simulasi*0.5-1))]
low_bound[1] = list_b_initial[0][int(math.floor(banyak_simulasi*0.1-1))]
#
#Generating P-10,P-50 and P-90 for Decline rate
upper_prop[0] = list_decline_initial[0][int(math.floor(banyak_simulasi*0.9-1))]
most_prop[0] = list_decline_initial[0][int(math.floor(banyak_simulasi*0.5-1))]
low_bound[0] = list_decline_initial[0][int(math.floor(banyak_simulasi*0.1-1))]

print('Upper Bound : ',upper_prop)
print('Most Likely Bound : ' ,most_prop)
print('Lower Bound : ', low_bound)

#Generating Graph

plt.plot(xData_cek, cek, 'mo', label = 'experimental data cek')
plt.plot(xForecast, f2(xForecast, *upper_prop), 'r-', label='Data fit P90')
plt.plot(xForecast, f2(xForecast, *most_prop), 'm-', label='Data fit P50')
plt.plot(xForecast, f2(xForecast, *low_bound), 'c-', label='Data fit P10')

plt.xlabel('Time (Month)')
plt.ylabel('Data Production (bbl/m)')
plt.legend()
plt.show()






