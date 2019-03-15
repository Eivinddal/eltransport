# Copyright © 2018 Mads-Emil Kvammen <mekvammen@gmail.com>
#
# Distributed under terms of the MIT license.
# https://no.wikipedia.org/wiki/MIT-lisensen

import numpy as np
from numpy.linalg import inv
import openpyxl
import cmath
import sys

def Tij(YbusG,YbusB,theta,i,j):
    return (YbusG.item(i,j)*np.cos(theta.item(i)-theta.item(j))+YbusB.item(i,j)*np.sin(theta.item(i)-theta.item(j)))
def Uij(YbusG,YbusB,theta,i,j):
    return (YbusG.item(i,j)*np.sin(theta.item(i)-theta.item(j))-YbusB.item(i,j)*np.cos(theta.item(i)-theta.item(j)))
def update_VT(bus_def, b, Theta, Volt, VT_estimate):
    count2 = 0
    for n in range(0, bus_def.size):
        if bus_def.item(n) == 0:
            continue
        elif bus_def.item(n) == 1:
            Theta.itemset(n, (Theta.item(n) + VT_estimate.item(count2)))
            count2 += 1
        else:
            d = 0
            for k in range(0, count2+1):
                if bus_def.item(k) == 1:
                    d += 1
            c = count2 + b - d
            Theta.itemset(n, (Theta.item(n) + VT_estimate.item(count2)))
            Volt.itemset(n, (Volt.item(n) + VT_estimate.item(c)))
            count2 += 1
    return Theta, Volt
def update_Theta(bus_def, Theta, dTheta):
    count2 = 0
    for n in range(0, bus_def.size):
        if bus_def.item(n) == 0:
            continue
        else:
            Theta.itemset(n, (Theta.item(n) + dTheta.item(count2)))
            count2 += 1
    return Theta
def update_V(bus_def, Volt, dV):
    count2 = 0
    for n in range(0, bus_def.size):
        if bus_def.item(n) == 0:
            continue
        elif bus_def.item(n) == 1:
            continue
        else:
            Volt.itemset(n, (Volt.item(n) + dV.item(count2)))
            count2 += 1
    return Volt
def update_PQ(bus_def, a, b, P_init, P_calc, Q_init, Q_calc):
    Delta_PQ = np.zeros((a, 1))
    count = 0
    for m in range(0, bus_def.size):
        if bus_def.item(m) == 0:
            continue
        elif bus_def.item(m) == 1:
            Delta_PQ.itemset((count, 0), P_init.item(m) - P_calc.item(m))
            count += 1
        else:
            d = 0
            for k in range(0, count+1):
                if bus_def.item(k) == 1:
                    d += 1

            c = count + b - d
            Delta_PQ.itemset((count, 0), P_init.item(m) - P_calc.item(m))
            Delta_PQ.itemset((c, 0), Q_init.item(m) - Q_calc.item(m))
            count += 1
    return Delta_PQ
def update_dP(bus_def, b, P_init, P_calc):
    Delta_P = np.zeros((b, 1))
    count = 0
    for m in range(0, bus_def.size):
        if bus_def.item(m) == 0:
            continue
        else:
            Delta_P.itemset((count, 0), P_init.item(m) - P_calc.item(m))
            count += 1
    return Delta_P
def update_dQ(bus_def, a, b, Q_init, Q_calc):
    Delta_Q = np.zeros((a-b, 1))
    count = 0
    for m in range(0, bus_def.size):
        if bus_def.item(m) == 0:
            continue
        elif bus_def.item(m) == 1:
            continue
        else:
            Delta_Q.itemset((count, 0), Q_init.item(m) - Q_calc.item(m))
            count += 1
    return Delta_Q
def Active(bus_def, YbusG, YbusB, theta, voltage, bus):
    p = 0
    for i in range(0, bus_def.size):
        if i == bus:
            continue
        p = p + voltage.item(i)*Tij(YbusG,YbusB,theta,bus,i)
    p = voltage.item(bus)**2*YbusG.item(bus,bus) + voltage.item(bus)*p
    return p
def Reactive(bus_def, YbusG, YbusB, theta, voltage, bus):
    q = 0
    for i in range(0, bus_def.size):
        if i == bus:
            continue
        q = q + voltage.item(i)*Uij(YbusG,YbusB,theta,bus,i)
    q = -voltage.item(bus)**2*YbusB.item(bus,bus) + voltage.item(bus)*q
    return q
def jacob_init(bus_def, YbusG, YbusB, theta, voltage):
    a = bus_def.sum()
    b = np.count_nonzero(bus_def)
    jacob = np.zeros((a,a))
    PQ = np.zeros((a,4), dtype=np.int)
    g = 0
    for h in range(0, bus_def.size):
        if bus_def.item(h) == 0:
            continue
        elif bus_def.item(h) == 1:
            PQ.itemset((g,0), 0) #Setting the uknown variable as P
            PQ.itemset((g,1), h) #Defining bus
            PQ.itemset((g,2), 0) #Setting the known variable as theta
            PQ.itemset((g,3), h) #Defining bus
            g = g + 1
        elif bus_def.item(h) == 2:
            d = 0
            for k in range(0, h):
                if bus_def.item(k) == 1:
                    d += 1
            c = g + b - d
            PQ.itemset((g, 0), 0)  # Setting the uknown variable as P
            PQ.itemset((g, 1), h)  # Defining bus
            PQ.itemset((g, 2), 0)  # Setting the known variable as theta
            PQ.itemset((g, 3), h)  # Defining bus
            PQ.itemset((c, 0), 1)  # Setting the uknown variable as Q
            PQ.itemset((c, 1), h)  # Defining bus
            PQ.itemset((c, 2), 1)  # Setting the known variable as V
            PQ.itemset((c, 3), h)  # Defining bus
            g = g + 1

    for i in range(0,a):
        for j in range(0,a):
            if PQ.item(i,0) == 0 and PQ.item(j,2) == 0: #Calculating dP/dTheta
                if PQ.item(i,1) == PQ.item(j,3): #If P and Theta are for the same bus
                    dPdT = 0
                    for k in range(0, bus_def.size):
                        if k == PQ.item(i,1):
                            continue
                        dPdT = dPdT + voltage.item(k)*Uij(YbusG,YbusB,theta,PQ.item(i,1),k)
                    dPdT = -voltage.item(PQ.item(i,1))*dPdT
                    jacob.itemset((i, j), dPdT)
                else:                           #If P and Theta are for different buses
                    jacob.itemset((i,j), (voltage.item(PQ.item(i,1))
                                       *voltage.item(PQ.item(j,3))
                                       *Uij(YbusG,YbusB,theta,PQ.item(i,1),PQ.item(j,3))))
            if PQ.item(i,0) == 0 and PQ.item(j,2) == 1: #Calculating dP/dV
                if PQ.item(i,1) == PQ.item(j,3): #If P and V are for the same bus
                    dPdV = 0
                    for k in range(0, bus_def.size):
                        if k == PQ.item(i,1):
                            continue
                        dPdV = dPdV + voltage.item(k) * Tij(YbusG, YbusB, theta, PQ.item(i, 1), k)
                    dPdV = 2*voltage.item(PQ.item(i,1))*YbusG.item(PQ.item(i,1),PQ.item(i,1)) + dPdV
                    jacob.itemset((i, j), dPdV)
                else:                           #If P and V are for different buses
                    jacob.itemset((i,j), (voltage.item(PQ.item(i,1))
                                       *Tij(YbusG,YbusB,theta,PQ.item(i,1),PQ.item(j,3))))
            if PQ.item(i, 0) == 1 and PQ.item(j, 2) == 0: #Calculating dQ/dTheta
                if PQ.item(i, 1) == PQ.item(j, 3): #If Q and Theta are for the same bus
                    dPdT = 0
                    for k in range(0, bus_def.size):
                        if k == PQ.item(i,1):
                            continue
                        dPdT = dPdT + voltage.item(k) * Tij(YbusG, YbusB, theta, PQ.item(i, 1), k)
                    dPdT = voltage.item(PQ.item(i, 1))*dPdT
                    jacob.itemset((i, j), dPdT)
                else:                           #If Q and Theta are for different buses
                    jacob.itemset((i, j), (-voltage.item(PQ.item(i, 1))
                                        * voltage.item(PQ.item(j, 3))
                                        * Tij(YbusG, YbusB, theta, PQ.item(i, 1), PQ.item(j, 3))))
            if PQ.item(i, 0) == 1 and PQ.item(j, 2) == 1: #Calculating dQ/dV
                if PQ.item(i, 1) == PQ.item(j, 3): #If C and V are for the same bus
                    dPdV = 0
                    for k in range(0, bus_def.size):
                        if k == PQ.item(i,1):
                            continue
                        dPdV = dPdV + voltage.item(k) * Uij(YbusG, YbusB, theta, PQ.item(i, 1), k)
                    dPdV = -(2 * voltage.item(PQ.item(i, 1)) * YbusB.item(PQ.item(i, 1), PQ.item(i, 1))) + dPdV
                    jacob.itemset((i, j), dPdV)
                else:                           #If C and V are for different buses
                    jacob.itemset((i, j), (voltage.item(PQ.item(i, 1))
                                        * Uij(YbusG, YbusB, theta, PQ.item(i, 1), PQ.item(j, 3))))
    return a, b, PQ, jacob
def Check_violations(bus_def, bus_def_original, i, Volt, Volt_init, Q_init, Q_calc, Q_max, Q_min, again, stop):
    for j in range(0, bus_def_original.size):
        if bus_def_original[0,j] == 1 and bus_def[0,j] == 1:
            if Q_calc[j] >= Q_max[0,j]:
                bus_def[0, j] = 2
                Q_init[0,j] = Q_max[0,j]
                again = 1
                stop = 0
                print('\nBus number {}, changed to a PQ-bus in iteration {}.'.format(j+1,i))
            elif Q_calc[j] <= Q_min[0,j]:
                bus_def[0, j] = 2
                Q_init[0,j] = Q_min[0,j]
                again = 1
                stop = 0
                print('\nBus number {}, changed to a PQ-bus in iteration {}.'.format(j+1,i))
        elif bus_def_original[0,j] == 1 and bus_def[0,j] == 2:
            if (Q_calc[j] >= Q_max[0,j] and Volt[0,j] > Volt_init[0,j]) or (Q_calc[j] <= Q_min[0,j] and Volt[0,j] < Volt_init[0,j]):
                bus_def[0,j] = 1
                Volt[0,j] = Volt_init[0,j]
                again = 1
                stop = 0
                print('\nBus number {}, changed back to a PV-bus in iteration {}.'.format(j+1,i))
    return bus_def, Q_init, Volt, again, stop
def NR_method(iter, convergence, bus_def, YbusG, YbusB, theta, Volt, P_init, Q_init, Q_max, Q_min, from_line, to_line, B, Line_adm):
    P_calc = np.zeros(bus_def.size)
    Q_calc = np.zeros(bus_def.size)
    bus_def_original = bus_def.copy()
    Volt_init = Volt.copy()

    for j in range(bus_def.size):
        P_calc.itemset(j, Active(bus_def, YbusG, YbusB, theta, Volt, j))
        Q_calc.itemset(j, Reactive(bus_def, YbusG, YbusB, theta, Volt, j))


    i = 1
    a, b, PQ, jacob = jacob_init(bus_def, YbusG, YbusB, theta, Volt)
    jacob_inv = inv(jacob)
    Delta_PQ = update_PQ(bus_def, a, b, P_init, P_calc, Q_init, Q_calc)

    stop = 1
    for k in range(0, Delta_PQ.size):
        if Delta_PQ[k] > convergence:
            stop = 0
    if stop == 1:
        print('\nConvergence in {} iterations.'.format(i))
    VT_estimate = np.dot(jacob_inv, Delta_PQ)  # Calculating dTheta and dV
    theta, Volt = update_VT(bus_def, b, theta, Volt, VT_estimate)
    again = 1
    while again == 1:
        again = 0
        while i < iter and stop == 0:
            for j in range(bus_def.size):
                P_calc.itemset(j, Active(bus_def, YbusG, YbusB, theta, Volt, j))
                Q_calc.itemset(j, Reactive(bus_def, YbusG, YbusB, theta, Volt, j))


            a, b, PQ, jacob = jacob_init(bus_def, YbusG, YbusB, theta, Volt)
            jacob_inv = inv(jacob)
            Delta_PQ = update_PQ(bus_def, a, b, P_init, P_calc, Q_init, Q_calc)
            stop = 1
            for k in range(0, Delta_PQ.size):
                if np.abs(Delta_PQ[k]) > convergence:
                    stop = 0
            VT_estimate = np.dot(jacob_inv, Delta_PQ)  # Calculating dTheta and dV
            theta, Volt = update_VT(bus_def, b, theta, Volt, VT_estimate)
            i = i + 1

        if i == iter and stop == 0:
            print('\nNo solution found.\nDid not converge in {} iterations.'.format(i))
            sys.exit()
        #else:
        #    bus_def, Q_init, Volt, again, stop = Check_violations(bus_def, bus_def_original, i, Volt, Volt_init, Q_init, Q_calc, Q_max,
        #                                         Q_min, again, stop)
        if stop == 1:
            print('\nConvergence in {} iterations.'.format(i-1))

    S, loss = Line_flow(Volt, theta, from_line, to_line, B, Line_adm)
    return P_calc, Q_calc, Volt, theta, jacob, a, b, PQ, VT_estimate, Delta_PQ, i, S, loss, bus_def_original
def Line_flow(Volt, theta, from_line, to_line, B, Line_adm):
    Volt_complex = np.zeros((Volt.size), dtype=complex)
    for i in range(Volt.size):
        Volt_complex[i] = Volt[0,i]*cmath.exp(theta[0,i]*complex(0,1))
    S = np.zeros((from_line.size,10))
    for i in range(0, from_line.size):
        S[i, 0] = from_line[0,i]
        S[i, 1] = to_line[0,i]
        S[i, 2] = np.real(Volt_complex[from_line[0,i]-1]*(
            ((np.conj(Volt_complex[from_line[0,i]-1])*np.conj(complex(0,B[i])))/2)+
                                                            (np.conj(Volt_complex[from_line[0,i]-1])-np.conj(Volt_complex[to_line[0,i]-1]))
                                                            *np.conj(-Line_adm[i])))
        S[i, 3] = np.imag(Volt_complex[from_line[0, i] - 1] * (
                ((np.conj(Volt_complex[from_line[0, i] - 1]) * np.conj(complex(0, B[i]))) / 2) +
                    (np.conj(Volt_complex[from_line[0, i] - 1]) - np.conj(Volt_complex[to_line[0, i] - 1]))
                    * np.conj(-Line_adm[i])))
        S[i, 6] = np.imag(Volt_complex[from_line[0, i] - 1] * (
                (np.conj(Volt_complex[from_line[0, i] - 1]) - np.conj(Volt_complex[to_line[0, i] - 1]))
                * np.conj(-Line_adm[i])))
        S[i, 8] = np.imag(Volt_complex[from_line[0, i] - 1] * (
                ((np.conj(Volt_complex[from_line[0, i] - 1]) * np.conj(complex(0, B[i]))) / 2)))

        S[i, 4] = np.real(Volt_complex[to_line[0, i] - 1] * (
                    ((np.conj(Volt_complex[to_line[0, i] - 1]) * np.conj(complex(0, B[i]))) / 2) +
                    (np.conj(Volt_complex[to_line[0, i] - 1]) - np.conj(Volt_complex[from_line[0, i] - 1]))
                    * np.conj(-Line_adm[i])))
        S[i, 5] = np.imag(Volt_complex[to_line[0, i] - 1] * (
            ((np.conj(Volt_complex[to_line[0, i] - 1]) * np.conj(complex(0, B[i]))) / 2) +
                (np.conj(Volt_complex[to_line[0, i] - 1]) - np.conj(Volt_complex[from_line[0, i] - 1]))
                * np.conj(-Line_adm[i])))
        S[i, 7] = np.imag(Volt_complex[to_line[0, i] - 1] * (
                (np.conj(Volt_complex[to_line[0, i] - 1]) - np.conj(Volt_complex[from_line[0, i] - 1]))
                * np.conj(-Line_adm[i])))
        S[i, 9] = np.imag(Volt_complex[to_line[0, i] - 1] * (
                ((np.conj(Volt_complex[to_line[0, i] - 1]) * np.conj(complex(0, B[i]))) / 2)))
    loss = np.zeros((from_line.size,4))
    for k in range(0, from_line.size):
        loss[k,0] = from_line[0,k]
        loss[k,1] = to_line[0,k]
        loss[k,2] = S[k,2] + S[k,4]
        loss[k,3] = S[k,6] + S[k,7]

    return S, loss
def ReadSystemData(filenam,Ark):
    wb = openpyxl.load_workbook(filename=filenam, data_only=True)
    sheet = wb[Ark]
    all_rows = []
    for row in sheet:
        if row[0].value == None:
            continue
        current_row = []
        for cell in row:
            current_row.append(cell.value)
        all_rows.append(current_row)
    del all_rows[0:1]

    arr = np.asarray(all_rows)
    return arr
def BusData(filenam,bus_data_sheet):
    bus_data = ReadSystemData(filenam, bus_data_sheet)
    bus_def = np.matrix(bus_data[:, 1])
    V_base = np.matrix(bus_data[0, 11])
    Volt0 = np.matrix(bus_data[:, 2]) / V_base
    theta0 = np.matrix(bus_data[:, 3])
    S_base = np.matrix(bus_data[0, 10])
    P_init = np.matrix(bus_data[:, 4]) / S_base  # Slack bus is give the initial value = 0
    Q_init = np.matrix(bus_data[:, 5]) / S_base  # Slack bus is give the initial value = 0
    P_load = -np.matrix(bus_data[:, 6]) / S_base
    Q_load = -np.matrix(bus_data[:, 7]) / S_base
    Q_max = 0
    Q_min = 0
    # Q_max = np.matrix(bus_data[:, 8]) / S_base # Not to be included in this project
    # Q_min = np.matrix(bus_data[:, 9]) / S_base # Not to be included in this project

    P_init = P_init + P_load
    Q_init = Q_init + Q_load

    return bus_def, Volt0, theta0, S_base, P_init, Q_init, Q_max, Q_min, P_load, Q_load
def BranchData(filenam,branch_data_sheet,bus_def):
    branch_data = ReadSystemData(filenam, branch_data_sheet)

    from_line = np.matrix(branch_data[:, 0], dtype=int)
    to_line = np.matrix(branch_data[:, 1], dtype=int)
    R = branch_data[:, 2]
    X = branch_data[:, 3]
    B = branch_data[:, 4]

    Ybus = np.zeros((bus_def.size, bus_def.size),dtype=complex)
    for i in range(0,bus_def.size):
        for j in range(0,from_line.size):
            if i + 1 == from_line[0,j] or i + 1 == to_line[0,j]:
                Ybus[i,i] = Ybus[i,i] + 1/complex(R[j],X[j]) + complex(0,B[j]) / 2
    Line_adm = np.zeros(from_line.size, dtype=complex)
    for k in range(0,from_line.size):
        Line_adm[k] = -1 / complex(R[k], X[k])
        if Ybus[from_line[0,k]-1, to_line[0,k]-1] == 0:
            Ybus[from_line[0,k]-1, to_line[0,k]-1] = -1 / complex(R[k], X[k])
            Ybus[to_line[0,k]-1, from_line[0,k]-1] = -1 / complex(R[k], X[k])
        else:
            Ybus[from_line[0, k] - 1, to_line[0, k] - 1] = 1 / (-complex(R[k], X[k]))+Ybus[from_line[0, k] - 1, to_line[0, k] - 1]
            Ybus[to_line[0, k] - 1, from_line[0, k] - 1] = 1 / (-complex(R[k], X[k]))+Ybus[to_line[0, k] - 1, from_line[0, k] - 1]

    return Ybus, from_line, to_line, B, Line_adm
def Load_flow(filenam, branch_data_sheet, bus_data_sheet):

    ## Initialisation
    bus_def, Volt0, theta0, S_base, P_init, Q_init, Q_max, Q_min, P_load, Q_load = BusData(filenam, bus_data_sheet)
    Ybus, from_line, to_line, B,Line_adm = BranchData(filenam, branch_data_sheet, bus_def)
    YbusG = np.real(Ybus)
    YbusB = np.imag(Ybus)


    ## Newton-Rapsons method
    theta = theta0.copy()
    Volt = Volt0.copy()
    convergence = 1.0e-10 # Convergence criterion for dP and dQ.
    iter = 100 # Number of iterations wanted in NR-Method. Overridden if convergence criterion is met.
    P_calc, Q_calc, Volt, theta, jacob, a, b, PQ, dVdT, dPdQ, numb_iter, S, loss, bus_def_original = NR_method(iter,convergence, bus_def, YbusG,
                                                                                    YbusB, theta, Volt, P_init, Q_init,
                                                                                    Q_max, Q_min, from_line, to_line, B, Line_adm)
    P_gen = np.zeros(bus_def.size)
    Q_gen = np.zeros(bus_def.size)
    P_load_calc = np.zeros(bus_def.size)
    Q_load_calc = np.zeros(bus_def.size)
    for i in range(bus_def.size):
        if bus_def_original[0,i] == 0:
            P_gen[i] = P_calc[i] - P_load[0,i]
            Q_gen[i] = Q_calc[i] - Q_load[0,i]
            P_load_calc[i] = P_load[0,i]
            Q_load_calc[i] = Q_load[0,i]
        if bus_def_original[0, i] == 1:
            P_gen[i] = P_calc[i] - P_load[0, i]
            Q_gen[i] = Q_calc[i] - Q_load[0, i]
            P_load_calc[i] = P_load[0, i]
            Q_load_calc[i] = Q_load[0, i]
        elif bus_def_original[0,i] == 2:
            P_load_calc[i] = P_calc[i]
            Q_load_calc[i] = Q_calc[i]

    theta = theta*180/np.pi
    BusInfo = np.vstack((Volt, theta, P_gen, Q_gen, P_load_calc, Q_load_calc)).T
    total = np.zeros([4])
    for k in range(len(BusInfo)):
        total[0] = total[0] + BusInfo[k, 2]
        total[1] = total[1] + BusInfo[k, 3]
        total[2] = total[2] + BusInfo[k, 4]
        total[3] = total[3] + BusInfo[k, 5]

    total_loss = np.zeros([2])
    for k in range(len(loss)):
        total_loss[0] = total_loss[0] + loss[k, 2]
        total_loss[1] = total_loss[1] + loss[k, 3]

    return BusInfo, total, S, loss, total_loss, Ybus, YbusG, YbusB