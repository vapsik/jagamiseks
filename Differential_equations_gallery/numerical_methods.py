import matplotlib.pyplot as plt
import numpy as np

dt = 0.05
maxtime = 50

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
line, = ax.plot([],[], "b-", lw = 2, label="explicit" )
line1, = ax.plot([],[], "g-", lw = 2, label = "implicit")
lineA, = ax.plot([],[], "y-", lw = 2, label = "analytical")
lineRK2, = ax.plot([],[], "r-", lw = 2, label = "RK2")
lineRK4, = ax.plot([],[], "black", lw = 2, label = "RK4")
#lineLF, = ax.plot([],[], "purple", lw = 2, label = "Leap-Frog-ish")
ax.set_title("explicit = blue, implicit = green, RK2 = red")

legend = plt.legend()

time_data = []
x_data = []

x = 1.0
v = 0.0
t = 0.0

x_imp_data = []
x_imp = 1.0
v_imp = 0.0

x_RK2_data = []
x_RK2 = 1.0
v_RK2 = 0.0

x_RK4_data = []
x_RK4 = 1.0
v_RK4 = 0.0

x_LF_data = []
v_LF = 0.0
x_LF = 1.0

x_data.append(x)
x_imp_data.append(x_imp)
x_RK2_data.append(x_RK2)
x_RK4_data.append(x_RK4)
x_LF_data.append(x_LF)

time_data.append(t)

plt.grid()

while t < maxtime:

    #explicit
    # x(t+dt) = x(t) + v(t)*dt
    # v(t+dt) = v(t) + a*dt, a = -x(t)
    a = -x
    x_uus = x + v*dt
    v_uus = v + a*dt

    x, v = x_uus, v_uus
    t += dt
    x_data.append(x)
    time_data.append(t)


    line.set_data(time_data, x_data)
    #implicit
    # x(t+dt) = x(t) + v(t+dt)*dt
    # v(t+dt) = v(t) + a*dt, a = -x(t+dt)
    # ==> x(t+dt) = (x(t) + v(t)*dt)/(1 + (dt)^2)
    # ==> v(t+dt) = (v(t) - x(t)*dt)/(1 + (dt)^2)
    
    x_imp_uus = (x_imp + v_imp * dt)/ (1 + dt*dt)
    v_imp_uus = (v_imp - x_imp * dt)/ (1 + dt*dt)

    v_imp, x_imp = v_imp_uus, x_imp_uus
    x_imp_data.append(x_imp)

    line1.set_data(time_data, x_imp_data)

    
    #RK2
    x_RK2 = (x + x_imp)/2.0

    x_RK2_data.append(x_RK2)
    lineRK2.set_data(time_data, x_RK2_data)

    #RK4
    #Wikipediast: v(t+dt) = v(t) + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)*dt
    #k_1 = f(t, x(t)) = -x(t)
    #k_2 = f(t + 0.5 * dt, x(t) + 0.5 * k_1*dt) = -(x + 0.5 * k_1 * dt)
    #k_3 = f(t + 0.5 * dt, x(t) + 0.5 * k_2*dt) = -(x + 0.5 * k_2 * dt)
    #k_4 = f(t + dt, x(t) + k_3 * dt) = -(x + k_3 * dt)
    #
    #x(t + dt) töötab analoogiliselt
    #f = f(t, v) x(t+dt) jaoks
    
    #-selle rakendamine x-i jaoks oli vale 

    #õige implementatsioon all: x ja v kirjeldavad k-d vahelduvad millegipärast
    #pole veel kindel miks
    #chat-gpt pakutud all:

    k1_x = v_RK4
    k1_v = -x_RK4

    k2_x = v_RK4 + 0.5 * k1_v * dt
    k2_v = -(x_RK4 + 0.5 * k1_x * dt)
    
    k3_x = v_RK4 + 0.5 * k2_v * dt
    k3_v = -(x_RK4 + 0.5 * k2_x * dt)
    
    k4_x = v_RK4 + k3_v * dt
    k4_v = -(x_RK4 + k3_x * dt)

    x_RK4 += (1/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x) * dt
    v_RK4 += (1/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt
    
    x_RK4_data.append(x_RK4)
    lineRK4.set_data(time_data, x_RK4_data)

    
    #leap frog-ish
    #x_LF = x_LF + v_LF*dt + 0.5 * (-x) * dt * dt
    #v_LF = v_LF + (-x)*dt
    #x_LF_data.append(x_LF)
    #lineLF.set_data(time_data, x_LF_data)

    #analüütiline
    lineA.set_data(time_data, np.cos(np.array(time_data)))
    ax.set_xlim(max(0, t - 10), t + 1)
    plt.pause(0.01)

    