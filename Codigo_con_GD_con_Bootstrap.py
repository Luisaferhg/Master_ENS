import pandapower as pp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.utils import resample

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

net=pp.create_empty_network()

#CREACIÓN DE BARRAJES
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="BUS")
for idx in df.index:
    pp.create_bus(net,name=df.at[idx,"name"],vn_kv=df.at[idx,"vn_kv"],max_vm_pu=df.at[idx,"max_vm_pu"],min_vm_pu=df.at[idx,"min_vm_pu"])
#     print (net.bus)

#CREACIÓN DE LINEAS
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="LINE")
for idx in df.index:
    pp.create_line_from_parameters(net,name=df.at[idx,"name"],from_bus=df.at[idx,"from_bus"],to_bus=df.at[idx,"to_bus"],length_km=df.at[idx,"length_km"],r_ohm_per_km=df.at[idx,"r_ohm_per_km"],x_ohm_per_km=df.at[idx,"x_ohm_per_km"],c_nf_per_km=df.at[idx,"c_nf_per_km"],max_i_ka=df.at[idx,"max_i_ka"])
#     print (net.line)
    Datos_linea=pd.DataFrame(net.line)

# CREACIÓN DE CARGA
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="LOAD")
for idx in df.index:
    pp.create_load(net,name=df.at[idx,"name"],bus=df.at[idx,"bus"],p_mw=df.at[idx,"p_mw"],q_mvar=df.at[idx,"q_mvar"])
#     print (net.load)
    Datos_carga=pd.DataFrame(net.load)
    
# CREACIÓN DE TRANSFORMADORES
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="TRAFO")
for idx in df.index:
    pp.create_transformer_from_parameters(net,name=df.at[idx,"name"],hv_bus=df.at[idx,"hv_bus"],lv_bus=df.at[idx,"lv_bus"],sn_mva=df.at[idx,"sn_mva"],vn_hv_kv=df.at[idx,"vn_hv_kv"],vn_lv_kv=df.at[idx,"vn_lv_kv"],vk_percent=df.at[idx,"vk_percent"],vkr_percent=df.at[idx,"vkr_percent"],pfe_kw=df.at[idx,"pfe_kw"],i0_percent=df.at[idx,"i0_percent"])
#     print (net.trafo)
    Datos_trafo=pd.DataFrame(net.trafo)
    
# CREACIÓN DE GENERADORES
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="GEN")
for idx in df.index:
    pp.create_gen(net,name=df.at[idx,"name"],bus=df.at[idx,"bus"],p_mw=df.at[idx,"p_mw"],vm_pu=df.at[idx,"vm_pu"],slack=df.at[idx,"slack"],vn_kv=df.at[idx,"vn_kv"])
#     print (net.gen)
    Datos_gen=pd.DataFrame(net.gen)
    
# CREACIÓN DE SLACK
df=pd.read_excel("IEEE39NODES.xlsx",sheet_name="SLACK")
for idx in df.index:
    pp.create_ext_grid(net,bus=df.at[idx,"bus"],vm_pu=df.at[idx,"vm_pu"],va_degree=df.at[idx,"va_degree"])
#     print (net.ext_grid)
    Datos_trafo=pd.DataFrame(net.trafo)
    
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Función para realizar resampling tipo bootstrap
def bootstrap_resample(data, n_samples):
    resamples = [resample(data, replace=True) for _ in range(n_samples)]
    return np.array(resamples)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
simulaciones = 3
ENS_simulaciones = []
ENS_10_años_total = []
# Inicializar los contadores y las variables de ENS

for sm in range(simulaciones):
    
    print("Simulación: ", sm + 1)

    iteraciones = 150
    ENS_1_año = []
    ENS_iteraciones = []

    for it in range (iteraciones):
        
        print("Simulación: ", sm + 1, "Iteración: ", it + 1)
        periodo = 20
        ENS_periodo = []
        tiempo_periodo = []
        
        num_generacion_distribuida = 0        
        net.sgen.drop(net.sgen.index, inplace=True) # elimina la generación distribuida que pudo crearse en el Periodo anterior
        
        for años in range (periodo):
            
            print("Simulación: ", sm + 1, "Iteración: ", it + 1, "periodo: ", años + 1)
            duracion_total = 4380
            amplitud = 1
            Val = Datos_linea.shape[0]
            Load_not_supply_total = []
            
            inicio_periodo = time.time()
            
#------------------------------------------------------------------------------------
#                    CREACIÓN CURVA DE CARGA DE GD
#------------------------------------------------------------------------------------    
          
            num_generacion_distribuida = num_generacion_distribuida + 1
            
            print ("numero de generadores distribuidos: ", num_generacion_distribuida)  

            generadores = []
            horas = np.arange(duracion_total) #duracion_total
            curvas_pv = {}
            buses_list = np.random.choice(38, size=num_generacion_distribuida, replace=False)
            print(buses_list)
           
            for gen_ind in range (num_generacion_distribuida):                

                buses = buses_list[gen_ind]
                potencias_pico_MW = np.random.uniform(7.5*0.4, 1104*0.4)    
                
                # Crear una curva de generación fotovoltaica para 24 horas (simulación simple)
                
                curva = np.maximum(0, np.sin((horas - 6) * np.pi / 12))  # Modelo básico de generación
                curva = curva / curva.max() * potencias_pico_MW  # Ajustar a la potencia máxima
                curvas_pv[buses] = curva
                curvas_pv = pd.DataFrame(curvas_pv)
                
                print(f"Se agrega generación en el bus: {buses} con potencia de: {potencias_pico_MW:.2f}")

            # Graficar las curvas de generación de todos los generadores
            plt.figure(figsize=(12, 6))
 
            for gen_i, curva_i in curvas_pv.items():
                plt.plot(horas, curva_i, marker='o', linestyle='-', label=f"Generador industial {gen_i}")
 
            plt.xlabel("Hora del día")
            plt.ylabel("Generación (MW)")
            plt.title("Curva de Generación Fotovoltaica por Nodo")
            plt.grid()
            plt.legend()
            plt.show()   
            
#------------------------------------------------------------------------------------
#                    CREACIÓN DEL REGIMEN DE OPERACIÓN DE LOS ACTIVOS
#------------------------------------------------------------------------------------           
                       
            n = [0] * Datos_carga.shape[0]
            t = [0] * Datos_carga.shape[0]
            ENS_por_bus = [0] * Datos_carga.shape[0]

#-------------------------------------------------------LINEAS-----------------------------------------------------------------

            TTF = [[] for _ in range(Val)]
            TTR = [[] for _ in range(Val)]
            T = [0] * Val
            LNS_Total = []

            for i in range(0, duracion_total, 1):
                            
                Falla_linea = 0.6 #Falla/año*km                            
                Datos_linea['f'] = Falla_linea / Datos_linea['length_km'] # 0.000799087 0.080308219
                Datos_linea['u'] = 0.3182
                Datos_linea['U1'] = np.random.rand(Val)
                Datos_linea['U2'] = np.random.rand(Val)
                            
                ttf_values = (-1 / Datos_linea.f * np.log(Datos_linea.U1)).to_numpy().astype(int)
                ttr_values = (-1 / Datos_linea.u * np.log(Datos_linea.U2)).to_numpy().astype(int)
                                    
                for j in range(Val):
                    
                    TTF[j].append(ttf_values[j])
                    TTR[j].append(ttr_values[j])
                    T[j] += ttf_values[j] + ttr_values[j]
                                
                    if T[j] > duracion_total:
                        
                        break

            tiempo = np.arange(0, duracion_total, 1)

            OP = [np.zeros_like(tiempo) for _ in range(Val)]

            for k in range(Val):
                
                indice_tiempo = 0
                
                for ttf, ttr in zip(TTF[k], TTR[k]):
                    OP[k][indice_tiempo:indice_tiempo + ttf] = amplitud
                    indice_tiempo += ttf
                    indice_tiempo += ttr
                    
                    if indice_tiempo >= len(tiempo):
                        break
                    
             Lineas_grafica = 5
 
             fig, axs = plt.subplots(Lineas_grafica, 1, figsize=(12, 10))  # 5 filas, 1 columna
             Colores_graficas = ['blue','pink','green','orange','purple']
 
             for k in Lineas_grafica:
                 axs[k].plot(tiempo, OP[k], drawstyle='steps-pre', label=f'Linea {k}', color=Colores_graficas[k])
                 axs[k].set_xlabel("Tiempo (h)",fontsize=10)
                 axs[k].set_ylabel("Amplitud",fontsize=10)
                 axs[k].set_yticks(np.arange(0, 1.5, 1)) 
                 axs[k].grid(True)
                 axs[k].legend()
             plt.subplots_adjust(hspace=0.8)
             plt.show()

#-------------------------------------------------------TRANSFORMADORES-----------------------------------------------------------------

            Val_tr = Datos_trafo.shape[0]
            TTF_TR = [[] for _ in range(Val_tr)]
            TTR_TR = [[] for _ in range(Val_tr)]
            T_TR = [0] * Val_tr

            for i in range(0, duracion_total, 1):
                
                for VT in range(Val_tr):
                    
                    if Datos_trafo['vn_hv_kv'].iloc[VT] == 138:
                        
                        Datos_trafo.loc[VT, 'f'] = 0.02210 # 0.0064
                        Datos_trafo.loc[VT, 'u'] = 0.119
                    
                    elif Datos_trafo['vn_hv_kv'].iloc[VT] == 230:
                        
                        Datos_trafo.loc[VT, 'f'] = 0.05351 # 0.0064
                        Datos_trafo.loc[VT, 'u'] = 0.80
                        
                    else:
                    
                        Datos_trafo.loc[VT, 'f'] = 0.02289 # 0.0064
                        Datos_trafo.loc[VT, 'u'] = 0.62
                        
                Datos_trafo['U1'] = np.random.rand(Val_tr)
                Datos_trafo['U2'] = np.random.rand(Val_tr)
                
                ttf_tr_values = (-1 / Datos_trafo.f * np.log(Datos_trafo.U1)).to_numpy().astype(int)
                ttr_tr_values = (-1 / Datos_trafo.u * np.log(Datos_trafo.U2)).to_numpy().astype(int)
                        
                for j in range(Val_tr):
                    TTF_TR[j].append(ttf_tr_values[j])
                    TTR_TR[j].append(ttr_tr_values[j])
                    T_TR[j] += ttf_tr_values[j] + ttr_tr_values[j]
                    
                    if T_TR[j] > duracion_total:
                        break

            tiempo_tr = np.arange(0, duracion_total, 1)

            OP_tr = [np.zeros_like(tiempo_tr) for _ in range(Val_tr)]

            for k in range(Val_tr):
                indice_tiempo_tr = 0
                
                for ttf_tr, ttr_tr in zip(TTF_TR[k], TTR_TR[k]):
                    OP_tr[k][indice_tiempo_tr:indice_tiempo_tr + ttf_tr] = amplitud
                    indice_tiempo_tr += ttf_tr
                    indice_tiempo_tr += ttr_tr
                    
                    if indice_tiempo_tr >= len(tiempo_tr):
                        break
                         
             trafos_grafica = 5
            
             fig, axs = plt.subplots(trafos_grafica, 1, figsize=(12, 10))
             Colores_graficas = ['blue','pink','green','orange','purple']
 
             for k in range(trafos_grafica):
                 axs[k].plot(tiempo_tr, OP_tr[k], drawstyle='steps-pre', label=f'Trafo {k}', color=Colores_graficas[k])
                 axs[k].set_xlabel("Tiempo (h)",fontsize=10)
                 axs[k].set_ylabel("Amplitud",fontsize=10)
                 axs[k].set_yticks(np.arange(0, 1.5, 1)) 
                 axs[k].grid(True)
                 axs[k].legend()
             plt.subplots_adjust(hspace=0.8) 
             plt.show()
            
#-------------------------------------------------------GENERADORES----------------------------------------------------------

            Val_gr = Datos_gen.shape[0]
            TTF_gr = [[] for _ in range(Val_gr)]
            TTR_gr = [[] for _ in range(Val_gr)]
            T_gr = [0] * Val_gr

            for i in range(0, duracion_total, 1):
                
                for VG in range(Val_gr):
                    
                    if Datos_gen['p_mw'].iloc[VG] == 1000:
                        
                        Datos_gen.loc[VG, 'f'] = 0.0080 # 0.0064
                        Datos_gen.loc[VG, 'u'] = 0.175
                    
                    elif Datos_gen['p_mw'].iloc[VG] >= 632 and Datos_gen['p_mw'].iloc[VG] <= 650:
                        
                        Datos_gen.loc[VG, 'f'] = 0.00717 # 0.0064
                        Datos_gen.loc[VG, 'u'] = 0.192

                    elif Datos_gen['p_mw'].iloc[VG] >= 508 and Datos_gen['p_mw'].iloc[VG] <= 560:
                        
                        Datos_gen.loc[VG, 'f'] = 0.0010 # 0.0064
                        Datos_gen.loc[VG, 'u'] = 0.4
                    
                    elif Datos_gen['p_mw'].iloc[VG] == 830:
                        
                        Datos_gen.loc[VG, 'f'] = 0.0054 # 0.0064
                        Datos_gen.loc[VG, 'u'] = 0.4
                        
                    else:
                    
                        Datos_trafo.loc[VT, 'f'] = 0.00104 # 0.0064
                        Datos_trafo.loc[VT, 'u'] = 0.04
                        
                Datos_gen['U1'] = np.random.rand(Val_gr)
                Datos_gen['U2'] = np.random.rand(Val_gr)
                
                ttf_gr_values = (-1 / Datos_gen.f * np.log(Datos_gen.U1)).to_numpy().astype(int)
                ttr_gr_values = (-1 / Datos_gen.u * np.log(Datos_gen.U2)).to_numpy().astype(int)
                        
                for j in range(Val_gr):
                    TTF_gr[j].append(ttf_gr_values[j])
                    TTR_gr[j].append(ttr_gr_values[j])
                    T_gr[j] += ttf_gr_values[j] + ttr_gr_values[j]
                    
                    if T_gr[j] > duracion_total:
                        break

            tiempo_gr = np.arange(0, duracion_total, 1)

            OP_gr = [np.zeros_like(tiempo_gr) for _ in range(Val_gr)]

            for k in range(Val_gr):
                indice_tiempo_gr = 0
                
                for ttf_gr, ttr_gr in zip(TTF_gr[k], TTR_gr[k]):
                    OP_gr[k][indice_tiempo_gr:indice_tiempo_gr + ttf_gr] = amplitud
                    indice_tiempo_gr += ttf_gr
                    indice_tiempo_gr += ttr_gr
                    
                    if indice_tiempo_gr >= len(tiempo_gr):
                        break
                    
             gen_grafica = 5
 
             fig, axs = plt.subplots(gen_grafica, 1, figsize=(12, 10))
             Colores_graficas = ['blue','pink','green','orange','purple']
 
             for k in range(gen_grafica):
                 axs[k].plot(tiempo_gr, OP_gr[k], drawstyle='steps-pre', label=f'Generador {k}', color=Colores_graficas[k])
                 axs[k].set_xlabel("Tiempo (h)",fontsize=10)
                 axs[k].set_ylabel("Amplitud",fontsize=10)
                 axs[k].set_yticks(np.arange(0, 1.5, 1)) 
                 axs[k].grid(True)
                 axs[k].legend()    
             plt.subplots_adjust(hspace=0.8) 
             plt.show()
            
#------------------------------------------------------------------------------------
#                               FLUJO DE POTENCIA
#------------------------------------------------------------------------------------
            
            print("...............Corriendo Flujo de potencia.....................")

            for a in range(duracion_total):
                            
                for i in range(Val):            
                    if OP[i][a] == 1:
                        net.line.loc[i, "in_service"] = True                               
                    else:
                        net.line.loc[i, "in_service"] = False
                
                for i in range(Val_tr):            
                    if OP_tr[i][a] == 1:
                        net.trafo.loc[i, "in_service"] = True                               
                    else:
                        net.trafo.loc[i, "in_service"] = False
                        
                for i in range(Val_gr):            
                    if OP_gr[i][a] == 1:
                        net.trafo.loc[i, "in_service"] = True                               
                    else:
                        net.trafo.loc[i, "in_service"] = False
                
                net.sgen.drop(net.sgen.index, inplace=True)

                for i in range(num_generacion_distribuida):
                        
                    sgen = pp.create_sgen(net, bus = buses_list[i], p_mw = curvas_pv.iloc[a, i], k = 1, type='PV', in_service = True)
                    #print("bus: ",net.sgen['bus'], "potencia: ",net.sgen['p_mw'])          
                                
                try:                    
                    
                    pp.runpp(net)                    
                                      
                    Load_not_supply = net.load.p_mw - net.res_load.p_mw
                    Load_not_supply_total.append(Load_not_supply)                
                    
                    for d in range(Datos_carga.shape[0]):
                        
                        if Load_not_supply[d] > 0:
                            
                            n[d] = 1
                            t[d] = t[d] + n[d]
                        
                        else:
                            
                            t[d] = 0
                        
                        ENS_por_bus[d] = (net.load.p_mw[d] - net.res_load.p_mw[d]) * t[d]
                    
                    ENS_total = sum(ENS_por_bus)

                except pp.LoadflowNotConverged:
                    
                    ENS_total = 0 #sum(net.load.p_mw)*duracion_total  # Si no converge, ENS es 0
                #             print(f"Iteración {mc + 1}: No convergió, ENS establecido en 0.")
                
                net.sgen.drop(net.sgen.index, inplace=True)
                
                       
            fin_periodo = time.time()
            tiempo_periodo = fin_periodo - inicio_periodo
            print("Tiempo periodo: ", tiempo_periodo)            
            
            ENS_periodo.append(ENS_total)
            ENS_periodo_df = pd.DataFrame(ENS_periodo)            
            ENS_periodo_df.to_excel(f"ENS_periodo_{años + 1}_iteracion_{it + 1}_simulacion_{sm + 1}.xlsx", index=False)
            
        n_bootstrap_it = 10000
        
        ENS_it_total = bootstrap_resample(ENS_periodo_df, n_bootstrap_it)    
        media_ENS_iteraciones = np.mean(ENS_it_total, axis=1)      
        ENS_iteraciones_total = pd.DataFrame({'Bootstrap Media': media_ENS_iteraciones.flatten()})
        ENS_iteraciones_total.to_excel(f"ENS_10000_iteraciones_{it+1}_Simulación_{sm+1}.xlsx", index=False)
        
        # Graficar los resultados de ENS_MC_1_año
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(ENS_iteraciones_total)),ENS_iteraciones_total['Bootstrap Media'], marker='o', linestyle='-', color='b', label='ENS Monte Carlo')
        plt.title('Evolución de ENS a lo largo de las simulaciones Monte Carlo')
        plt.xlabel('Simulación #')
        plt.ylabel('ENS total (MWh)')
        plt.grid(True)
        plt.legend()        

        # Guardar la gráfica en un archivo
        grafico_archivo = f"ENS_10000_iteraciones_{it+1}_Simulación_{sm+1}.png"
        plt.savefig(grafico_archivo)
        plt.close()  # Cerrar la gráfica para que no se muestre

        # Graficar un histograma de la distribución de ENS_MC
        plt.figure(figsize=(10, 6))
        plt.hist(ENS_iteraciones_total, bins=10, color='b', edgecolor='black', alpha=0.7)
        plt.title('Distribución de ENS a lo largo de las simulaciones Monte Carlo')
        plt.xlabel('ENS total (MWh)')
        plt.ylabel('Frecuencia')
        plt.grid(True)        

        # Guardar la gráfica en un archivo
        grafico_archivo = f"Histograma_ENS_10000_iteraciones_{it+1}_Simulación_{sm+1}.png"
        plt.savefig(grafico_archivo)
        plt.close()  # Cerrar la gráfica para que no se muestre

    n_bootstrap_simulaciones = 30 * simulaciones

    # Realizar resampling bootstrap
    bootstrap_results_simulaciones = bootstrap_resample(ENS_iteraciones_total, n_bootstrap_simulaciones)    
    media_bootstrap_simulaciones = np.mean(bootstrap_results_simulaciones, axis=1)      
    ENS_simul = pd.DataFrame({'Bootstrap Media': media_bootstrap_simulaciones.flatten()})
    ENS_simul.to_excel(f"Todas_las_simulaciones.xlsx", index=False)

    # Graficar los resultados de ENS_MC_1_año
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ENS_simul)),ENS_simul['Bootstrap Media'], marker='o', linestyle='-', color='b', label='ENS Monte Carlo')
    plt.title('Evolución de ENS a lo largo de las simulaciones Monte Carlo')
    plt.xlabel('Simulación #')
    plt.ylabel('ENS total (MWh)')
    plt.grid(True)
    plt.legend()        

    # Guardar la gráfica en un archivo
    grafico_archivo = f"Todas_las_simulaciones.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre

    # Graficar un histograma de la distribución de ENS_MC
    plt.figure(figsize=(10, 6))
    plt.hist(ENS_simul, bins=10, color='b', edgecolor='black', alpha=0.7)
    plt.title('Distribución de ENS a lo largo de las simulaciones Monte Carlo')
    plt.xlabel('ENS total (MWh)')
    plt.ylabel('Frecuencia')
    plt.grid(True)        

    # Guardar la gráfica en un archivo
    grafico_archivo = f"Histograma_todas_las_simulaciones.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre
