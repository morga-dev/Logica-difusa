import numpy as np
import matplotlib.pyplot as plt
import fuzzy

# Universos de discurso
cpu_universe = np.linspace(0, 100, 101)        # Uso del CPU: 0 .. 100 %
time_universe = np.linspace(0, 1000, 1001)     # Tiempo de respuesta: 0 .. 1000 ms
ajuste_universe = np.linspace(0, 100, 101)     # Ajuste de recursos: 0 .. 100 %

# Definición de las funciones de membresía (trapezoidales)
CPU_MFs = {
    'Muy Bajo': (fuzzy.trapmf, [0, 0, 10, 25]),
    'Bajo'    : (fuzzy.trapmf, [10, 20, 35, 45]),
    'Medio'   : (fuzzy.trapmf, [35, 45, 65, 75]),
    'Alto'    : (fuzzy.trapmf, [65, 75, 88, 95]),
    'Muy Alto': (fuzzy.trapmf, [85, 95, 100, 100]),
}

TIME_MFs = {
    'Rápido'  : (fuzzy.trapmf, [0, 0, 100, 200]),
    'Normal'  : (fuzzy.trapmf, [100, 200, 300, 450]),
    'Lento'   : (fuzzy.trapmf, [300, 450, 600, 750]),
    'Crítico' : (fuzzy.trapmf, [600, 750, 1000, 1000]),
}

AJUSTE_MFs = {
    'Ninguno' : (fuzzy.trapmf, [0, 0, 5, 15]),
    'Bajo'    : (fuzzy.trapmf, [5, 15, 30, 40]),
    'Moderado': (fuzzy.trapmf, [30, 40, 60, 70]),
    'Alto'    : (fuzzy.trapmf, [60, 70, 85, 92]),
    'Máximo'  : (fuzzy.trapmf, [85, 92, 100, 100]),
}

# Etiquetas y su orden
CPU_LABELS = ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
TIME_LABELS = ['Rápido', 'Normal', 'Lento', 'Crítico']
AJUSTE_LABELS = ['Ninguno', 'Bajo', 'Moderado', 'Alto', 'Máximo']

# Funciones de fuzzificación
def fuzzify_input(value, mfs_dict):
    grados = {}
    for label, (f, params) in mfs_dict.items():
        grados[label] = float(f(value, params))
    return grados

# Regla difusa
def cpu_index(label):
    return CPU_LABELS.index(label)

def time_index(label):
    return TIME_LABELS.index(label)

def severity_to_ajuste_label(severity):
    if severity <= 2:
        return 'Ninguno'
    elif severity <= 4:
        return 'Bajo'
    elif severity <= 6:
        return 'Moderado'
    elif severity <= 8:
        return 'Alto'
    else:
        return 'Máximo'

# Creamos la base de reglas como lista de tuplas:
# (cpu_label, time_label, ajuste_label)
RULES = []
for c_label in CPU_LABELS:
    for t_label in TIME_LABELS:
        severity = 2*cpu_index(c_label) + time_index(t_label)
        a_label = severity_to_ajuste_label(severity)
        RULES.append((c_label, t_label, a_label))

# Evaluación de reglas y agregación
def infer(cpu_val, time_val, plot=False):
    """
    Ejecutamos el SID (Sistema de Inferencia Difuso): fuzzificación, activación de reglas (min), 
    recorte de consecuentes y agregación (max), y defuzzificación (centroid).
    """
    # Fuzzificación
    cpu_grados = fuzzify_input(cpu_val, CPU_MFs)
    time_grados = fuzzify_input(time_val, TIME_MFs)

    # Para guardar contribuciones por regla y detalles
    consequents_mfs = []   # lista de arreglos
    details = []           # lista de (cpu_label, time_label, ajuste_label, activation)

    # Recorremos todas las reglas
    for (c_label, t_label, a_label) in RULES:
        a_cpu = cpu_grados[c_label]
        a_time = time_grados[t_label]
        activation = min(a_cpu, a_time)   # conjunción por min()
        if activation > 0.0:
            # obtenemos la MF directa del consecuente
            f_consecuente, params_consecuente = AJUSTE_MFs[a_label]
            raw_mf = f_consecuente(ajuste_universe, params_consecuente)
            # recortamos con el grado de activación (método 'cut' / min)
            cut_mf = fuzzy.cut(activation, raw_mf)
            consequents_mfs.append(cut_mf)
            details.append((c_label, t_label, a_label, activation))

    # Agregación por máximo (union)
    if len(consequents_mfs) == 0:
        # Ninguna regla activada -> salida 0
        aggregated = np.zeros_like(ajuste_universe)
    else:
        aggregated = fuzzy.union(consequents_mfs)

    # Defuzzificación (centroid)
    # manejar caso división por cero en defuzz()
    if np.max(aggregated) == 0:
        ajuste_crisp = 0.0
    else:
        ajuste_crisp = float(fuzzy.defuzz(ajuste_universe, aggregated, 'centroid'))

    # Plotting
    if plot:
        plt.figure(figsize=(10, 5))
        # Graficar MFs de salida
        for label, (f, params) in AJUSTE_MFs.items():
            plt.plot(ajuste_universe, f(ajuste_universe, params), linestyle='--', alpha=0.5)
        # Graficar cada recorte 
        for cut_mf in consequents_mfs:
            plt.fill_between(ajuste_universe, np.zeros_like(ajuste_universe), cut_mf, alpha=0.25)
        # Graficar la agregación 
        plt.plot(ajuste_universe, aggregated, linewidth=2, label=f'Agregado (centroid={ajuste_crisp:.2f}%)')
        plt.title(f'Agregación de reglas para CPU={cpu_val}%, Tiempo={time_val}ms')
        plt.xlabel('Ajuste de recursos (%)')
        plt.ylabel('μ')
        plt.legend()
        plt.grid(True)
        plt.show()

    return ajuste_crisp, aggregated, details, cpu_grados, time_grados

# Ejemplo numérico de nuestra presentación:
#    CPU = 72%, Time = 380 ms
def ejemplo_presentacion():
    cpu_val = 72.0
    time_val = 380.0

    ajuste_crisp, aggregated, details, cpu_grados, time_grados = infer(cpu_val, time_val, plot=True)

    print("=== FUSIFICACIÓN (entradas) ===")
    print("CPU = {:.2f}% -> grados:".format(cpu_val))
    for k, v in cpu_grados.items():
        print(f"  {k:8s}: {v:.4f}")
    print("Tiempo = {:.2f} ms -> grados:".format(time_val))
    for k, v in time_grados.items():
        print(f"  {k:8s}: {v:.4f}")

    print("\n=== REGLAS ACTIVADAS (muestra) ===")
    # Ordenamos por activation descendente y mostramos las más significativas
    details_sorted = sorted(details, key=lambda x: x[3], reverse=True)
    for c_label, t_label, a_label, act in details_sorted[:12]:
        print(f"SI CPU es {c_label} Y Tiempo es {t_label} => Ajuste {a_label}  (activación={act:.4f})")

    print("\nValor defuzzificado (Ajuste de recursos): {:.2f} %".format(ajuste_crisp))

    # Mostrar los valores de las reglas específicas:
    print("\n--- Reglas que muestra la presentación (7..10) ---")
    search = [('Medio','Normal'), ('Medio','Lento'), ('Alto','Normal'), ('Alto','Lento')]
    for pair in search:
        for det in details:
            if det[0] == pair[0] and det[1] == pair[1]:
                print(f"Regla: CPU={det[0]} , Tiempo={det[1]} -> Ajuste={det[2]} , activ={det[3]:.4f}")

if __name__ == "__main__":
    ejemplo_presentacion()