#Para correr la interface = "python gui.py" en terminal
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

import fuzzy
from control_caso_estudio import infer, ajuste_universe, AJUSTE_MFs, CPU_MFs, TIME_MFs, cpu_universe, time_universe


def evaluar(event=None):
    try:
        cpu = float(cpu_var.get())
        t = float(time_var.get())
    except Exception:
        messagebox.showerror("Error", "Introduce valores numéricos válidos dentro de los rangos.")
        return

    ajuste_crisp, aggregated, details, cpu_grados, time_grados = infer(cpu, t, plot=False)
    result_var.set(f"{ajuste_crisp:.2f} %")

    # Actualizar lista de reglas activadas
    rules_listbox.delete(0, tk.END)
    if details:
        details_sorted = sorted(details, key=lambda x: x[3], reverse=True)
        for c_label, t_label, a_label, act in details_sorted:
            rules_listbox.insert(tk.END, f"SI CPU={c_label} Y Tiempo={t_label} => {a_label}  (act={act:.3f})")
    else:
        rules_listbox.insert(tk.END, "Ninguna regla activada")

    # Dibujar: limpiar todos los ejes
    ax_cpu.clear()
    ax_time.clear()
    ax_ajuste.clear()
    ax_main.clear()

    # CPU MFs (gráfica 1)
    for label, (f, params) in CPU_MFs.items():
        ax_cpu.plot(cpu_universe, f(cpu_universe, params), label=label)
    ax_cpu.set_title('Uso del CPU (%)')
    ax_cpu.set_xlabel('CPU (%)')
    ax_cpu.set_ylabel('μ')
    ax_cpu.set_ylim(0, 1.05)
    ax_cpu.legend(fontsize='small')

    # Dibujar indicador del valor de CPU y sombrear recortes según activación
    try:
        ax_cpu.axvline(cpu, color='k', linestyle='--', linewidth=1)
        for label, (f, params) in CPU_MFs.items():
            raw = f(cpu_universe, params)
            act = cpu_grados.get(label, 0.0)
            if act > 0:
                cut = fuzzy.cut(act, raw)
                ax_cpu.fill_between(cpu_universe, 0, cut, alpha=0.18)
                ax_cpu.plot([cpu], [act], marker='o', color='k', markersize=4)
    except Exception:
        pass

    # Tiempo MFs (gráfica 2)
    for label, (f, params) in TIME_MFs.items():
        ax_time.plot(time_universe, f(time_universe, params), label=label)
    ax_time.set_title('Tiempo de respuesta (ms)')
    ax_time.set_xlabel('ms')
    ax_time.set_ylim(0, 1.05)
    ax_time.legend(fontsize='small')

    # Dibujar indicador del valor de Tiempo y sombrear recortes según activación
    try:
        ax_time.axvline(t, color='k', linestyle='--', linewidth=1)
        for label, (f, params) in TIME_MFs.items():
            raw = f(time_universe, params)
            act = time_grados.get(label, 0.0)
            if act > 0:
                cut = fuzzy.cut(act, raw)
                ax_time.fill_between(time_universe, 0, cut, alpha=0.18)
                ax_time.plot([t], [act], marker='o', color='k', markersize=4)
    except Exception:
        pass

    # Ajuste MFs base (gráfica 3)
    for label, (f, params) in AJUSTE_MFs.items():
        ax_ajuste.plot(ajuste_universe, f(ajuste_universe, params), label=label)
    ax_ajuste.set_title('Ajuste de recursos (%) - MFs')
    ax_ajuste.set_xlabel('%')
    ax_ajuste.set_ylim(0, 1.05)
    ax_ajuste.legend(fontsize='small')

    # Mostrar recortes por etiqueta de salida (activación máxima por etiqueta)
    try:
        output_acts = {label: 0.0 for label in AJUSTE_MFs.keys()}
        for (_c_label, _t_label, a_label, activation) in details:
            output_acts[a_label] = max(output_acts.get(a_label, 0.0), activation)
        for label, (f, params) in AJUSTE_MFs.items():
            act = output_acts.get(label, 0.0)
            if act > 0:
                raw = f(ajuste_universe, params)
                cut = fuzzy.cut(act, raw)
                ax_ajuste.fill_between(ajuste_universe, 0, cut, alpha=0.18)
        # línea vertical en la salida defuzzificada
        ax_ajuste.axvline(ajuste_crisp, color='k', linestyle='--', linewidth=1)
    except Exception:
        pass

    # Agregado y recortes en el eje principal (debajo)
    for (c_label, t_label, a_label, activation) in details:
        f_con, params_con = AJUSTE_MFs[a_label]
        raw = f_con(ajuste_universe, params_con)
        cut = fuzzy.cut(activation, raw)
        ax_main.fill_between(ajuste_universe, 0, cut, alpha=0.25)

    ax_main.plot(ajuste_universe, aggregated, linewidth=2, color='black', label='Agregado')
    ax_main.set_xlabel('Ajuste de recursos (%)')
    ax_main.set_ylabel('μ')
    ax_main.set_title(f'Agregado y defuzzificado: Ajuste={ajuste_crisp:.2f}%')
    ax_main.set_xlim(ajuste_universe[0], ajuste_universe[-1])
    ax_main.set_ylim(0, 1.05)
    ax_main.legend(loc='upper right', fontsize='small')
    canvas.draw()

    # Actualizar panel de explicación
    try:
        explain = generate_explanation(ajuste_crisp, details, cpu, t)
        explain_text.config(state='normal')
        explain_text.delete('1.0', tk.END)
        explain_text.insert(tk.END, explain)
        explain_text.config(state='disabled')
    except Exception:
        pass


def set_example():
    cpu_scale.set(72.0)
    time_scale.set(380.0)
    cpu_var.set('72.0')
    time_var.set('380.0')
    # seleccionar etiqueta más representativa para la CPU en el combobox
    try:
        cpu_val = 72.0
        best = None
        best_v = -1
        for label, (f, params) in CPU_MFs.items():
            v = float(f(cpu_val, params))
            if v > best_v:
                best_v = v
                best = label
        if best:
            cpu_level_cb.set(best)
    except Exception:
        pass
    evaluar()


def generate_explanation(ajuste_crisp, details, cpu, t):
    parts = []
    parts.append(f"Entrada: CPU={cpu:.1f}%  Tiempo={t:.1f} ms")
    parts.append(f"Ajuste sugerido (defuzzificado): {ajuste_crisp:.2f}%")
    parts.append("")
    parts.append("Cómo leer la gráfica:")
    parts.append("- Líneas punteadas: funciones de membresía de salida (etiquetas).")
    parts.append("- Áreas sombreadas: contribuciones recortadas por cada regla activada; su altura indica el grado de activación.")
    parts.append("- Curva negra (Agregado): es la unión de todas las contribuciones; la defuzzificación calcula el centroide sobre esta curva.")
    parts.append("- Reglas listadas: muestran las reglas activadas y su nivel de influencia (mayor activación → más peso).")
    parts.append("")
    parts.append("Interpretación rápida:")
    if ajuste_crisp < 10:
        parts.append("- Ajuste bajo: no es necesario aumentar recursos.")
    elif ajuste_crisp < 40:
        parts.append("- Ajuste moderado: se recomienda una pequeña asignación de recursos.")
    elif ajuste_crisp < 70:
        parts.append("- Ajuste notable: conviene aumentar recursos para mejorar la respuesta.")
    else:
        parts.append("- Ajuste alto/máximo: asignar recursos con prioridad.")

    if details:
        top = sorted(details, key=lambda x: x[3], reverse=True)[:3]
        parts.append("")
        parts.append("Reglas con mayor activación:")
        for c_label, t_label, a_label, act in top:
            parts.append(f"  SI CPU={c_label} Y Tiempo={t_label} => {a_label}  (act={act:.3f})")

    return "\n".join(parts)


root = tk.Tk()
root.title('Interfaz Fuzzy - Ajuste de Recursos')
root.geometry('900x500')

# Layout: izquierda controles, derecha plot + reglas
main = ttk.Frame(root, padding=10)
main.pack(fill=tk.BOTH, expand=True)

left = ttk.Frame(main)
left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

ttk.Label(left, text='Entradas', font=('Segoe UI', 11, 'bold')).pack(anchor='w')

# CPU: slider + entry
ttk.Label(left, text='CPU (%)').pack(anchor='w', pady=(6,0))
cpu_var = tk.StringVar(value='72.0')
cpu_scale = ttk.Scale(left, from_=0, to=100, orient='horizontal')
cpu_scale.set(72.0)
cpu_scale.pack(fill=tk.X)

def on_cpu_scale(e):
    val = cpu_scale.get()
    cpu_var.set(f"{val:.1f}")

cpu_scale.bind('<B1-Motion>', on_cpu_scale)
cpu_scale.bind('<ButtonRelease-1>', on_cpu_scale)

cpu_entry = ttk.Entry(left, textvariable=cpu_var, width=10)
cpu_entry.pack(anchor='w', pady=(4,0))

def on_cpu_entry(event=None):
    try:
        v = float(cpu_var.get())
        v = max(0.0, min(100.0, v))
        cpu_scale.set(v)
        cpu_var.set(f"{v:.1f}")
    except Exception:
        pass

cpu_entry.bind('<Return>', on_cpu_entry)

# CPU nivel (texto) - Combobox para seleccionar etiquetas como 'Bajo', 'Muy Bajo', etc.
cpu_level_var = tk.StringVar()
cpu_level_cb = ttk.Combobox(left, textvariable=cpu_level_var, values=list(CPU_MFs.keys()), state='readonly')
cpu_level_cb.pack(anchor='w', pady=(6,0))

def on_cpu_level_select(event=None):
    label = cpu_level_var.get()
    if not label:
        return
    try:
        params = CPU_MFs[label][1]
        # tomar el punto central de la meseta (b,c)
        b = float(params[1])
        c = float(params[2])
        mid = (b + c) / 2.0
        cpu_scale.set(mid)
        cpu_var.set(f"{mid:.1f}")
        evaluar()
    except Exception:
        pass

cpu_level_cb.bind('<<ComboboxSelected>>', on_cpu_level_select)

# Tiempo: slider + entry
ttk.Label(left, text='Tiempo (ms)').pack(anchor='w', pady=(10,0))
time_var = tk.StringVar(value='380.0')
time_scale = ttk.Scale(left, from_=0, to=1000, orient='horizontal')
time_scale.set(380.0)
time_scale.pack(fill=tk.X)

def on_time_scale(e):
    val = time_scale.get()
    time_var.set(f"{val:.0f}")

time_scale.bind('<B1-Motion>', on_time_scale)
time_scale.bind('<ButtonRelease-1>', on_time_scale)

time_entry = ttk.Entry(left, textvariable=time_var, width=10)
time_entry.pack(anchor='w', pady=(4,0))
time_entry.bind('<Return>', lambda e: time_scale.set(float(time_var.get())))

# Botones
btns = ttk.Frame(left)
btns.pack(fill=tk.X, pady=(12,0))
ttk.Button(btns, text='Evaluar', command=evaluar).pack(side=tk.LEFT, expand=True, fill=tk.X)
ttk.Button(btns, text='Ejemplo', command=set_example).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6,0))
ttk.Button(btns, text='Salir', command=root.destroy).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6,0))

ttk.Label(left, text='Resultado (defuzzificado):', padding=(0,10,0,0)).pack(anchor='w', pady=(12,0))
result_var = tk.StringVar(value='0.00 %')
ttk.Label(left, textvariable=result_var, font=('Segoe UI', 12, 'bold')).pack(anchor='w')

ttk.Label(left, text='Reglas activadas (más importantes):', padding=(0,10,0,0)).pack(anchor='w', pady=(12,0))
rules_frame = ttk.Frame(left)
rules_frame.pack(fill=tk.BOTH, expand=True)
rules_listbox = tk.Listbox(rules_frame, height=10)
rules_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
rules_scroll = ttk.Scrollbar(rules_frame, orient='vertical', command=rules_listbox.yview)
rules_scroll.pack(side=tk.RIGHT, fill=tk.Y)
rules_listbox.config(yscrollcommand=rules_scroll.set)


right = ttk.Frame(main)
right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Figura con 3 gráficos arriba (CPU, Tiempo, Ajuste) y uno grande abajo (Agregado)
fig = plt.figure(figsize=(9,6))
gs = fig.add_gridspec(2, 3)
ax_cpu = fig.add_subplot(gs[0, 0])
ax_time = fig.add_subplot(gs[0, 1])
ax_ajuste = fig.add_subplot(gs[0, 2])
ax_main = fig.add_subplot(gs[1, :])
canvas = FigureCanvasTkAgg(fig, master=right)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Panel de interpretación / explicación (se muestra debajo del gráfico)
explain_frame = ttk.LabelFrame(right, text='Interpretación / Ayuda', padding=8)
explain_frame.pack(fill=tk.BOTH, expand=False, pady=(8,0))
explain_text = tk.Text(explain_frame, height=8, wrap='word')
explain_scroll = ttk.Scrollbar(explain_frame, orient='vertical', command=explain_text.yview)
explain_text.configure(yscrollcommand=explain_scroll.set)
explain_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
explain_scroll.pack(side=tk.RIGHT, fill=tk.Y)
explain_text.configure(state='disabled')

# Inicializar con el ejemplo
set_example()

root.bind('<Return>', evaluar)
root.mainloop()
