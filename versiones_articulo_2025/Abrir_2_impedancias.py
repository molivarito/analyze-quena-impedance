import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def read_and_format_data(filepath):
    data = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line_mod = line.replace(',', '.').replace(';', ' ').strip()
            if not line_mod:
                continue
            parts = line_mod.split()
            if len(parts) < 2:
                continue
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                data.append((x_val, y_val))
            except ValueError:
                continue
        if data:
            return np.array(data)
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Error leyendo {filepath}: {e}")
        return None

note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']

# ---------------- SET 1: DAT FILES --------------------
set1_dir = "MEDICIONES"
set1_quenas = [d for d in os.listdir(set1_dir) if d.startswith("Quena") and os.path.isdir(os.path.join(set1_dir, d))]
set1_quenas.sort()  # Para asegurar Quena1, Quena2, ...

for quena_name in set1_quenas:
    print(f"\n[INFO] Mostrando datos de {quena_name} (Set 1)")
    plt.figure(figsize=(14, 12))
    for i, note in enumerate(note_order):
        plt.subplot(4, 2, i + 1)
        dat_path = os.path.join(set1_dir, quena_name, f"{quena_name}_{note}.dat")
        data = read_and_format_data(dat_path)
        if data is not None:
            fq = data[:, 0]
            Z = data[:, 1]
            Y = np.where(Z != 0, 1 / Z, np.nan)
            plt.plot(fq, 20 * np.log10(np.abs(Y)), color='blue', label="Admitancia (Set 1)")
        plt.title(f"{note}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Admitancia (dB)")
        plt.grid(True)
        plt.legend(fontsize=8)
    plt.suptitle(f"Curvas de Admitancia - {quena_name} (Set 1)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    # plt.savefig(f"imagenes/Set1/admitancia_{quena_name}.png", dpi=200)

# --------------- SET 2: MAT FILES ---------------------
set2_dir = "MEDICIONES/Zmesurees_correctes"
quena_nums = sorted(list({int(f.split('_')[1]) for f in os.listdir(set2_dir) if f.startswith('quena_') and f.endswith('.mat')}))
for qnum in quena_nums:
    mat1_path = os.path.join(set2_dir, f"quena_{qnum}_1.mat")
    mat2_path = os.path.join(set2_dir, f"quena_{qnum}_2.mat")
    if not os.path.exists(mat1_path) or not os.path.exists(mat2_path):
        continue
    mat1 = scipy.io.loadmat(mat1_path)
    mat2 = scipy.io.loadmat(mat2_path)
    y_var1 = next(k for k in mat1.keys() if k.startswith('Ymes_emb_ad'))
    y_var2 = next(k for k in mat2.keys() if k.startswith('Ymes_emb_ad'))
    Y1 = mat1[y_var1]
    Y2 = mat2[y_var2]
    fq = mat1['fb'].squeeze()  # Se asume igual en ambos archivos
    plt.figure(figsize=(14, 12))
    for i, note in enumerate(note_order):
        plt.subplot(4, 2, i + 1)
        y1 = Y1[i, :]
        y2 = Y2[i, :]
        # Mostramos ambos archivos de la misma quena con distintos estilos
        plt.plot(fq, 20 * np.log10(np.abs(y1)), label='Admitancia archivo 1', color='orange', linestyle='-')
        plt.plot(fq, 20 * np.log10(np.abs(y2)), label='Admitancia archivo 2', color='orange', linestyle='--')
        plt.title(f"{note}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Admitancia (dB)")
        plt.grid(True)
        plt.legend(fontsize=8)
    plt.suptitle(f"Curvas de Admitancia - quena_{qnum} (Set 2)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    # plt.savefig(f"imagenes/Set2/admitancia_quena{qnum}.png", dpi=200)