import os
import scipy.io
import numpy as np

# Ruta al directorio de los archivos .mat
data_dir = "MEDICIONES/Zmesurees_correctes"

# Lista de archivos .mat en el directorio
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

set2_quenas = {}

for fname in sorted(mat_files):
    full_path = os.path.join(data_dir, fname)
    mat_data = scipy.io.loadmat(full_path)
    base_name = os.path.splitext(fname)[0]

    y_var_name = next((key for key in mat_data.keys() if 'Ymes_emb_ad' in key), None)
    fq_var_name = next((key for key in mat_data.keys() if 'FQ_mes' in key), None)

    if y_var_name is None or fq_var_name is None:
        continue

    y_data = mat_data[y_var_name]
    fq_data = np.squeeze(mat_data[fq_var_name])

    quena_id = base_name.split('_')[1]
    position = base_name.split('_')[2]

    if quena_id not in set2_quenas:
        set2_quenas[quena_id] = {}

    set2_quenas[quena_id][position] = {
        'frecuencia': fq_data,
        'admitancia': y_data
    }