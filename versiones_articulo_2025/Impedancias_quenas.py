import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict
import io
import scipy.io

# --- Configuration Constants ---
QUENA_COLORS = {
    "Quena1": "tab:blue", "Quena2": "tab:orange", "Quena3": "tab:green", "Quena4": "tab:red",
    "quena_1": "tab:purple", "quena_2": "tab:brown", "quena_3": "tab:pink",
    "quena_4": "cyan",       
    "quena_5": "magenta",    
    "quena_6": "olive",      
}
TEMPERED_FREQUENCIES = {
    'G': 391.995, 'A': 440.000, 'B': 493.883, 'C': 523.251,
    'D': 587.330, 'E': 659.255, 'F#': 739.989, 'G2': 783.991,
}
PLOT_NOTE_ORDER_FULL = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
PLOT_NOTE_ORDER_INHARM_DEV = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
PLOT_FREQ_RANGE = (200, 3000) 
PEAK_FIND_FREQ_RANGE = (200, 3000) 

# Constants adjusted based on your version
MAX_ADMITTANCE_FOR_PLOT_DISPLAY = 100.0 
Y_LIM_LOWER_DEFAULT = 1e-4
Y_LIM_UPPER_DEFAULT = 100.0 
Y_LIM_GLOBAL_CEILING = 20.0 

peaks_processed_data = {}

# --- Helper Functions ---
def normalize_note_name(note):
    return 'F#' if note == 'F' else note

def read_dat_file(filepath):
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        processed_lines = [line.replace(',', '.').strip().replace(';', ' ') for line in lines]
        valid_lines = [line for line in processed_lines if line.strip() and len(line.split()) == 2]
        if not valid_lines: return np.array([])
        return np.loadtxt(io.StringIO("\n".join(valid_lines)))
    except Exception as e:
        print(f"Error reading .dat file {filepath}: {e}")
        return np.array([])

def find_peaks_in_admittance(frecuencia_filtered, admittance_filtered, n_peaks=2,
                                 savgol_window=31, savgol_poly=3, prominence=2, height_thresh=-40):
    admittance_filtered = np.abs(np.real_if_close(admittance_filtered))
    admittance_filtered_positive = np.where(admittance_filtered > 1e-12, admittance_filtered, 1e-12)
    admittance_log = 20 * np.log10(admittance_filtered_positive)
    if len(admittance_log) < 5: return np.array([]), np.array([])
    if savgol_window % 2 == 0: savgol_window += 1
    if savgol_window > len(admittance_log):
        savgol_window = len(admittance_log) - (len(admittance_log) % 2) - 1
        if savgol_window < 3: savgol_window = 3
    if savgol_poly >= savgol_window: savgol_poly = savgol_window - 1
    if savgol_poly < 1: savgol_poly = 1
    admittance_smoothed_log = savgol_filter(admittance_log, savgol_window, savgol_poly) if len(admittance_log) >= savgol_window else admittance_log
    peaks, _ = find_peaks(admittance_smoothed_log, prominence=prominence)
    if len(peaks) == 0: return np.array([]), np.array([])
    height_mask = admittance_smoothed_log[peaks] >= height_thresh
    detected_freqs = frecuencia_filtered[peaks][height_mask]
    detected_admittances = admittance_filtered[peaks][height_mask]
    if len(detected_freqs) == 0: return np.array([]), np.array([])
    sorted_indices_by_frequency = np.argsort(detected_freqs)
    return detected_freqs[sorted_indices_by_frequency][:n_peaks], detected_admittances[sorted_indices_by_frequency][:n_peaks]

def calculate_intonation_metrics(quena_name, note, freqs, all_peaks_data):
    result = {
        'f0': np.nan, 'f1_own': np.nan,
        'dev1': np.nan, 'dev2_own': np.nan, 'dev2_g2_ref': np.nan,
        'inharm_own': np.nan, 'inharm_g2_ref': np.nan
    }
    processed_note = normalize_note_name(note)
    if len(freqs) >= 1: result['f0'] = freqs[0]
    if len(freqs) >= 2: result['f1_own'] = freqs[1]
    if not np.isnan(result['f0']) and processed_note in TEMPERED_FREQUENCIES:
        ref1 = TEMPERED_FREQUENCIES[processed_note]
        if ref1 > 0: result['dev1'] = 1200 * np.log2(result['f0'] / ref1)
        if not np.isnan(result['f1_own']):
            ref2_own = 2 * ref1
            if ref2_own > 0: result['dev2_own'] = 1200 * np.log2(result['f1_own'] / ref2_own)
    if not np.isnan(result['f0']) and not np.isnan(result['f1_own']) and result['f0'] > 0 and result['f1_own'] > 0 :
        result['inharm_own'] = 1200 * np.log2(result['f1_own'] / (2 * result['f0']))
    if processed_note == 'G':
        g2_key = f"{quena_name}_G2"
        if g2_key in all_peaks_data and len(all_peaks_data[g2_key]['freqs']) >= 1:
            g2_f0_for_g_ref = all_peaks_data[g2_key]['freqs'][0]
            if not np.isnan(result['f0']) and 'G' in TEMPERED_FREQUENCIES:
                 ref1_g = TEMPERED_FREQUENCIES['G']
                 ref2_g_octave = 2 * ref1_g
                 if ref2_g_octave > 0:
                     result['dev2_g2_ref'] = 1200 * np.log2(g2_f0_for_g_ref / ref2_g_octave)
            if not np.isnan(result['f0']) and result['f0'] > 0 and g2_f0_for_g_ref > 0:
                result['inharm_g2_ref'] = 1200 * np.log2(g2_f0_for_g_ref / (2 * result['f0']))
    for k in ['dev1', 'dev2_own', 'dev2_g2_ref', 'inharm_own', 'inharm_g2_ref']:
        if k in result and not np.isfinite(result.get(k, np.nan)): result[k] = np.nan
    return result

# --- Data Loading and Processing ---
def load_and_process_quena_data(base_dirs, selected_quenas):
    global peaks_processed_data
    peaks_processed_data = {}
    set1_dir = base_dirs.get("set1_dir")
    if set1_dir and os.path.exists(set1_dir):
        print(f"\n--- Processing Set 1 (.dat files) from {set1_dir} ---")
        for quena_folder_name in sorted(os.listdir(set1_dir)):
            if not (quena_folder_name.startswith("Quena") and os.path.isdir(os.path.join(set1_dir, quena_folder_name))): continue
            if quena_folder_name not in selected_quenas: continue
            data_directory = os.path.join(set1_dir, quena_folder_name)
            for note_ref in PLOT_NOTE_ORDER_FULL:
                filepath = None
                if note_ref == 'F#':
                    path_fsharp = os.path.join(data_directory, f"{quena_folder_name}_F#.dat")
                    path_f = os.path.join(data_directory, f"{quena_folder_name}_F.dat")
                    if os.path.exists(path_fsharp): filepath = path_fsharp
                    elif os.path.exists(path_f): filepath = path_f
                else:
                    path_std = os.path.join(data_directory, f"{quena_folder_name}_{note_ref}.dat")
                    if os.path.exists(path_std): filepath = path_std
                if not filepath: continue 
                data = read_dat_file(filepath)
                if data.ndim != 2 or data.shape[1] != 2 or data.size == 0: continue
                key = f"{quena_folder_name}_{normalize_note_name(note_ref)}"
                impedance_values = data[:, 1]
                # Using your updated logic for admittance calculation for Set 1
                admittance = np.where(np.abs(impedance_values) > 1e-12, 1 / impedance_values, 1e-12) 
                peaks_processed_data[key] = {'original_admittance_curve_freq': data[:, 0], 'original_admittance_curve_val': admittance}

    set2_dir = base_dirs.get("set2_dir")
    if set2_dir and os.path.exists(set2_dir):
        print(f"\n--- Processing Set 2 (.mat files) from {set2_dir} ---")
        quena_nums = sorted(list({int(f.split('_')[1]) for f in os.listdir(set2_dir) if f.startswith('quena_') and f.endswith('.mat')}))
        for qnum in quena_nums:
            quena_name = f"quena_{qnum}"
            if quena_name not in selected_quenas: continue
            mat1_path, mat2_path = os.path.join(set2_dir, f"quena_{qnum}_1.mat"), os.path.join(set2_dir, f"quena_{qnum}_2.mat")
            if not os.path.exists(mat1_path) and not os.path.exists(mat2_path): continue
            combined_admittance_per_note = defaultdict(list)
            for mat_path in [mat1_path, mat2_path]:
                if os.path.exists(mat_path):
                    try:
                        mat_data = scipy.io.loadmat(mat_path)
                        y_var = next(k for k in mat_data.keys() if k.startswith('Ymes_emb_ad'))
                        Y_matrix, frequency_mat = mat_data[y_var], mat_data['fb'].squeeze()
                        for i, note_ref in enumerate(PLOT_NOTE_ORDER_FULL):
                            if i < Y_matrix.shape[0]:
                                combined_admittance_per_note[normalize_note_name(note_ref)].append((frequency_mat, np.abs(Y_matrix[i, :])))
                    except Exception as e: print(f"    Error reading .mat file {mat_path}: {e}")
            for note_ref, data_tuples in combined_admittance_per_note.items():
                if data_tuples: peaks_processed_data[f"{quena_name}_{normalize_note_name(note_ref)}"] = {'original_admittance_curve_data': data_tuples}
    
    print("\n--- Finding peaks and calculating intonation metrics ---")
    for key in list(peaks_processed_data.keys()):
        freq_peak, adm_peak = None, None
        if 'original_admittance_curve_freq' in peaks_processed_data[key]:
            freq_peak, adm_peak = peaks_processed_data[key]['original_admittance_curve_freq'], peaks_processed_data[key]['original_admittance_curve_val']
        elif 'original_admittance_curve_data' in peaks_processed_data[key] and peaks_processed_data[key]['original_admittance_curve_data']:
            freq_peak, adm_peak = peaks_processed_data[key]['original_admittance_curve_data'][0]
        if freq_peak is not None and adm_peak is not None and freq_peak.size > 0:
            mask = (freq_peak >= PEAK_FIND_FREQ_RANGE[0]) & (freq_peak <= PEAK_FIND_FREQ_RANGE[1])
            if np.any(mask):
                peaks_processed_data[key]['freqs'], _ = find_peaks_in_admittance(freq_peak[mask], adm_peak[mask]) 
            else: peaks_processed_data[key]['freqs'] = np.array([])
        else: peaks_processed_data[key]['freqs'] = np.array([])
        peaks_processed_data[key]['metrics'] = {}
    print("\n--- Calculating intonation metrics (second pass) ---")
    for key in list(peaks_processed_data.keys()):
        quena_name, processed_note = key.rsplit('_', 1)
        peaks_processed_data[key]['metrics'] = calculate_intonation_metrics(quena_name, processed_note, peaks_processed_data[key]['freqs'], peaks_processed_data)
    print("\nData loading and initial processing complete.")

# --- Plotting Functions ---

def plot_individual_admittance_curves(quena_name, peaks_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(1, len(PLOT_NOTE_ORDER_FULL), figsize=(3.5 * len(PLOT_NOTE_ORDER_FULL), 7), sharey=True) 
    if len(PLOT_NOTE_ORDER_FULL) == 1: axs = [axs]
    note_to_ax = dict(zip(PLOT_NOTE_ORDER_FULL, axs))
    
    all_valid_adm_vals_for_figure = [] # Collect all valid y-values for this figure's ylim

    for note_ref in PLOT_NOTE_ORDER_FULL:
        key = f"{quena_name}_{normalize_note_name(note_ref)}"
        ax = note_to_ax[note_ref]
        ax.set_title(note_ref, fontsize=14); ax.set_xlim(*PLOT_FREQ_RANGE); ax.grid(True, alpha=0.6)
        
        plotted_curve_on_ax = False
        if key in peaks_data:
            entry = peaks_data[key]
            curves = entry.get('original_admittance_curve_data', []) or [(entry.get('original_admittance_curve_freq'), entry.get('original_admittance_curve_val'))]
            for freq, adm in curves:
                if freq is not None and adm is not None and freq.size > 0 and np.any(np.isfinite(adm)):
                    adm_abs = np.abs(np.real_if_close(adm))
                    adm_display = np.clip(adm_abs, 1e-12, MAX_ADMITTANCE_FOR_PLOT_DISPLAY) # Use your clip for display
                    ax.plot(freq, adm_display, lw=1.5, alpha=0.8, color=QUENA_COLORS.get(quena_name, 'gray'))
                    plotted_curve_on_ax = True
                    
                    # Collect original (unclipped by MAX_ADMITTANCE_FOR_PLOT_DISPLAY) positive values for ylim calculation
                    freq_mask = (freq >= PLOT_FREQ_RANGE[0]) & (freq <= PLOT_FREQ_RANGE[1])
                    adm_in_x_view = adm_abs[freq_mask]
                    valid_adm_in_x_view = adm_in_x_view[np.isfinite(adm_in_x_view) & (adm_in_x_view > 0)] # strictly positive
                    if valid_adm_in_x_view.size > 0:
                        all_valid_adm_vals_for_figure.extend(valid_adm_in_x_view)
            
            if plotted_curve_on_ax:
                for f_peak in entry.get('freqs', []): ax.axvline(x=f_peak, color='red', ls='--', alpha=0.8)
            else: 
                ax.text(0.5, 0.5, 'No Curve Data', ha='center', va='center', transform=ax.transAxes)
        else: 
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', transform=ax.transAxes)
        ax.tick_params(axis='x', labelsize=10, rotation=45); ax.tick_params(axis='y', labelsize=10)

    # Determine and set Y-axis limits based on your provided logic
    if all_valid_adm_vals_for_figure:
        min_adm = max(np.min(all_valid_adm_vals_for_figure) / 1.5, Y_LIM_LOWER_DEFAULT)
        max_adm = min(np.max(all_valid_adm_vals_for_figure) * 1.5, Y_LIM_GLOBAL_CEILING)
        if max_adm <= min_adm: # Ensure there's a range
            max_adm = min(min_adm * 100, Y_LIM_GLOBAL_CEILING) # Cap if it becomes too large
        if max_adm <= min_adm: # If still invalid (e.g. min_adm is already Y_LIM_GLOBAL_CEILING)
             max_adm = min_adm * 10 # Default small range above min_adm
        axs[0].set_ylim(min_adm, max_adm)
    else:
        axs[0].set_ylim(Y_LIM_LOWER_DEFAULT, Y_LIM_UPPER_DEFAULT)
    axs[0].set_yscale("log")

    axs[0].set_ylabel("Admittance (Siemens)", fontsize=12)
    fig.suptitle(f"Admittance Curves - {quena_name}", fontsize=16); plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(os.path.join(output_dir, f"admittance_{quena_name}.png"), dpi=300); plt.close(fig)

def plot_admittance_summary(peaks_data, selected_quenas, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plot_note_order = PLOT_NOTE_ORDER_FULL
    fig, axs = plt.subplots(1, len(plot_note_order), figsize=(3.5 * len(plot_note_order), 7), sharey=True)
    if len(plot_note_order) == 1: axs = [axs]
    note_to_ax_map = dict(zip(plot_note_order, axs)); 
    all_valid_adm_vals_for_summary = []

    for plot_note in plot_note_order:
        ax = note_to_ax_map[plot_note]; ax.set_xlim(*PLOT_FREQ_RANGE); ax.set_title(plot_note)
        for quena in sorted(selected_quenas):
            key = f"{quena}_{normalize_note_name(plot_note)}"
            if key in peaks_data:
                entry = peaks_data[key]
                curves = entry.get('original_admittance_curve_data', []) or [(entry.get('original_admittance_curve_freq'), entry.get('original_admittance_curve_val'))]
                for freq, adm in curves:
                    if freq is not None and adm is not None:
                        adm_abs = np.abs(np.real_if_close(adm))
                        adm_display = np.clip(adm_abs, 1e-12, MAX_ADMITTANCE_FOR_PLOT_DISPLAY)
                        ax.plot(freq, adm_display, color=QUENA_COLORS.get(quena, 'gray'), alpha=0.8, lw=1)
                        
                        freq_mask = (freq >= PLOT_FREQ_RANGE[0]) & (freq <= PLOT_FREQ_RANGE[1])
                        adm_in_x_view = adm_abs[freq_mask]
                        valid_adm_in_x_view = adm_in_x_view[np.isfinite(adm_in_x_view) & (adm_in_x_view > 0)]
                        if valid_adm_in_x_view.size > 0:
                            all_valid_adm_vals_for_summary.extend(valid_adm_in_x_view)
        ax.set_yscale("log"); ax.grid(True, alpha=0.6); ax.tick_params(axis='x', rotation=45)
    
    if all_valid_adm_vals_for_summary:
        min_adm = max(np.min(all_valid_adm_vals_for_summary) / 1.5, Y_LIM_LOWER_DEFAULT)
        max_adm = min(np.max(all_valid_adm_vals_for_summary) * 1.5, Y_LIM_GLOBAL_CEILING)
        if max_adm <= min_adm: 
            max_adm = min(min_adm * 100, Y_LIM_GLOBAL_CEILING)
        if max_adm <= min_adm:
             max_adm = min_adm * 10
        axs[0].set_ylim(min_adm, max_adm)
    else: 
        axs[0].set_ylim(Y_LIM_LOWER_DEFAULT, Y_LIM_UPPER_DEFAULT)
        
    handles = [plt.Line2D([0], [0], color=QUENA_COLORS[q], lw=2, label=q) for q in sorted(selected_quenas) if q in QUENA_COLORS]
    fig.legend(handles=handles, loc="upper right", title="Quena"); plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(os.path.join(output_dir, "admittance_summary.png"), dpi=300); plt.close(fig)

# --- (All other 8 plotting functions remain the same as the last complete code version) ---
# --- Make sure to copy them here in your actual script for completeness      ---
# plot_peaks_by_quena, plot_fundamental_deviation, plot_deviation_summary, 
# plot_deviation_mean_std, plot_inharmonicity, plot_inharmonicity_summary,
# plot_inharmonicity_mean_std, plot_deviation_and_inharmonicity

# --- Main execution block ---
if __name__ == "__main__":
    base_directories = {"set1_dir": "MEDICIONES", "set2_dir": "MEDICIONES/Zmesurees_correctes"}
    all_selected_quenas = [
        "Quena1", "Quena2", "Quena3", "Quena4", 
        "quena_1", "quena_2", "quena_3", "quena_4", "quena_5", "quena_6"
    ] 
    set1_quenas = [q for q in all_selected_quenas if q.startswith("Quena")]
    set2_quenas = [q for q in all_selected_quenas if q.startswith("quena_")]
    set1_output_dir = "images/Set1"
    set2_output_dir = "images/Set2"

    load_and_process_quena_data(base_directories, all_selected_quenas)

    common_plot_args = {'peaks_data': peaks_processed_data}

    def run_plots(quenas_set, output_dir_set):
        if quenas_set:
            print(f"\n--- Generating plots for: {', '.join(quenas_set)} in {output_dir_set} ---")
            for quena in quenas_set: 
                plot_individual_admittance_curves(quena, **common_plot_args, output_dir=output_dir_set)
            
            # --- Make sure to uncomment these and include the functions if you want them ---
            # plot_peaks_by_quena(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_fundamental_deviation(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            plot_admittance_summary(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_inharmonicity(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_deviation_and_inharmonicity(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_inharmonicity_summary(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_deviation_summary(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_inharmonicity_mean_std(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)
            # plot_deviation_mean_std(**common_plot_args, selected_quenas=quenas_set, output_dir=output_dir_set)

    run_plots(set1_quenas, set1_output_dir)
    run_plots(set2_quenas, set2_output_dir)

    print("\nAnalysis and plotting process completed.")