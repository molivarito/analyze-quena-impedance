import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict
import io

# Global dictionary to store peaks from all quenas
peaks_global = {}

def read_and_format_data(filepath):
    """
    Reads a data file with ';' delimiter and comma as decimal separator,
    and returns the data as a NumPy array.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Replace commas with periods and semicolons with spaces
        processed_lines = [line.replace(',', '.').strip().replace(';', ' ') for line in lines]
        
        # Filter empty or invalid lines before loading
        valid_lines = [line for line in processed_lines if line.strip() and len(line.split()) == 2]
        
        if not valid_lines:
            print(f"Warning: No valid data found in {filepath}.")
            return np.array([])

        # Use io.StringIO so np.loadtxt can read the string as a file
        data = np.loadtxt(io.StringIO("\n".join(valid_lines)))
        return data
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return np.array([])

def find_peaks_in_admittance(frecuencia_filtered, admittance_filtered, n_peaks=2):
    """
    Finds the first 'n_peaks' of admittance in frequency order
    within the already filtered data, applying smoothing and detection on a logarithmic scale.

    Args:
        frecuencia_filtered (np.array): Filtered frequency array.
        admittance_filtered (np.array): Filtered admittance values array.
        n_peaks (int): Number of peaks to find.

    Returns:
        tuple: A tuple containing (peak_frequencies, peak_admittances).
               Arrays will be empty if no peaks are found.
    """
    # Handle very small or zero admittance values before log transformation
    # A small value is added to prevent log(0) or log of negative numbers
    admittance_filtered_positive = np.where(admittance_filtered > 1e-12, admittance_filtered, 1e-12)
    admittance_log = 20 * np.log10(admittance_filtered_positive)

    min_points_for_savgol = 5
    if len(admittance_log) < min_points_for_savgol:
        admittance_smoothed_log = admittance_log
    else:
        # Ensure window_length is odd and less than data length
        window_length_savgol = 31
        polyorder_savgol = 3
        
        if window_length_savgol % 2 == 0: # Must be odd
            window_length_savgol += 1
        if window_length_savgol > len(admittance_log):
            window_length_savgol = len(admittance_log) - (len(admittance_log) % 2) - 1 # Largest possible odd number less than len
            if window_length_savgol < 3: # Ensure minimum of 3
                window_length_savgol = 3
        
        if polyorder_savgol >= window_length_savgol: # polyorder must be less than window_length
            polyorder_savgol = window_length_savgol - 1
            if polyorder_savgol < 1: # Ensure minimum of 1
                polyorder_savgol = 1
                
        admittance_smoothed_log = savgol_filter(admittance_log, window_length_savgol, polyorder_savgol)

    # Peak detection parameters:
    # prominence: How "prominent" a peak must be relative to its surroundings.
    # A value of 2 dB is a good starting point for audio/acoustic data.
    # ADJUST THIS VALUE IF TOO MUCH NOISE IS DETECTED OR IMPORTANT PEAKS ARE MISSED.
    peaks, _ = find_peaks(admittance_smoothed_log, prominence=2)
    
    if len(peaks) == 0:
        return np.array([]), np.array([])

    # Minimum height threshold in dB. Peaks below this are not considered.
    # ADJUST THIS VALUE IF THERE ARE TOO MANY UNRELEVANT LOW PEAKS OR WEAK HARMONICS ARE MISSED.
    min_height_log_threshold = -40 # Equivalent to 0.01 on a linear scale (20*log10(0.01) = -40)
    height_mask = admittance_smoothed_log[peaks] >= min_height_log_threshold
    
    detected_freqs = frecuencia_filtered[peaks][height_mask]
    detected_admittances = admittance_filtered[peaks][height_mask]

    if len(detected_freqs) == 0:
        return np.array([]), np.array([])

    # Sort peaks by frequency and take the first 'n_peaks'
    sorted_indices_by_frequency = np.argsort(detected_freqs)
    final_freqs = detected_freqs[sorted_indices_by_frequency][:n_peaks]
    final_admittances = detected_admittances[sorted_indices_by_frequency][:n_peaks]

    return final_freqs, final_admittances

def plot_impedances_from_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    # Specific note order for visualization
    note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, len(note_order), figsize=(3.5 * len(note_order), 7), sharey=True)
    # If there's only one note, axs might not be an array, ensure it is
    if len(note_order) == 1:
        axs = [axs]
    
    # Map each note to its corresponding subplot for easy access
    note_to_ax = dict(zip(note_order, axs))

    for file in files:
        # print(f"File found: {file}")
        path = os.path.join(directory, file)
        data = read_and_format_data(path)
        
        # Check if reading was successful and if data has expected shape
        if data.ndim != 2 or data.shape[1] != 2 or data.size == 0:
            print(f"Warning: File {file} does not have a 2-column format or reading failed. Skipping.")
            continue

        frequency = data[:, 0]
        impedance = data[:, 1]
        
        # Filter the frequency range of interest
        range_of_interest_mask = (frequency >= 200) & (frequency <= 3000)
        filtered_frequency_for_calculation = frequency[range_of_interest_mask]
        filtered_impedance_for_calculation = impedance[range_of_interest_mask]
        
        # Calculate admittance (1/impedance) and handle division by zero/NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            filtered_admittance_for_calculation = np.where(filtered_impedance_for_calculation != 0, 1 / filtered_impedance_for_calculation, np.nan)
        
        # Extract note from filename (e.g., Quena1_B3.dat -> B3)
        file_note = os.path.splitext(file)[0].split('_')[-1]
        # Normalize F to F# for all processing
        if file_note == 'F':
            processed_note = 'F#'
        else:
            processed_note = file_note
        
        # If the note is not in our defined order, skip it
        if processed_note not in note_order:
            continue

        # Store peaks in the global dictionary
        quena_name = os.path.basename(directory)
        key = f"{quena_name}_{processed_note}"
        peaks_global[key] = find_peaks_in_admittance(filtered_frequency_for_calculation, filtered_admittance_for_calculation)
        print(f"PEAKS_GLOBAL_KEY: {key} | peaks: {peaks_global[key][0]}")

        # Get the correct subplot for the current note
        ax = note_to_ax[processed_note]
        # Plot the admittance curve
        ax.plot(filtered_frequency_for_calculation, filtered_admittance_for_calculation, label=processed_note, linewidth=1.5, color='blue')
        
        # Plot vertical lines at detected peak frequencies
        if key in peaks_global and peaks_global[key][0].size > 0:
            for f_peak in peaks_global[key][0]:
                ax.axvline(x=f_peak, color='red', linestyle='--', alpha=0.8, linewidth=1.5) # Red color and dashed style
        
        # Subplot configuration
        ax.set_title(processed_note, fontsize=14)
        ax.set_yscale("log") # Logarithmic scale for Y-axis (admittance)
        ax.grid(True, which="both", ls="-", alpha=0.6)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # General X and Y axis configuration for all subplots
    for ax_idx, ax_obj in enumerate(axs):
        ax_obj.set_xlabel("Frequency (Hz)", fontsize=12)
        if ax_idx == 0: # Only the first subplot has the Y-axis label
           ax_obj.set_ylabel("Admittance (Siemens)", fontsize=12)
           
    # Main figure title
    fig.suptitle(f"Admittance Curves - Quena: {os.path.basename(directory)}", fontsize=16, weight='bold')
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # Adjust for main title
    os.makedirs("images", exist_ok=True) # Create directory if it doesn't exist
    output_filename = f"images/admittance_{os.path.basename(directory)}.png"
    plt.savefig(output_filename, dpi=300)
    # print(f"Plot saved to {output_filename}")
    plt.close()

def plot_peaks_by_quena(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    # Get fundamental G2 frequencies for referencing G's octave
    g2_refs_complete = {}
    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        if key.endswith('G2') and len(freqs) > 0:
            quena_name = key.split('_')[0]
            g2_refs_complete[quena_name] = freqs

    quenas_plot_data = defaultdict(list)
    for key, (freqs, admitts) in filtered_peaks_data.items(): # Use filtered_peaks_data
        quena_name, original_note = key.split('_')
        if original_note == 'F':
            processed_note = 'F#'
        else:
            processed_note = original_note
        
        # Iterate through detected peak frequencies for each note
        for i, f_val in enumerate(freqs):
            octave_str = "1st Octave" if i == 0 else "2nd Octave" # Classify by first or second octave
            quenas_plot_data[quena_name].append((processed_note, f_val, octave_str))
            
            # If it's note G and it's the first harmonic, and we have a G2 reference
            # and if the second harmonic of G is present and very close to G2 fundamental
            if processed_note == 'G' and i == 0 and quena_name in g2_refs_complete:
                if len(freqs) >= 2 and np.isclose(freqs[1], g2_refs_complete[quena_name][0], rtol=0.01): # If G's 2nd harmonic is approx G2
                    # We don't add a duplicate, it's already added as '2nd Octave' above for G
                    pass
                elif g2_refs_complete[quena_name].size > 0: # If not, but G2 exists, add as reference
                    f_g2_first = g2_refs_complete[quena_name][0]
                    quenas_plot_data[quena_name].append(("G", f_g2_first, "2nd Octave (from G2)"))
            
            # If it's note G2, and we are plotting its harmonics
            if processed_note == 'G2' and i >= 0:
                g2_harmonic_str = f"G2 Harmonic {i+1}"
                quenas_plot_data[quena_name].append(("G2", f_val, g2_harmonic_str)) # Explicitly add G2 harmonics

    os.makedirs("images", exist_ok=True)
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']

    # Generate a plot for each quena
    for quena_name in sorted(quenas_plot_data.keys()): # Iterate through sorted quena names
        quena_data = quenas_plot_data[quena_name] # Get data for current quena
        
        fig, axs = plt.subplots(1, len(plot_note_order), figsize=(3.8 * len(plot_note_order), 7), sharey=True)
        fig.suptitle(f"Admittance Peak Frequencies by Note - Quena: {quena_name}", fontsize=16, weight='bold')
        
        # Handle the case of a single subplot
        if len(plot_note_order) == 1:
            axs = [axs]
        note_to_ax_map = dict(zip(plot_note_order, axs))

        # Define colors and markers for different octave types
        color_map = {
            "1st Octave": 'tab:blue',
            "2nd Octave": 'tab:orange',
            "2nd Octave (from G2)": 'tab:purple',
            "G2 Harmonic 1": 'tab:green',
            "G2 Harmonic 2": 'tab:green',
        }
        marker_map = {
            "1st Octave": 'o',
            "2nd Octave": 'o',
            "2nd Octave (from G2)": 's',
            "G2 Harmonic 1": '^',
            "G2 Harmonic 2": '^',
        }

        # To ensure a single legend entry for each type of resonance per plot
        plotted_legend_types = set()

        for note_val, f_val, octave_str in quena_data:
            if note_val not in plot_note_order: # Only plot notes that are in our order
                continue
            
            color_val = color_map.get(octave_str, 'gray')
            marker_val = marker_map.get(octave_str, 'o')
            
            ax_target = note_to_ax_map[note_val]
            
            label_for_legend = None
            if octave_str not in plotted_legend_types:
                label_for_legend = octave_str
                plotted_legend_types.add(octave_str)
            
            ax_target.scatter([note_val], [f_val], color=color_val, s=120, edgecolors='k', marker=marker_val,
                               label=label_for_legend, zorder=3)

            # Add text label with frequency
            label_text = f"{f_val:.0f} Hz"
            ax_target.annotate(label_text, (note_val, f_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6, ec="none"))
            
            # Subplot configuration
            ax_target.set_title(note_val, fontsize=14)
            ax_target.grid(True, which="both", ls="-", alpha=0.6)
            ax_target.tick_params(axis='x', labelsize=10)
            ax_target.tick_params(axis='y', labelsize=10)

    # Axis labels configuration for all subplots
    for ax_obj in axs:
        ax_obj.set_xlabel("Note", fontsize=12)
    axs[0].set_ylabel("Frequency (Hz)", fontsize=12)

    # Create legend with unique handles
    handles, labels = [], []
    for oct_str in ["1st Octave", "2nd Octave", "2nd Octave (from G2)", "G2 Harmonic 1", "G2 Harmonic 2"]:
        if oct_str in plotted_legend_types: # Only add if that type was actually plotted
            h = plt.Line2D([0], [0], color=color_map.get(oct_str), marker=marker_map.get(oct_str), linestyle='None',
                                markersize=8, label=oct_str, markeredgecolor='k')
            handles.append(h)
            labels.append(oct_str)
    
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 0.95), title="Resonance Type",
                   fontsize=10, title_fontsize=12, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    output_filename = f"images/peak_frequencies_{quena_name}.png"
    plt.savefig(output_filename, dpi=300)
    # print(f"Plot saved to {output_filename}")
    plt.close()

def plot_inharmonicity(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#'] # Only notes that have their own harmonic octave
    quena_inharm_data = defaultdict(lambda: {'notes': [], 'deltas': [], 'markers': []})

    # Collect fundamental G2 frequencies for referencing G's octave
    g2_f0_refs = {}
    for key, (freqs, _) in filtered_peaks_data.items():
        if key.endswith('G2') and len(freqs) > 0:
            quena_name = key.split('_')[0]
            g2_f0_refs[quena_name] = freqs

    for quena_name in sorted(list(set(k.split('_')[0] for k in filtered_peaks_data.keys()))):
        # For each note in the order, always append something, even if np.nan
        for current_note in plot_note_order:
            # Normalize 'F' to 'F#' in note keys
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            note_key = f"{quena_name}_{key_note}"
            # Default values
            delta_oct = np.nan
            marker = 'o'
            # Try to get inharmonicity if data exists
            if note_key in filtered_peaks_data:
                note_freqs, _ = filtered_peaks_data[note_key]
                print(f"DEBUG-INHARM | {note_key}: {note_freqs}")
                f0, f1 = None, None
                if len(note_freqs) >= 1:
                    f0 = note_freqs[0]
                if key_note == 'G' and quena_name in g2_f0_refs:
                    # For G, use G2 as f1 if available
                    f1 = g2_f0_refs[quena_name][0]
                elif len(note_freqs) >= 2:
                    f1 = note_freqs[1]
                if f0 is not None and f1 is not None and f0 > 0 and f1 > 0:
                    delta_oct = 1200 * np.log2(f1 / (2 * f0))
                    if not np.isfinite(delta_oct):
                        delta_oct = np.nan
            # Always append (note, delta, marker) for this note
            quena_inharm_data[quena_name]['notes'].append(key_note)
            quena_inharm_data[quena_name]['deltas'].append(delta_oct)
            quena_inharm_data[quena_name]['markers'].append(marker)

        # For G2's own inharmonicity as a reference at G (triangle marker)
        g2_key = f"{quena_name}_G2"
        delta_oct_g2_own = np.nan
        if g2_key in filtered_peaks_data:
            g2_freqs, _ = filtered_peaks_data[g2_key]
            if len(g2_freqs) >= 2:
                f0_g2, f1_g2 = g2_freqs[0], g2_freqs[1]
                if f0_g2 > 0 and f1_g2 > 0:
                    delta_oct_g2_own = 1200 * np.log2(f1_g2 / (2 * f0_g2))
                    if not np.isfinite(delta_oct_g2_own):
                        delta_oct_g2_own = np.nan
        # Only for G, add an extra marker '^' for G2's inharmonicity (even if nan)
        # Check if already exists (avoid duplicate if same as own inharmonicity)
        already_has_g2 = False
        for i, (n, m, d) in enumerate(zip(quena_inharm_data[quena_name]['notes'],
                                          quena_inharm_data[quena_name]['markers'],
                                          quena_inharm_data[quena_name]['deltas'])):
            if n == 'G' and m == '^' and np.isclose(d, delta_oct_g2_own, atol=1e-6, equal_nan=True):
                already_has_g2 = True
                break
        if not already_has_g2:
            # Insert after G (i.e., after first note)
            idx_g = None
            for idx, n in enumerate(quena_inharm_data[quena_name]['notes']):
                if n == 'G':
                    idx_g = idx
                    break
            if idx_g is not None:
                quena_inharm_data[quena_name]['notes'].insert(idx_g + 1, 'G')
                quena_inharm_data[quena_name]['deltas'].insert(idx_g + 1, delta_oct_g2_own)
                quena_inharm_data[quena_name]['markers'].insert(idx_g + 1, '^')


    os.makedirs("images", exist_ok=True)
    for quena_name, data in quena_inharm_data.items():
        # DEBUGGING: Print filtered_peaks_data keys and inharmonicity values for each note
        # DEBUGGING
        print(f"\nDEBUG: Notas con datos para {quena_name}:")
        print("filtered_peaks_data keys:", list(filtered_peaks_data.keys()))
        for n, d in zip(quena_inharm_data[quena_name]['notes'], quena_inharm_data[quena_name]['deltas']):
            print(f"Nota: {n} | Inharmonicidad: {d}")
        # DEBUGGING END

        print(f"TO PLOT for {quena_name} | notes: {data['notes']} | inharmonicities: {data['deltas']}")

        plt.figure(figsize=(10, 6))
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)')

        # Map notes to numerical positions for the X-axis
        x_map = {note: i for i, note in enumerate(plot_note_order)}

        # DEBUGGING: Print the contents of the lists notes and deltas for each quena
        # (already printed above)

        # Force alignment: for each note in plot_note_order, always plot a point (even if nan)
        # If there's a triangle marker for G (G2 inharmonicity), plot it after the 'G' square.
        # We'll build a list of points to plot for each marker type ('o', '^') for each note.
        # There may be multiple 'G' points (own and triangle)
        # Build a list of (note, delta, marker) tuples for this quena
        note_marker_tuples = list(zip(data['notes'], data['deltas'], data['markers']))
        # For each note in plot_note_order, plot all matching markers (could be more than one e.g. for G)
        has_plotted_own_inarm_label = False
        has_plotted_g2_ref_inarm_label = False
        for i_note, note in enumerate(plot_note_order):
            # Find all tuples for this note
            for n, d, m in note_marker_tuples:
                if n == note:
                    x_pos = x_map[note]
                    current_label_for_point = None
                    if m == 'o' and not has_plotted_own_inarm_label:
                        current_label_for_point = 'Own Inharmonicity'
                        has_plotted_own_inarm_label = True
                    elif m == '^' and not has_plotted_g2_ref_inarm_label:
                        current_label_for_point = 'G2 Inharmonicity (referenced at G)'
                        has_plotted_g2_ref_inarm_label = True
                    plt.scatter(x_pos, d, color='tab:blue', marker=m, s=120, edgecolors='k', zorder=3,
                                label=current_label_for_point)
        # For notes with no data (not present in data['notes']), plot nan (no marker)
        # (Optional: not strictly needed since nothing will be plotted for missing notes)

        plt.xticks(ticks=list(x_map.values()), labels=list(x_map.keys()), fontsize=11)
        plt.yticks(fontsize=10)
        plt.title(f"Octave Inharmonicity - Quena: {quena_name}", fontsize=15, weight='bold')
        plt.xlabel("Note", fontsize=12)
        plt.ylabel("Inharmonicity (cents)", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.6)

        # Dynamically adjust Y-axis limits
        all_deltas = [d for d in data['deltas'] if np.isfinite(d)]
        if all_deltas:
            min_delta = np.min(all_deltas)
            max_delta = np.max(all_deltas)
            padding = (max_delta - min_delta) * 0.1
            plt.ylim(min_delta - padding, max_delta + padding)
        else:
            plt.ylim(-50, 50)

        # Reconstruct legend with unique handles
        handles, labels = [], []
        # Ensure the 0 cents line is in the legend and first
        line_0_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)')
        handles.append(line_0_handle)
        labels.append('Ideal Inharmonicity (0 cents)')
        # Add other handles and labels uniquely and in order
        order = ['Own Inharmonicity', 'G2 Inharmonicity (referenced at G)']
        for l_ord in order:
            for h, l_curr in zip(plt.gca().get_legend_handles_labels()[0], plt.gca().get_legend_handles_labels()[1]):
                if l_curr == l_ord and l_curr not in labels:
                    handles.append(h)
                    labels.append(l_curr)
        plt.legend(handles, labels, loc='best', fontsize=10)
        plt.tight_layout()
        output_filename = f"images/inharmonicity_{quena_name}.png"
        plt.savefig(output_filename, dpi=300)
        plt.close()

def plot_fundamental_deviation(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    # Tempered frequencies reference
    tempered_notes = {
        'G': 391.995, 'A': 440.000, 'B': 493.883, 'C': 523.251,
        'D': 587.330, 'E': 659.255, 'F#': 739.989, 'G2': 783.991,
    }
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    quenas_data = defaultdict(lambda: {'note': [], 'delta1': [], 'delta2': []})

    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        quena_name, original_note = key.split('_')
        if original_note == 'F':
            processed_note = 'F#'
        else:
            processed_note = original_note
        # If the note is not in our tempered map or has fewer than 2 peaks, skip it
        if processed_note not in tempered_notes or len(freqs) < 2:
            continue
        f0, f1 = freqs[0], freqs[1] # First and second peak
        ref1 = tempered_notes[processed_note] # Tempered frequency for the fundamental
        ref2 = 2 * ref1 # Reference for the second octave (harmonic)
        # Calculate deviation in cents
        delta1 = 1200 * np.log2(f0 / ref1) if f0 > 0 and ref1 > 0 else np.nan
        delta2 = 1200 * np.log2(f1 / ref2) if f1 > 0 and ref2 > 0 else np.nan
        quenas_data[quena_name]['note'].append(processed_note)
        quenas_data[quena_name]['delta1'].append(delta1)
        quenas_data[quena_name]['delta2'].append(delta2)


    os.makedirs("images", exist_ok=True)
    for quena_name in sorted(quenas_data.keys()): # Iterate through sorted quena names
        data = quenas_data[quena_name] # Get data for current quena
        if not data['note']: continue

        # Order data by predefined note order
        # Create a map of current quena data indexed by note for easy access
        current_data_map = {n: {'delta1': d1, 'delta2': d2} for n, d1, d2 in zip(data['note'], data['delta1'], data['delta2'])}
        
        plot_notes = []
        plot_delta1 = []
        plot_delta2 = []

        for note_ref in plot_note_order:
            if note_ref in current_data_map:
                plot_notes.append(note_ref)
                plot_delta1.append(current_data_map[note_ref]['delta1'])
                plot_delta2.append(current_data_map[note_ref]['delta2'])
            else: # Add NaNs if note is missing to maintain consistent X-axis
                plot_notes.append(note_ref)
                plot_delta1.append(np.nan)
                plot_delta2.append(np.nan)

        plt.figure(figsize=(10, 6))
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)')

        # Plot deviations
        plt.plot(plot_notes, plot_delta1, color='tab:blue', linestyle='--', marker='o', markersize=8, label='1st Octave',
                 linewidth=1, zorder=3)
        plt.plot(plot_notes, plot_delta2, color='tab:orange', linestyle=':', marker='s', markersize=8, label='2nd Octave',
                 linewidth=1, zorder=3)
        
        plt.title(f"Resonance Deviation from Tempered Note - Quena: {quena_name}", fontsize=15, weight='bold')
        plt.xlabel("Note", fontsize=12)
        plt.ylabel("Deviation (cents)", fontsize=12)
        
        # Adjust Y-axis limits
        all_deltas = [d for d in plot_delta1 + plot_delta2 if np.isfinite(d)]
        if all_deltas:
            min_delta = np.min(all_deltas)
            max_delta = np.max(all_deltas)
            padding = (max_delta - min_delta) * 0.1
            plt.ylim(min_delta - padding, max_delta + padding)
        else:
             plt.ylim(-50, 50)
        
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="-", alpha=0.6)
        plt.xticks(plot_note_order, fontsize=11)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        output_filename = f"images/fundamental_deviation_{quena_name}.png"
        plt.savefig(output_filename, dpi=300)
        # print(f"Deviation plot saved to {output_filename}")
        plt.close()

def plot_inharmonicity_summary(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    summary_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#'] # Only notes that have their own octave
    summary_plot_data = defaultdict(lambda: {'notes': [], 'deltas': [], 'markers': []})

    quena_colors = {
        "Quena1": "tab:blue", "Quena2": "tab:orange",
        "Quena3": "tab:green", "Quena4": "tab:red",
    }

    # Collect fundamental G2 frequencies for each quena
    g2_f0_refs = {}
    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        if key.endswith("G2") and len(freqs) >= 1:
            quena_name = key.split('_')[0]
            g2_f0_refs[quena_name] = freqs[0]

    for quena_name in sorted(list(set(k.split('_')[0] for k in filtered_peaks_data.keys()))): # Iterate over selected quenas
        if quena_name not in selected_quenas: # Explicitly check
            continue

        # Use a temporary map to ensure each note has only one "own inharmonicity" point
        # and one "G2 inharmonicity" point, if both apply.
        temp_data_for_quena = defaultdict(lambda: {'own_delta': None, 'g2_ref_delta': None})

        for current_note in summary_note_order:
            # Normalize 'F' to 'F#'
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            note_key = f"{quena_name}_{key_note}"
            if note_key in filtered_peaks_data: # Use filtered_peaks_data
                note_freqs, _ = filtered_peaks_data[note_key] # Use filtered_peaks_data
                f0, f1 = None, None
                if len(note_freqs) >= 1: f0 = note_freqs[0]
                if key_note == 'G' and quena_name in g2_f0_refs:
                    f1 = g2_f0_refs[quena_name] # Use G2 as f1 for G
                elif len(note_freqs) >= 2:
                    f1 = note_freqs[1]
                if f0 is not None and f1 is not None and f0 > 0 and f1 > 0:
                    delta_oct = 1200 * np.log2(f1 / (2 * f0))
                    if np.isfinite(delta_oct):
                        temp_data_for_quena[key_note]['own_delta'] = delta_oct
        
        # If G2 is present, calculate its own inharmonicity and add it as a reference for note G
        g2_key = f"{quena_name}_G2"
        if g2_key in filtered_peaks_data: # Use filtered_peaks_data
            g2_freqs, _ = filtered_peaks_data[g2_key] # Use filtered_peaks_data
            if len(g2_freqs) >= 2:
                f0_g2, f1_g2 = g2_freqs[0], g2_freqs[1]
                if f0_g2 > 0 and f1_g2 > 0:
                    delta_oct_g2_own = 1200 * np.log2(f1_g2 / (2 * f0_g2))
                    if np.isfinite(delta_oct_g2_own):
                        temp_data_for_quena['G']['g2_ref_delta'] = delta_oct_g2_own

        # After collecting all data for the quena, add it to summary_plot_data
        for current_note in summary_note_order:
            if temp_data_for_quena[current_note]['own_delta'] is not None:
                summary_plot_data[quena_name]['notes'].append(current_note)
                summary_plot_data[quena_name]['deltas'].append(temp_data_for_quena[current_note]['own_delta'])
                summary_plot_data[quena_name]['markers'].append('o') # Marker for own inharmonicity
            
            if temp_data_for_quena[current_note]['g2_ref_delta'] is not None:
                # Ensure no duplication if own_delta was already the same as g2_ref_delta for 'G'
                if current_note == 'G' and temp_data_for_quena['G']['own_delta'] is not None and \
                   np.isclose(temp_data_for_quena['G']['own_delta'], temp_data_for_quena['G']['g2_ref_delta'], atol=1e-6):
                    pass # Already plotted as 'o' and is the same value
                else:
                    summary_plot_data[quena_name]['notes'].append(current_note)
                    summary_plot_data[quena_name]['deltas'].append(temp_data_for_quena[current_note]['g2_ref_delta'])
                    summary_plot_data[quena_name]['markers'].append('^') # Marker for G2 inharmonicity

    plt.figure(figsize=(12, 7))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)', zorder=2)
    
    # Adjust Y-axis limits globally for all quenas
    all_deltas = []
    for quena_name in selected_quenas: # Iterate only over selected quenas
        if quena_name in summary_plot_data:
            all_deltas.extend([d for d in summary_plot_data[quena_name]['deltas'] if np.isfinite(d)])

    if all_deltas:
        min_delta = np.min(all_deltas)
        max_delta = np.max(all_deltas)
        padding = (max_delta - min_delta) * 0.15
        plt.ylim(min_delta - padding, max_delta + padding)
    else:
        plt.ylim(-50, 50) # Default limits

    # Plot data for each quena
    # Will use a list for legend handles to manage duplications and order
    legend_handles = []
    seen_legend_labels = set()

    # Add the handle for the 0 cents line first
    line_0_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)')
    legend_handles.append(line_0_handle)
    seen_legend_labels.add('Ideal Inharmonicity (0 cents)')

    for quena_name in sorted(selected_quenas): # Iterate over selected quenas
        if quena_name not in summary_plot_data: # If quena has no data after filtering
            continue
        data = summary_plot_data[quena_name]
        if not data['notes']: continue
        color = quena_colors.get(quena_name, 'gray')
        
        # Flags to ensure only one legend entry per marker type FOR EACH QUENA
        plotted_quena_o_label = False
        plotted_quena_tri_label = False

        for i in range(len(data['notes'])):
            note_val = data['notes'][i]
            delta_val = data['deltas'][i]
            marker_val = data['markers'][i]
            
            label_for_point = None
            
            if marker_val == 'o' and not plotted_quena_o_label:
                label_for_point = f"{quena_name} (own)"
                plotted_quena_o_label = True
            elif marker_val == '^' and not plotted_quena_tri_label:
                label_for_point = f"{quena_name} (from G2)"
                plotted_quena_tri_label = True
            
            plt.scatter(note_val, delta_val, marker=marker_val, color=color, s=80, edgecolors='k', linewidth=0.5, zorder=3,
                        label=label_for_point) # Label will be used if not None

            # Build legend handles explicitly and uniquely
            if label_for_point and label_for_point not in seen_legend_labels:
                new_handle = plt.Line2D([0], [0], marker=marker_val, color='w', markerfacecolor=color, markersize=8, label=label_for_point, markeredgecolor='k')
                legend_handles.append(new_handle)
                seen_legend_labels.add(label_for_point)

    plt.title("Octave Inharmonicity - Comparative Summary of Quenas", fontsize=16, weight='bold')
    plt.xlabel("Note", fontsize=12)
    plt.ylabel("Inharmonicity (cents)", fontsize=12)
    
    # Order legend labels (optional, but improves readability)
    # First "ideal", then quenas by name and type
    def sort_key_legend(label):
        if "ideal" in label: return (0, label)
        parts = label.split(' ')
        quena_name = parts[0]
        type_str = parts[-1]
        return (1, quena_name, type_str)

    # Sort handles and labels for the final legend
    sorted_handles_labels = sorted(zip(legend_handles, [h.get_label() for h in legend_handles]), key=lambda x: sort_key_legend(x[1]))
    final_handles = [h for h, l in sorted_handles_labels]
    final_labels = [l for h, l in sorted_handles_labels]

    plt.legend(handles=final_handles, labels=final_labels, loc='best', title="Quena and Calculation Type",
               fontsize=10, title_fontsize=12, frameon=True)

    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.xticks(summary_note_order, fontsize=11)
    plt.yticks(fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # Adjust for title and legend
    os.makedirs("images", exist_ok=True)
    output_filename = "images/inharmonicity_comparative_summary.png"
    plt.savefig(output_filename, dpi=300)
    # print(f"Inharmonicity summary plot saved to {output_filename}")
    plt.close()

def plot_admittance_summary(base_directory, subdirectories, selected_quenas): # Added selected_quenas
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    fig, axs = plt.subplots(1, len(plot_note_order), figsize=(3.5 * len(plot_note_order), 7), sharey=True)
    
    # Handle the case of a single subplot
    if len(plot_note_order) == 1:
        axs = [axs]
    note_to_ax_map = dict(zip(plot_note_order, axs))

    quena_colors_map = {
        "Quena1": "tab:blue", "Quena2": "tab:orange",
        "Quena3": "tab:green", "Quena4": "tab:red",
    }
    
    data_to_plot = defaultdict(list) # {(note, quena_name): [(frequencies, admittances)]}

    for subdir_name in selected_quenas: # Iterate only over selected quenas
        quena_data_directory = os.path.join(base_directory, subdir_name)
        if not os.path.exists(quena_data_directory):
            continue
        
        dat_files = [f for f in os.listdir(quena_data_directory) if f.endswith('.dat')]
        for file_name in dat_files:
            full_path = os.path.join(quena_data_directory, file_name)
            file_data = read_and_format_data(full_path)
            
            # Check file format
            if file_data.ndim != 2 or file_data.shape[1] != 2 or file_data.size == 0:
                # print(f"Warning: File {file_name} in {subdir_name} does not have 2-column format or is empty. Skipping in summary.")
                continue

            frequency_vals = file_data[:, 0]
            impedance_vals = file_data[:, 1]
            
            # Filter by frequency range
            interest_mask = (frequency_vals >= 200) & (frequency_vals <= 3000)
            filtered_frequency = frequency_vals[interest_mask]
            filtered_impedance = impedance_vals[interest_mask]
            
            # Calculate admittance
            with np.errstate(divide='ignore', invalid='ignore'):
                calculated_admittance = np.where(filtered_impedance != 0, 1 / filtered_impedance, np.nan)
            
            # Extract note and normalize F to F#
            file_note = os.path.splitext(file_name)[0].split('_')[-1]
            if file_note == 'F':
                processed_note = 'F#'
            else:
                processed_note = file_note
            if processed_note not in plot_note_order:
                continue
            data_to_plot[(processed_note, subdir_name)].append((filtered_frequency, calculated_admittance))
    
    # Plot curves in corresponding subplots
    for note_idx, plot_note in enumerate(plot_note_order):
        ax_target = note_to_ax_map[plot_note]
        
        # To control the legend, as each line should not have a unique label for each iteration
        plotted_legend_labels = set()

        for subdir_name in selected_quenas: # Iterate only over selected quenas
            data_key = (plot_note, subdir_name)
            if data_key in data_to_plot:
                color_val = quena_colors_map.get(subdir_name, 'gray')
                label_for_legend = None
                if subdir_name not in plotted_legend_labels:
                    label_for_legend = subdir_name
                    plotted_legend_labels.add(subdir_name)

                for frec, admit in data_to_plot[data_key]:
                    ax_target.plot(frec, admit, color=color_val, linewidth=1.5, alpha=0.8, label=label_for_legend)
                    label_for_legend = None # Only the first line of each quena will have the label

        ax_target.set_title(plot_note, fontsize=14)
        ax_target.set_yscale("log")
        ax_target.grid(True, which="both", ls="-", alpha=0.6)
        ax_target.tick_params(axis='x', labelsize=10)
        ax_target.tick_params(axis='y', labelsize=10)

    for ax_obj in axs:
        ax_obj.set_xlabel("Frequency (Hz)", fontsize=12)
    axs[0].set_ylabel("Admittance (Siemens)", fontsize=12)
    fig.suptitle("Admittance Curves - Comparative Summary of Quenas", fontsize=16, weight='bold')
    
    # Create legend for quenas (collecting handles from a single ax if necessary)
    # Better approach for global legend: create manually
    legend_handles = []
    for quena_name in sorted(selected_quenas): # Iterate over selected quenas for legend
        if quena_name in quena_colors_map:
            legend_handles.append(plt.Line2D([0], [0], color=quena_colors_map[quena_name], lw=2.5, label=quena_name))
    
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 0.95), title="Quena",
                   fontsize=10, title_fontsize=12, frameon=True)
        
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    os.makedirs("images", exist_ok=True)
    output_filename = "images/admittance_comparative_summary.png"
    plt.savefig(output_filename, dpi=300)
    # print(f"Admittance summary plot saved to {output_filename}")
    plt.close()

def plot_deviation_and_inharmonicity(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    tempered_notes_map = {
        'G': 391.995, 'A': 440.000, 'B': 493.883, 'C': 523.251,
        'D': 587.330, 'E': 659.255, 'F#': 739.989, 'G2': 783.991,
    }
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    
    # Store G2 fundamental for use as f1 for note G
    g2_f0_for_G_f1 = {}
    # Store G2's own inharmonicity
    g2_inharmonicity = {}
    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        if key.endswith('G2') and len(freqs) >= 1:
            quena_name = key.split('_')[0]
            g2_f0_for_G_f1[quena_name] = freqs[0]
            if len(freqs) >= 2 and freqs[0] > 0 and freqs[1] > 0:
                g2_inharmonicity[quena_name] = 1200 * np.log2(freqs[1] / (2 * freqs[0]))

    # Structure to store combined data per quena
    quena_combined_summary = defaultdict(lambda: {
        'notes': [], 'dev1': [], 'dev2': [], 'inharm': [], 'inharm_markers': []
    })

    for quena_name in sorted(selected_quenas): # Iterate only over selected quenas
        # Create a temporary map to organize data by note before adding to the list
        temp_data_map = {note: {'dev1': np.nan, 'dev2': np.nan, 'inharm': np.nan, 'inharm_marker': 's'} for note in plot_note_order}

        for current_note in plot_note_order:
            # Normalize 'F' to 'F#' in note keys
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            current_note_key = f"{quena_name}_{key_note}"
            if current_note_key in filtered_peaks_data: # Use filtered_peaks_data
                note_freqs, _ = filtered_peaks_data[current_note_key] # Use filtered_peaks_data
                f0 = note_freqs[0] if len(note_freqs) >= 1 else None
                f1 = None
                # Para la desviación de la 2ª octava, siempre usa el segundo armónico detectado en la nota, tanto para G como el resto
                if len(note_freqs) >= 2:
                    f1 = note_freqs[1]
                else:
                    f1 = None
                # Calculate 1st octave deviation
                if f0 is not None and f0 > 0:
                    if tempered_notes_map.get(key_note) is not None:
                        ref1 = tempered_notes_map[key_note]
                        delta1_val = 1200 * np.log2(f0 / ref1)
                        temp_data_map[key_note]['dev1'] = delta1_val if np.isfinite(delta1_val) else np.nan
                # Calculate 2nd octave deviation and inharmonicity
                if f0 is not None and f1 is not None and f0 > 0 and f1 > 0:
                    if tempered_notes_map.get(key_note) is not None:
                        ref2 = 2 * tempered_notes_map[key_note]
                        delta2_val = 1200 * np.log2(f1 / ref2)
                        temp_data_map[key_note]['dev2'] = delta2_val if np.isfinite(delta2_val) else np.nan
                    delta_oct_val = 1200 * np.log2(f1 / (2 * f0))
                    temp_data_map[key_note]['inharm'] = delta_oct_val if np.isfinite(delta_oct_val) else np.nan
                    # Determine inharmonicity marker: square for own, triangle for G2 referenced in G
                    if key_note == 'G' and quena_name in g2_inharmonicity and np.isclose(delta_oct_val, g2_inharmonicity[quena_name], atol=1e-6):
                        temp_data_map[key_note]['inharm_marker'] = '^'
                    else:
                        temp_data_map[key_note]['inharm_marker'] = 's'
            
        # --- Nueva lógica explícita para la nota G: ---
        # Si para la nota G existen tanto el valor propio (cuadrado) como el de G2 (triángulo), y son distintos, agrega ambos.
        if 'G' in temp_data_map and quena_name in g2_inharmonicity:
            own_inharm = temp_data_map['G']['inharm']
            g2_inharm = g2_inharmonicity[quena_name]
            if np.isfinite(own_inharm) and np.isfinite(g2_inharm) and not np.isclose(own_inharm, g2_inharm, atol=1e-6):
                # Primero el cuadrado propio
                quena_combined_summary[quena_name]['notes'].append('G')
                quena_combined_summary[quena_name]['dev1'].append(temp_data_map['G']['dev1'])
                quena_combined_summary[quena_name]['dev2'].append(temp_data_map['G']['dev2'])
                quena_combined_summary[quena_name]['inharm'].append(own_inharm)
                quena_combined_summary[quena_name]['inharm_markers'].append('s')
                # Luego el triángulo de G2
                quena_combined_summary[quena_name]['notes'].append('G')
                quena_combined_summary[quena_name]['dev1'].append(np.nan)
                quena_combined_summary[quena_name]['dev2'].append(np.nan)
                quena_combined_summary[quena_name]['inharm'].append(g2_inharm)
                quena_combined_summary[quena_name]['inharm_markers'].append('^')
                # Para evitar agregar dos veces G abajo, marca como "ya hecho"
                skip_g = True
            else:
                skip_g = False
        else:
            skip_g = False

        # Caso general: agrega los datos de cada nota
        for note in plot_note_order:
            if skip_g and note == 'G':
                continue  # Ya agregados arriba ambos puntos de G
            if note not in quena_combined_summary[quena_name]['notes']:
                quena_combined_summary[quena_name]['notes'].append(note)
                quena_combined_summary[quena_name]['dev1'].append(temp_data_map[note]['dev1'])
                quena_combined_summary[quena_name]['dev2'].append(temp_data_map[note]['dev2'])
                quena_combined_summary[quena_name]['inharm'].append(temp_data_map[note]['inharm'])
                quena_combined_summary[quena_name]['inharm_markers'].append(temp_data_map[note]['inharm_marker'])
        
    os.makedirs("images", exist_ok=True)
    for quena_name in sorted(selected_quenas): # Iterate only over selected quenas
        if quena_name not in quena_combined_summary: # If quena has no data after filtering
            continue
        data = quena_combined_summary[quena_name]
        if not data['notes']:
            continue
        plt.figure(figsize=(10, 6))
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Reference (0 cents)', zorder=2)
        
        # Plot 1st octave deviation (en orden correcto)
        plot_x_dev1, plot_y_dev1 = [], []
        for note in plot_note_order:
            if note in data['notes']:
                idx = data['notes'].index(note)
                val = data['dev1'][idx]
                if np.isfinite(val):
                    plot_x_dev1.append(note)
                    plot_y_dev1.append(val)
                else:
                    plot_x_dev1.append(note)
                    plot_y_dev1.append(np.nan)
            else:
                plot_x_dev1.append(note)
                plot_y_dev1.append(np.nan)
        plt.plot(plot_x_dev1, plot_y_dev1, color='tab:blue', linestyle='--', marker='o', markersize=8, label='1st Octave (deviation)', zorder=3)

        # Plot 2nd octave deviation (en orden correcto)
        plot_x_dev2, plot_y_dev2 = [], []
        for note in plot_note_order:
            if note in data['notes']:
                idx = data['notes'].index(note)
                val = data['dev2'][idx]
                if np.isfinite(val):
                    plot_x_dev2.append(note)
                    plot_y_dev2.append(val)
                else:
                    plot_x_dev2.append(note)
                    plot_y_dev2.append(np.nan)
            else:
                plot_x_dev2.append(note)
                plot_y_dev2.append(np.nan)
        plt.plot(plot_x_dev2, plot_y_dev2, color='tab:orange', linestyle=':', marker='s', markersize=8, label='2nd Octave (deviation)', zorder=3)

        # Flags to ensure a single legend entry per inharmonicity type
        plotted_inharm_s_label = False
        plotted_inharm_tri_label = False

        # Plot inharmonicity (en orden correcto)
        for note in plot_note_order:
            if note in data['notes']:
                # Puede haber varios (G, por ejemplo, si hay triángulo y cuadrado)
                idxs = [i for i, n in enumerate(data['notes']) if n == note]
                for idx in idxs:
                    v = data['inharm'][idx]
                    m = data['inharm_markers'][idx]
                    if not np.isnan(v):
                        current_label = None
                        if m == 's' and not plotted_inharm_s_label:
                            current_label = 'Own Inharmonicity'
                            plotted_inharm_s_label = True
                        elif m == '^' and not plotted_inharm_tri_label:
                            current_label = 'G2 Inharmonicity (referenced at G)'
                            plotted_inharm_tri_label = True
                        plt.scatter(note, v, marker=m, color='tab:green', s=100, edgecolors='k', linewidth=0.5,
                                    label=current_label, zorder=4)


        plt.title(f"Deviations and Inharmonicity - Quena: {quena_name}", fontsize=15, weight='bold')
        plt.xlabel("Note", fontsize=12)
        plt.ylabel("Deviation / Inharmonicity (cents)", fontsize=12)
        plt.xticks(plot_note_order, fontsize=11)
        plt.yticks(fontsize=10)
        
        # Adjust Y-axis limits
        all_values = [d for d in plot_y_dev1 + plot_y_dev2 + data['inharm'] if np.isfinite(d)]
        if all_values:
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            padding = (max_val - min_val) * 0.15
            plt.ylim(min_val - padding, max_val + padding)
        else:
            plt.ylim(-50, 50)

        # Create final legend, avoiding duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen_labels = set()
        for h, l in zip(handles, labels):
            if l not in seen_labels:
                unique_handles.append(h)
                unique_labels.append(l)
                seen_labels.add(l)
        
        plt.legend(unique_handles, unique_labels, loc='best', fontsize=10, title="Measurement", title_fontsize=12, frameon=True)
        plt.grid(True, which="both", ls="-", alpha=0.6)
        plt.tight_layout()
        output_filename = f"images/deviation_inharmonicity_{quena_name}.png"
        plt.savefig(output_filename, dpi=300)
        # print(f"Combined plot saved to {output_filename}")
        plt.close()

def plot_inharmonicity_mean_std(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    summary_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
    
    # Structure to store all inharmonicity values per note across all selected quenas
    inharmonicity_values_per_note = defaultdict(list)

    # Collect fundamental G2 for each quena
    g2_f0_refs = {}
    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        if key.endswith("G2") and len(freqs) >= 1:
            quena_name = key.split('_')[0]
            g2_f0_refs[quena_name] = freqs[0]

    for quena_name in sorted(selected_quenas): # Iterate only over selected quenas
        if quena_name not in selected_quenas: # Explicitly check
            continue
        # Temp storage to ensure only one "own" and one "G2-ref" inharmonicity per note per quena
        temp_inharm_data_for_quena = defaultdict(lambda: {'own': np.nan, 'g2_ref': np.nan})

        for current_note in summary_note_order:
            # Normalize 'F' to 'F#'
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            note_key = f"{quena_name}_{key_note}"
            if note_key in filtered_peaks_data: # Use filtered_peaks_data
                note_freqs, _ = filtered_peaks_data[note_key] # Use filtered_peaks_data
                f0, f1 = None, None
                if len(note_freqs) >= 1: f0 = note_freqs[0]
                if key_note == 'G' and quena_name in g2_f0_refs:
                    f1 = g2_f0_refs[quena_name] # Use G2 as f1 for G
                elif len(note_freqs) >= 2:
                    f1 = note_freqs[1]
                if f0 is not None and f1 is not None and f0 > 0 and f1 > 0:
                    delta_oct = 1200 * np.log2(f1 / (2 * f0))
                    if np.isfinite(delta_oct):
                        temp_inharm_data_for_quena[key_note]['own'] = delta_oct

        # If G2 is present, calculate its own inharmonicity
        g2_key = f"{quena_name}_G2"
        if g2_key in filtered_peaks_data: # Use filtered_peaks_data
            g2_freqs, _ = filtered_peaks_data[g2_key] # Use filtered_peaks_data
            if len(g2_freqs) >= 2:
                f0_g2, f1_g2 = g2_freqs[0], g2_freqs[1]
                if f0_g2 > 0 and f1_g2 > 0:
                    delta_oct_g2_own = 1200 * np.log2(f1_g2 / (2 * f0_g2))
                    if np.isfinite(delta_oct_g2_own):
                        temp_inharm_data_for_quena['G']['g2_ref'] = delta_oct_g2_own

        # Add the unique inharmonicity values for each note to the collective list
        for note in summary_note_order:
            if not np.isnan(temp_inharm_data_for_quena[note]['own']):
                inharmonicity_values_per_note[f"{note}_own"].append(temp_inharm_data_for_quena[note]['own'])
            
            if note == 'G' and not np.isnan(temp_inharm_data_for_quena['G']['g2_ref']):
                # Only add G2 reference if it's different from the 'own' inharmonicity for G
                if np.isnan(temp_inharm_data_for_quena['G']['own']) or \
                   not np.isclose(temp_inharm_data_for_quena['G']['own'], temp_inharm_data_for_quena['G']['g2_ref'], atol=1e-6):
                    inharmonicity_values_per_note[f"{note}_g2_ref"].append(temp_inharm_data_for_quena['G']['g2_ref'])

    # Calculate mean and standard deviation
    mean_inharmonicity = {}
    std_inharmonicity = {}

    for note_type, values in inharmonicity_values_per_note.items():
        if values:
            mean_inharmonicity[note_type] = np.mean(values)
            std_inharmonicity[note_type] = np.std(values)
        else:
            mean_inharmonicity[note_type] = np.nan
            std_inharmonicity[note_type] = np.nan

    # Prepare data for plotting
    plot_notes = []
    plot_means = []
    plot_stds = []
    plot_markers = [] # 'o' for own, '^' for G2-ref
    plot_colors = [] # Different colors for 'own' vs 'G2-ref' points

    # Ensure notes are in the desired order and include both types
    for note in summary_note_order:
        # Own inharmonicity
        key_own = f"{note}_own"
        if not np.isnan(mean_inharmonicity.get(key_own, np.nan)):
            plot_notes.append(note)
            plot_means.append(mean_inharmonicity[key_own])
            plot_stds.append(std_inharmonicity[key_own])
            plot_markers.append('o')
            plot_colors.append('tab:blue')

        # G2 referenced inharmonicity (only for G)
        if note == 'G':
            key_g2_ref = f"{note}_g2_ref"
            if not np.isnan(mean_inharmonicity.get(key_g2_ref, np.nan)):
                plot_notes.append(note)
                plot_means.append(mean_inharmonicity[key_g2_ref])
                plot_stds.append(std_inharmonicity[key_g2_ref])
                plot_markers.append('^')
                plot_colors.append('tab:purple') # Distinct color for G2 reference


    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)', zorder=2)

    # Plot mean values and error bars
    # Custom legend entries needed to avoid duplicates
    legend_handles = []
    seen_labels = set()

    for i, note in enumerate(plot_notes):
        mean_val = plot_means[i]
        std_val = plot_stds[i]
        marker_val = plot_markers[i]
        color_val = plot_colors[i]
        
        label_for_point = None
        if marker_val == 'o' and 'Mean Own Inharmonicity' not in seen_labels:
            label_for_point = 'Mean Own Inharmonicity'
            seen_labels.add('Mean Own Inharmonicity')
        elif marker_val == '^' and 'Mean G2 Inharmonicity (referenced at G)' not in seen_labels:
            label_for_point = 'Mean G2 Inharmonicity (referenced at G)'
            seen_labels.add('Mean G2 Inharmonicity (referenced at G)')

        plt.errorbar(note, mean_val, yerr=std_val, fmt=marker_val, color=color_val,
                     markersize=10, capsize=5, ecolor='black', markeredgecolor='k',
                     label=label_for_point, zorder=3)

    plt.title("Mean Octave Inharmonicity and Standard Deviation Across Selected Quenas", fontsize=15, weight='bold')
    plt.xlabel("Note", fontsize=12)
    plt.ylabel("Inharmonicity (cents)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    
    plt.xticks(summary_note_order, fontsize=11)
    plt.yticks(fontsize=10)

    # Dynamically adjust Y-axis limits based on means and standard deviations
    all_combined_values = []
    for m, s in zip(plot_means, plot_stds):
        if np.isfinite(m):
            all_combined_values.append(m)
            if np.isfinite(s):
                all_combined_values.append(m - s)
                all_combined_values.append(m + s)
    
    if all_combined_values:
        min_val = np.min(all_combined_values)
        max_val = np.max(all_combined_values)
        padding = (max_val - min_val) * 0.15
        plt.ylim(min_val - padding, max_val + padding)
    else:
        plt.ylim(-50, 50) # Default limits

    # Reconstruct legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_final = []
    unique_labels_final = []
    seen_labels_final = set()

    # Add the 0 cents line first
    line_0_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Inharmonicity (0 cents)')
    unique_handles_final.append(line_0_handle)
    unique_labels_final.append('Ideal Inharmonicity (0 cents)')
    seen_labels_final.add('Ideal Inharmonicity (0 cents)')

    # Add other labels in a preferred order
    order = ['Mean Own Inharmonicity', 'Mean G2 Inharmonicity (referenced at G)']
    for l_ord in order:
        for h, l_curr in zip(handles, labels):
            if l_curr == l_ord and l_curr not in seen_labels_final:
                unique_handles_final.append(h)
                unique_labels_final.append(l_curr)
                seen_labels_final.add(l_curr)

    plt.legend(unique_handles_final, unique_labels_final, loc='best', title="Inharmonicity Type",
               fontsize=10, title_fontsize=12, frameon=True)

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    output_filename = "images/inharmonicity_mean_std_summary.png"
    plt.savefig(output_filename, dpi=300)
    # print(f"Mean and STD inharmonicity plot saved to {output_filename}")
    plt.close()

def plot_deviation_summary(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    tempered_notes = {
        'G': 391.995, 'A': 440.000, 'B': 493.883, 'C': 523.251,
        'D': 587.330, 'E': 659.255, 'F#': 739.989, 'G2': 783.991,
    }
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    
    quena_colors = {
        "Quena1": "tab:blue", "Quena2": "tab:orange",
        "Quena3": "tab:green", "Quena4": "tab:red",
    }

    plt.figure(figsize=(12, 7))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)', zorder=2)

    legend_handles = []
    seen_labels = set()

    # Add the handle for the 0 cents line first
    line_0_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)')
    legend_handles.append(line_0_handle)
    seen_labels.add('Ideal Deviation (0 cents)')

    for quena_name in sorted(selected_quenas): # Iterate only over selected quenas
        data_for_quena = defaultdict(lambda: {'dev1': np.nan, 'dev2': np.nan})
        for current_note in plot_note_order:
            # Normalize 'F' to 'F#'
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            note_key = f"{quena_name}_{key_note}"
            if note_key in filtered_peaks_data: # Use filtered_peaks_data
                note_freqs, _ = filtered_peaks_data[note_key] # Use filtered_peaks_data
                if len(note_freqs) >= 2:
                    f0, f1 = note_freqs[0], note_freqs[1]
                    ref1 = tempered_notes.get(key_note)
                    ref2 = 2 * ref1 if ref1 is not None else None
                    if f0 > 0 and ref1 is not None and ref1 > 0:
                        delta1 = 1200 * np.log2(f0 / ref1)
                        if np.isfinite(delta1):
                            data_for_quena[key_note]['dev1'] = delta1
                    if f1 > 0 and ref2 is not None and ref2 > 0:
                        delta2 = 1200 * np.log2(f1 / ref2)
                        if np.isfinite(delta2):
                            data_for_quena[key_note]['dev2'] = delta2

        # Use fixed x indices for plotting to avoid misalignment
        x_map = {note: i for i, note in enumerate(plot_note_order)}
        x = np.arange(len(plot_note_order))
        current_plot_dev1 = [data_for_quena[n]['dev1'] if n in data_for_quena else np.nan for n in plot_note_order]
        current_plot_dev2 = [data_for_quena[n]['dev2'] if n in data_for_quena else np.nan for n in plot_note_order]

        color = quena_colors.get(quena_name, 'gray')

        # Plot 1st octave deviation for this quena
        label_dev1 = f"{quena_name} (1st Octave)"
        plt.plot(x, current_plot_dev1, color=color, linestyle='--', marker='o', markersize=6,
                 linewidth=1, label=label_dev1, zorder=3)
        if label_dev1 not in seen_labels:
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='--', marker='o', markersize=6, label=label_dev1))
            seen_labels.add(label_dev1)

        # Plot 2nd octave deviation for this quena
        label_dev2 = f"{quena_name} (2nd Octave)"
        plt.plot(x, current_plot_dev2, color=color, linestyle=':', marker='s', markersize=6,
                 linewidth=1, label=label_dev2, zorder=3)
        if label_dev2 not in seen_labels:
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle=':', marker='s', markersize=6, label=label_dev2))
            seen_labels.add(label_dev2)

    plt.title("Resonance Deviation from Tempered Note - Comparative Summary of Selected Quenas", fontsize=16, weight='bold')
    plt.xlabel("Note", fontsize=12)
    plt.ylabel("Deviation (cents)", fontsize=12)
    
    # Adjust Y-axis limits
    all_deviations = []
    for quena_name in sorted(selected_quenas): # Iterate only over selected quenas
        data_for_quena = defaultdict(lambda: {'dev1': np.nan, 'dev2': np.nan})
        for current_note in plot_note_order:
            # Normalize 'F' to 'F#'
            key_note = current_note
            if key_note == 'F':
                key_note = 'F#'
            note_key = f"{quena_name}_{key_note}"
            if note_key in filtered_peaks_data: # Use filtered_peaks_data
                note_freqs, _ = filtered_peaks_data[note_key] # Use filtered_peaks_data
                if len(note_freqs) >= 2:
                    f0, f1 = note_freqs[0], note_freqs[1]
                    ref1 = tempered_notes.get(key_note)
                    ref2 = 2 * ref1 if ref1 is not None else None
                    if f0 > 0 and ref1 is not None and ref1 > 0:
                        delta1 = 1200 * np.log2(f0 / ref1)
                        if np.isfinite(delta1): all_deviations.append(delta1)
                    if f1 > 0 and ref2 is not None and ref2 > 0:
                        delta2 = 1200 * np.log2(f1 / ref2)
                        if np.isfinite(delta2): all_deviations.append(delta2)

    if all_deviations:
        min_delta = np.min(all_deviations)
        max_delta = np.max(all_deviations)
        padding = (max_delta - min_delta) * 0.15
        plt.ylim(min_delta - padding, max_delta + padding)
    else:
        plt.ylim(-50, 50)
    
    # Sort legend handles and labels
    sorted_handles_labels = sorted(zip(legend_handles, [h.get_label() for h in legend_handles]), key=lambda x: (x[1].split(' ')[0], x[1].split(' ')[1]))
    final_handles = [h for h, l in sorted_handles_labels]
    final_labels = [l for h, l in sorted_handles_labels]

    # Insert the 'Ideal Deviation' handle at the beginning
    final_handles.insert(0, plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)'))
    final_labels.insert(0, 'Ideal Deviation (0 cents)')

    plt.legend(handles=final_handles, labels=final_labels, loc='best', title="Quena and Octave",
               fontsize=9, title_fontsize=11, frameon=True)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    x = np.arange(len(plot_note_order))
    plt.xticks(x, plot_note_order, fontsize=11)
    plt.yticks(fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    os.makedirs("images", exist_ok=True)
    output_filename = "images/deviation_comparative_summary.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_deviation_mean_std(peaks_data, selected_quenas): # Added selected_quenas
    # Filter peaks_data to include only selected quenas
    filtered_peaks_data = {k: v for k, v in peaks_data.items() if k.split('_')[0] in selected_quenas}

    tempered_notes = {
        'G': 391.995, 'A': 440.000, 'B': 493.883, 'C': 523.251,
        'D': 587.330, 'E': 659.255, 'F#': 739.989, 'G2': 783.991,
    }
    plot_note_order = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']
    
    # Structure to store all deviation values per note across all selected quenas
    dev1_values_per_note = defaultdict(list)
    dev2_values_per_note = defaultdict(list)

    for key, (freqs, _) in filtered_peaks_data.items(): # Use filtered_peaks_data
        quena_name, original_note = key.split('_')
        if original_note == 'F':
            processed_note = 'F#'
        else:
            processed_note = original_note
        if processed_note in tempered_notes and len(freqs) >= 2:
            f0, f1 = freqs[0], freqs[1]
            ref1 = tempered_notes[processed_note]
            ref2 = 2 * ref1
            if f0 > 0 and ref1 > 0:
                delta1 = 1200 * np.log2(f0 / ref1)
                if np.isfinite(delta1):
                    dev1_values_per_note[processed_note].append(delta1)
            if f1 > 0 and ref2 > 0:
                delta2 = 1200 * np.log2(f1 / ref2)
                if np.isfinite(delta2):
                    dev2_values_per_note[processed_note].append(delta2)

    # Calculate mean and standard deviation
    mean_dev1 = {}
    std_dev1 = {}
    mean_dev2 = {}
    std_dev2 = {}

    for note in plot_note_order:
        if dev1_values_per_note[note]:
            mean_dev1[note] = np.mean(dev1_values_per_note[note])
            std_dev1[note] = np.std(dev1_values_per_note[note])
        else:
            mean_dev1[note] = np.nan
            std_dev1[note] = np.nan
        
        if dev2_values_per_note[note]:
            mean_dev2[note] = np.mean(dev2_values_per_note[note])
            std_dev2[note] = np.std(dev2_values_per_note[note])
        else:
            mean_dev2[note] = np.nan
            std_dev2[note] = np.nan

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)', zorder=2)

    # Plot 1st octave mean and std
    plot_x_dev1 = []
    plot_y_dev1 = []
    yerr_dev1 = []
    for note in plot_note_order:
        if not np.isnan(mean_dev1[note]):
            plot_x_dev1.append(note)
            plot_y_dev1.append(mean_dev1[note])
            yerr_dev1.append(std_dev1[note])
    
    if plot_x_dev1:
        plt.errorbar(plot_x_dev1, plot_y_dev1, yerr=yerr_dev1, fmt='o', color='tab:blue',
                     markersize=10, capsize=5, ecolor='black', markeredgecolor='k',
                     label='Mean 1st Octave Deviation', zorder=3)

    # Plot 2nd octave mean and std
    plot_x_dev2 = []
    plot_y_dev2 = []
    yerr_dev2 = []
    for note in plot_note_order:
        if not np.isnan(mean_dev2[note]):
            plot_x_dev2.append(note)
            plot_y_dev2.append(mean_dev2[note])
            yerr_dev2.append(std_dev2[note])

    if plot_x_dev2:
        plt.errorbar(plot_x_dev2, plot_y_dev2, yerr=yerr_dev2, fmt='s', color='tab:orange',
                     markersize=10, capsize=5, ecolor='black', markeredgecolor='k',
                     label='Mean 2nd Octave Deviation', zorder=3)

    plt.title("Mean Resonance Deviation from Tempered Note and Standard Deviation Across Selected Quenas", fontsize=15, weight='bold')
    plt.xlabel("Note", fontsize=12)
    plt.ylabel("Deviation (cents)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    
    plt.xticks(plot_note_order, fontsize=11)
    plt.yticks(fontsize=10)

    # Adjust Y-axis limits
    all_combined_values = []
    for m, s in zip(plot_y_dev1, yerr_dev1):
        if np.isfinite(m):
            all_combined_values.append(m)
            if np.isfinite(s):
                all_combined_values.append(m - s)
                all_combined_values.append(m + s)
    for m, s in zip(plot_y_dev2, yerr_dev2):
        if np.isfinite(m):
            all_combined_values.append(m)
            if np.isfinite(s):
                all_combined_values.append(m - s)
                all_combined_values.append(m + s)

    if all_combined_values:
        min_val = np.min(all_combined_values)
        max_val = np.max(all_combined_values)
        padding = (max_val - min_val) * 0.15
        plt.ylim(min_val - padding, max_val + padding)
    else:
        plt.ylim(-50, 50)

    # Reconstruct legend to ensure desired order
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_final = []
    unique_labels_final = []
    seen_labels_final = set()

    # Add the 0 cents line first
    line_0_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ideal Deviation (0 cents)')
    unique_handles_final.append(line_0_handle)
    unique_labels_final.append('Ideal Deviation (0 cents)')
    seen_labels_final.add('Ideal Deviation (0 cents)')

    # Add other labels in a preferred order
    order = ['Mean 1st Octave Deviation', 'Mean 2nd Octave Deviation']
    for l_ord in order:
        for h, l_curr in zip(handles, labels):
            if l_curr == l_ord and l_curr not in seen_labels_final:
                unique_handles_final.append(h)
                unique_labels_final.append(l_curr)
                seen_labels_final.add(l_curr)

    plt.legend(unique_handles_final, unique_labels_final, loc='best', title="Octave Deviation",
               fontsize=10, title_fontsize=12, frameon=True)
    
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    output_filename = "images/deviation_mean_std_summary.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

# --- Main execution block ---
if __name__ == "__main__":
    peaks_global = {} # Reset the peaks dictionary
    base_directory = "MEDICIONES"

    # Define here which quenas you want to include in the analysis and plots
    # For example, to exclude Quena1:
    selected_quenas = ["Quena1","Quena2", "Quena3", "Quena4"] 
    # To include all:
    # selected_quenas = ["Quena1", "Quena2", "Quena3", "Quena4"]

    # First, process all raw data to populate peaks_global,
    # as individual admittance plots are still useful for all quenas.
    # The subsequent summary plots will filter based on selected_quenas.
    
    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        print(f"Error: Base directory '{base_directory}' not found. Please create it and place your Quena data folders inside.")
    else:
        all_subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d)) and d.startswith("Quena")]

        for subdir in all_subdirectories:
            data_directory = os.path.join(base_directory, subdir)
            print(f"Processing {data_directory}...")
            plot_impedances_from_directory(data_directory) # This populates peaks_global for ALL quenas initially

        # Only generate summary plots if peak data was found for selected quenas
        # Check if there is data for at least one selected quena
        any_selected_quena_data = any(q in k for k in peaks_global for q in selected_quenas)

        print("\n------ CHECK-PEAKS FOR EACH QUENA AND NOTE ------")
        for quena_name in selected_quenas:
            for note in ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G2']:
                key = f"{quena_name}_{note}"
                if key in peaks_global:
                    freqs, _ = peaks_global[key]
                    print(f"CHECK-PEAKS | {key} | Num peaks: {len(freqs)} | Freqs: {freqs}")
                else:
                    print(f"CHECK-PEAKS | {key} | NOT FOUND")
        print("--------------------------------------------------\n")

        if not peaks_global or not any_selected_quena_data:
            print("No valid data found for the selected quenas after processing directories.")
            print("Ensure .dat files exist and are processed correctly for your selected quenas.")
        else:
            print(f"\nGenerating summary plots for selected quenas: {', '.join(selected_quenas)}...")
            plot_peaks_by_quena(peaks_global, selected_quenas)
            plot_inharmonicity(peaks_global, selected_quenas)
            plot_fundamental_deviation(peaks_global, selected_quenas)
            plot_inharmonicity_summary(peaks_global, selected_quenas)
            plot_admittance_summary(base_directory, all_subdirectories, selected_quenas) # Pass selected_quenas here too
            plot_deviation_and_inharmonicity(peaks_global, selected_quenas)
            plot_inharmonicity_mean_std(peaks_global, selected_quenas)
            plot_deviation_summary(peaks_global, selected_quenas)
            plot_deviation_mean_std(peaks_global, selected_quenas)

        print("\nAnalysis and plotting process completed. Plots saved in the 'images' folder.")