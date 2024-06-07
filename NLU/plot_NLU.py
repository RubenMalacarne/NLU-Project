import os
import matplotlib.pyplot as plt

def plot_performance(data, metric_prefix, title):
    descriptions = [d['description'] for d in data]
    f1_scores = [d[f'F1_{metric_prefix}'] for d in data]
    precision_scores = [d[f'P-{metric_prefix}'] for d in data]
    recall_scores = [d[f'R-{metric_prefix}'] for d in data]

    metrics = ['F1', 'Precision', 'Recall']
    values = [f1_scores, precision_scores, recall_scores]
    styles = ['-', '--', '-.']
    colors = ['purple', 'green', 'orange']

    plt.figure(figsize=(10, 6))

    for j in range(len(metrics)):
        plt.plot(descriptions, values[j], marker='o', linestyle=styles[j], color=colors[j], label=f'{metrics[j]}')

    plt.xlabel('Model Description')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0.8, 1)  # Limitiamo l'asse y per una migliore visualizzazione
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Creazione della cartella per le immagini, se non esiste
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Salvataggio del grafico
    path = os.path.join("images", f"{title.replace(' ', '_')}.png")
    plt.savefig(path, dpi=100)
    
    plt.show()


# I tuoi dati
intent_data = [
{'F1_intent': 0.920037938263585, 'P-intent': 0.9134382043391018, 'R-intent': 0.9339305711086227, 'description': 'Original'}, 
 {'F1_intent': 0.9261963436656339, 'P-intent': 0.9224326928600552, 'R-intent': 0.9384098544232923, 'description': 'Bidirectional'}, 
 {'F1_intent': 0.9315266586759192, 'P-intent': 0.9417789681048415, 'R-intent': 0.9428891377379619, 'description': 'Bidirectional_dropout'},
 {'F1_intent': 0.9618696389635593, 'P-intent': 0.964771931716168, 'R-intent': 0.9630459126539753, 'description': 'Bert'}
]

slot_data = [
  {'F1_slot': 0.9222378606615059, 'P-slot': 0.9206181945907973, 'R-slot': 0.923863235812478, 'description': 'Original'}, 
 {'F1_slot': 0.9391796322489391, 'P-slot': 0.9421780773323873, 'R-slot': 0.9362002114910116, 'description': 'Bidirectional'}, 
 {'F1_slot': 0.9419698314108252, 'P-slot': 0.9485346676197284, 'R-slot': 0.9354952414522383, 'description': 'Bidirectional_dropout'},
 {'F1_slot': 0.9010262654374674, 'P-slot': 0.8894230769230769, 'R-slot': 0.912936200211491, 'description': 'Bert'}
]


if __name__ == "__main__":
    plot_performance(intent_data, 'intent', 'Intent Performance Comparison of Different Models')
    plot_performance(slot_data, 'slot', 'Slot Performance Comparison of Different Models')

