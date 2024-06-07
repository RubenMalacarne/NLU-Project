import os
import matplotlib.pyplot as plt

def plot_result(intent_data, slot_data):
    titles = ["Intents Classification for Different Models", "Slots for Different Models"]
    datasets = [intent_data, slot_data]
    
    # Impostazione dei colori e degli stili
    colors = ['blue', 'green', 'orange']
    styles = ['-', '--', ':']

    for i, data in enumerate(datasets):
        plt.figure(figsize=(10, 6))
        
        # Estrazione delle descrizioni
        descriptions = [d['description'] for d in data]
        
        # Creazione delle serie di dati
        for j, metric in enumerate(['F1', 'P', 'R']):
            metric_type = 'intent' if i == 0 else 'slot'
            key = f'{metric}_{metric_type}'
            values = [d.get(key, 0) for d in data]  # Utilizza d.get per evitare KeyError
            plt.plot(descriptions, values, marker='o', linestyle=styles[j], color=colors[j], label=f'{metric}-{metric_type}')

        plt.title(titles[i])
        plt.xlabel('Models')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        # Creazione della cartella per le immagini, se non esiste
        if not os.path.exists("images"):
            os.makedirs("images")
        
        # Salvataggio del grafico
        path = os.path.join("images", f"{titles[i].replace(' ', '_')}.png")
        plt.savefig(path, dpi=100)
        
        # Visualizzazione del grafico
        plt.show()

# I tuoi dati
intent_data = [
   {'F1_intent': 0.7637707985511161, 'P-intent': 0.7235248428046838, 'R-intent': 0.8230683090705487, 'description': 'Original'}, 
 {'F1_intent': 0.8326774057232309, 'P-intent': 0.8009292137335925, 'R-intent': 0.8756998880179171, 'description': 'Bidirectional'}, 
 {'F1_intent': 0.8316989222851754, 'P-intent': 0.7980042860999581, 'R-intent': 0.8756998880179171, 'description': 'Bidirectional_dropout'}
]

slot_data = [
  {'F1_slot': 0.6207596993703026, 'P-slot': 0.7325023969319271, 'R-slot': 0.538597109622841, 'description': 'Original'}, 
 {'F1_slot': 0.7967479674796748, 'P-slot': 0.8372815533980582, 'R-slot': 0.7599577017976736, 'description': 'Bidirectional'}, 
 {'F1_slot': 0.8043120774712224, 'P-slot': 0.8349772382397572, 'R-slot': 0.775819527670074, 'description': 'Bidirectional_dropout'}
]


if __name__ == "__main__":
    plot_result(intent_data, slot_data)
