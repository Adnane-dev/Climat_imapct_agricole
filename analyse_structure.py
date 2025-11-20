import os

def afficher_arborescence(path, prefix="", output_lines=[]):
    elements = sorted(os.listdir(path))
    
    for i, element in enumerate(elements):
        full_path = os.path.join(path, element)
        connector = "└── " if i == len(elements) - 1 else "├── "
        line = prefix + connector + element
        output_lines.append(line)

        if os.path.isdir(full_path):
            extension = "    " if i == len(elements) - 1 else "│   "
            afficher_arborescence(full_path, prefix + extension, output_lines)

    return output_lines


if __name__ == "__main__":
    dossier = os.getcwd()  # 🔥 analyse le dossier actuel automatiquement
    
    print(f"\n📁 Structure du projet dans : {dossier}\n")

    lignes = afficher_arborescence(dossier)

    # Affichage à l'écran
    for ligne in lignes:
        print(ligne)

    # Sauvegarde dans un fichier texte
    with open("structure_projet.txt", "w", encoding="utf-8") as f:
        for ligne in lignes:
            f.write(ligne + "\n")

    print("\n✔️ Structure enregistrée dans 'structure_projet.txt'")
