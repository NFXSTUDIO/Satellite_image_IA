<p align="center">
  <br>
  <a href="#">
    <img src="Logo_IPSA.png" alt="Logo du Projet" width="200">
  </a>
  <br>
</p>

<h1 align="center">CI422-Satellite images and machine learning</h1>

<p align="center">
  Projet réalisé par : Charbel GHANEM / Yaasine MOSAFER / Arthur TOUATI
  <br>
  License : MIT
  <br>
  Version : 1.0
  <br>
</p>

---

## Description du Projet

Le but de ce projet est d'analyser des données satellite sur la couche d'ozone et de créer deux modeles de machine learning pour prédire la concentration d'ozone en fonction du temp.
Nous avons un modele de Random Forest Regressor et un modele custom (utilisant un systeme séquentiel).

## Installation

Pour installer et exécuter ce projet, veuillez suivre les étapes suivantes :

1.  **Installer les dépendances :**

    Assurez-vous d'avoir Python installé sur votre système. Ensuite, clonez le dépôt et installez les librairies nécessaires à l'aide de pip. Un fichier `requirements.txt` est généralement fourni pour lister ces dépendances.

    ```bash
    git clone [https://github.com/NFXSTUDIO/Satellite_image_IA](https://github.com/NFXSTUDIO/Satellite_image_IA)
    cd votre-repo
    pip install -r requirements.txt
    ```

2.  **Lancer le code :**

    Le projet propose deux modèles différents.

    * **Modèle Random Forest :** Exécutez le script principal pour lancer le modèle Random Forest.

        ```bash
        python main.py
        ```

    * **Modèle Custom :** Pour le modèle custom, suivez les instructions spécifiques dans le fichier dédié (ou un autre fichier mentionné dans la documentation) pour l'authentification et l'exécution.

        ```bash
        python google_earth_engine.py
        ```

3.  **Analyser les résultats :**

    Une fois l'exécution terminée, les résultats (par exemple, des visualisations, des métriques d'évaluation) seront sauvegardés dans des fichiers spécifiques ou affichés dans la console. Consultez la documentation ou les sorties du programme pour interpréter les résultats obtenus.
