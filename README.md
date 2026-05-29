# ruban-rose - Détection de l'IDC
Solution d'IA basée sur la Computer Vision pour la détection automatisée du carcinome canalaire invasif (IDC) à partir d'images histopathologiques. Ce projet vise à améliorer la précision du diagnostic du cancer du sein et à accélérer la prise en charge des patientes grâce au Deep Learning (CNN, ResNet) et à l'optimisation via Optuna.

## 📌 1. Contexte du Projet & Problématique

Le cancer du sein est le cancer le plus fréquent chez la femme. Dans près de 80% des cas, il s'agit d'un **Carcinome Canalaire Invasif (IDC)**. Le protocole clinique standard repose sur une biopsie mammaire analysée manuellement par un anatomo-pathologiste. Cette tâche est chronophage, complexe et dépend fortement de l'expertise du spécialiste et de la qualité du matériel.

Dans le cadre d'**Octobre Rose**, ce projet propose un outil d'aide au diagnostic basé sur l'apprentissage profond (*Deep Learning*). L'objectif est d'automatiser le repérage des amas tumoraux dans les tissus cellulaires afin de réduire les délais de prise en charge et de maximiser le taux de survie à 5 ans, qui s'élève à environ **99% lorsque la lésion est détectée de manière précoce**.

---

## 📊 2. Analyse Exploratoire des Données (EDA)

Les modélisations s'appuient sur le jeu de données public **Breast Histopathology Images**.

### Caractéristiques du Dataset :
* **Structure globale :** Le jeu de données comprend un total de 157 572 patchs d'images histologiques issus de **279 patients uniques**.
* **Équilibre parfait :** Le dataset présente une distribution strictement égale entre la **Classe 0 (Sain)** et la **Classe 1 (IDC)** avec exactement **78 786 images** par classe. Cet équilibre est un atout majeur pour l'apprentissage du modèle sans biais algorithmique.
* **Variabilité par patient :** L'analyse révèle une forte disparité dans le nombre de patchs par individu (allant de quelques dizaines à plusieurs centaines), ce qui justifie une attention particulière lors du découpage des données.

### Observations macroscopiques et microscopiques :
1.  **Dimensions des images :** Bien que le format théorique soit de $50 \times 50$ pixels, l'exploration statistique montre la présence de patchs aux dimensions hétérogènes en bordure de lame (ex: $50 \times 13$, $50 \times 48$). Un redimensionnement uniforme est donc appliqué lors du prétraitement.
2.  **Intensité des couleurs & Texture :** * *Classe 0 (Sain) :* Tissus à structure aérée, espaces intercellulaires ou graisse visibles, coloration rose claire prédominante.
    * *Classe 1 (IDC) :* Densité cellulaire nettement supérieure, saturation marquée par des nuances violettes intenses (traduisant la prolifération des noyaux cancéreux). La distribution de l'intensité moyenne des pixels confirme scientifiquement ce décalage chromatique.

---

## 🛠️ 3. Pipeline de Prétraitement & Stratégie de Validation

Pour garantir la robustesse des modèles et éviter tout phénomène de triche ou de surapprentissage (*Data Leakage*), le pipeline suivant a été mis en œuvre :

1.  **Cloisonnement par Patient (`GroupShuffleSplit`) :** La séparation des données préserve strictement l'étanchéité des individus. Aucun patch d'un patient présent dans l'ensemble de test n'est utilisé lors de l'entraînement.
    * **Entraînement :** 122 026 images
    * **Validation :** 17 846 images
    * **Test :** 17 700 images
2.  **Normalisation :** Conversion de l'espace de pixels $[0, 255] \rightarrow [0, 1]$ via un opérateur de remise à l'échelle (`rescale=1./255`) pour stabiliser la descente de gradient.
3.  **Augmentation de données (*Data Augmentation*) :** Application de rotations aléatoires ($20^\circ$), décalages en largeur/hauteur, et retournements horizontaux/verticaux afin de rendre le modèle invariant aux conditions opératoires des laboratoires.
4.  **Redimensionnement :** Forçage systématique des dimensions selon l'architecture cible ($50 \times 50$ ou $100 \times 100$).

---

## 🧠 4. Architectures de Modélisation & Optimisation

Deux stratégies complémentaires de Deep Learning ont été développées et comparées :

### Modèle 1 : CNN Custom (Architecture en Entonnoir)
Conçu spécifiquement pour traiter les patchs natifs de $50 \times 50$ pixels. Il empile des blocs de couches `Conv2D` (avec augmentation progressive des filtres : 32, 64, etc.), de la `BatchNormalization` pour stabiliser l'apprentissage, du `MaxPooling2D` pour la réduction spatiale, et des couches de régularisation `Dropout` pour limiter le surapprentissage.

### Modèle 2 : ResNet50V2 (Transfer Learning)
Exploitation d'un réseau de neurones profond pré-entraîné sur *ImageNet*. Le corps de l'architecture est gelé pour conserver les descripteurs visuels universels, tandis qu'une tête de classification sur mesure (`GlobalAveragePooling2D` + `Dense`) est entraînée sur des patchs interpolés en $100 \times 100$ pixels.

### Optimisation Automatisée avec Optuna
Pour maximiser l'efficacité du pipeline, le framework **Optuna** a été intégré pour orchestrer la recherche des meilleurs hyperparamètres sur un espace défini :
* *Taux d'apprentissage (Learning Rate) :* Exploration logarithmique entre $10^{-4}$ et $10^{-2}$.
* *Taille de batch (Batch Size) :* $[32, 64, 128]$.
* *Régularisation :* Évaluation du taux de Dropout et du Weight Decay.

---

## 📈 5. Résultats & Évaluation des Performances

Conformément aux exigences cliniques, l'évaluation ne repose pas uniquement sur l'exactitude (*Accuracy*), mais intègre des métriques ciblées pour minimiser les faux négatifs (le fait de rater un tissu cancéreux).

### Comparaison des Modèles (Données de Test) :

| Modèle | Précision (Precision) | Rappel (Recall) | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **CNN Custom (Base)** | *[Score %]* | *[Score %]* | *[Score %]* | *[Score %]* |
| **ResNet50V2 (Transfer Learning)** | *[Score %]* | *[Score %]* | *[Score %]* | *[Score %]* |
| **Modèle Final Optimisé (Optuna)** | **84.75 %** | *[Score %]* | *[Score %]* | *[Score %]* |

> 💡 **Choix de la métrique clé :** L'analyse utilise la métrique **PR-AUC (Precision-Recall Area Under Curve)** plutôt que la courbe ROC-AUC classique. Dans un contexte de diagnostic médical, la courbe PR-AUC offre une évaluation beaucoup plus rigoureuse et représentative des performances face à la détection de la classe minoritaire/critique (les cellules cancéreuses).

---

## 🌐 6. Veille Technologique

Le développement de cet outil a été guidé par l'étude des standards de l'état de l'art en imagerie médicale et en apprentissage profond :
* **Classification Multi-niveaux :** Étude des méthodologies d'extraction de caractéristiques géométriques et chromatiques pour l'analyse des tissus mammaires (*Histopathology Images Multi-level Features*).
* **Architectures Residuelles :** Analyse des apports des connexions de saut (*skip connections*) introduites par *He et al.* dans *Deep Residual Learning for Image Recognition*, permettant d'entraîner des réseaux profonds sans évanouissement du gradient.
* **Évolution vers les Transformers :** Veille sur l'adaptation des architectures d'attention au domaine de la vision avec les **Vision Transformers (ViT)** via la documentation de référence Hugging Face, ouvrant des perspectives de performance supérieures pour la capture de contextes globaux sur des lames histologiques complètes.
* **Sources et Références :** Veille technique continue alimentée par les publications scientifiques de plateformes spécialisées (Medium, Towards Data Science, PubMed Central).

---

## 📁 7. Structure du Repository

```text
ruban-rose/
├── data/               # Instructions d'accès au dataset (exclues du suivi Git)
├── notebooks/          # Notebook complet d'EDA, Prétraitement et Modélisation
├── outputs/            # Sauvegardes des meilleurs modèles (.keras) et graphiques
├── .gitignore          # Filtre des fichiers lourds (images, caches)
└── README.md           # Documentation du projet

8. Installation et Utilisation
Prérequis
Le projet est configuré pour s'exécuter dans un environnement Python 3.10+ équipé d'un GPU (recommandé).

Lancement sur Google Colab
Déposez l'archive du dataset archive.zip dans votre dossier Google Drive à l'emplacement /Projet_Ruban_rose/.

Ouvrez le notebook présent dans /notebooks/ sur Google Colab.

Activez l'accélérateur matériel T4 GPU dans les options d'exécution.

Exécutez les cellules séquentiellement. L'extraction des données et l'entraînement se lanceront automatiquement.
