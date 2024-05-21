# Ollama

## Qu'est-ce qu ollama
Ollama est une application développée en Go ( Golang, est un langage de programmation open-source développé par Google ), basée sur llama.cpp, qui simplifie l'utilisation des modèles de langage (LLM) sur une machine locale.

**Facilité d'utilisation** : Ollama rend l'acquisition et l'exécution des modèles LLM très simples. Il adapte automatiquement les prompts (les questions ou les textes d'entrée) au format attendu par chaque modèle et charge les modèles à la demande avec des commandes très simples.

**API REST** : Ollama offre une API REST qui permet d'exécuter et d'interagir avec les modèles LLM. Cela signifie que vous pouvez facilement intégrer Ollama dans vos applications en envoyant des requêtes HTTP pour obtenir des réponses des modèles.

**Compatibilité multiplateforme** : Ollama fonctionne en tâche de fond et est disponible sur Mac OS, Linux et, récemment, sur Windows en version preview. De plus, depuis octobre 2023, Ollama est intégré dans une image Docker, facilitant son utilisation sur Mac OS et Linux.

**Avantages de l'exécution locale** :
- Contrôle des coûts : Pas besoin de payer pour des services de cloud computing coûteux.
- Sécurité des données : Les données restent sur votre machine, ce qui est crucial pour la confidentialité.

**Commandes simples** : Vous pouvez télécharger un modèle avec la commande ollama pull mistral, puis le lancer avec ollama run mistral. Une fois lancé, vous pouvez interagir avec le modèle via l'API REST.

**Large choix de modèles** : Ollama supporte plus de 60 modèles différents, y compris les populaires Llama 2, Mistral, Mixtral et Gemma.

**Efficacité des ressources** : Ollama utilise efficacement la mémoire RAM pour exécuter les modèles. Par exemple, un modèle 7B nécessite 8 GB de RAM, un modèle 13B nécessite 16 GB de RAM, et un modèle 33B nécessite 32 GB de RAM.

En résumé, Ollama est une solution pratique et efficace pour utiliser des modèles LLM en local, offrant simplicité, contrôle et sécurité.

## Point important

### LangChain
LangChain est un cadre open source conçu pour faciliter la création d'applications basées sur de grands modèles de langage (LLM). Voici un aperçu de ses fonctionnalités et avantages :

**Qu'est-ce qu'un LLM ?** : Les LLM sont des modèles de deep learning de grande taille, préentraînés sur d'énormes quantités de données. Ils peuvent générer des réponses aux requêtes des utilisateurs, comme répondre à des questions ou créer des images à partir de descriptions textuelles.

**Outils et abstractions** : LangChain offre des outils et des abstractions pour améliorer la personnalisation, la précision et la pertinence des informations générées par les modèles. Cela permet aux développeurs de créer des applications plus sophistiquées et adaptées aux besoins spécifiques des utilisateurs.

**Création et personnalisation** : Avec LangChain, les développeurs peuvent facilement créer de nouvelles chaînes d'instructions ou personnaliser des modèles existants. Cela permet de tirer le meilleur parti des capacités des LLM pour diverses applications.

**Accès à de nouveaux jeux de données** : LangChain inclut des composants qui permettent aux LLM d'accéder à de nouveaux jeux de données sans nécessiter un nouvel entraînement. Cela signifie que les modèles peuvent utiliser des informations actualisées ou spécifiques sans devoir être réentraînés, ce qui économise du temps et des ressources.

En résumé, LangChain est un outil puissant et flexible pour développer des applications basées sur de grands modèles de langage, offrant des moyens innovants de personnaliser et d'améliorer les performances des modèles sans nécessiter de réentraînement complet.
### Embeddings
Les embeddings sont une technique utilisée en intelligence artificielle et en traitement du langage naturel pour représenter des données sous forme de vecteurs de nombres réels dans un espace de dimension inférieure. Voici une explication détaillée de ce concept :

Définition :
Les embeddings sont des représentations vectorielles continues des objets (comme des mots, des phrases ou des images) qui préservent les relations sémantiques ou contextuelles entre eux.

**Pourquoi les Embeddings ?** :
Réduction de la Dimensionnalité : Les embeddings permettent de représenter des données de haute dimension (comme des mots dans un vocabulaire) dans un espace de dimension inférieure, ce qui facilite le traitement et le calcul.
Capturer les Relations Sémantiques : Les vecteurs d'embeddings sont construits de manière à ce que des objets similaires soient proches les uns des autres dans l'espace vectoriel. Par exemple, les mots "roi" et "reine" auront des vecteurs similaires.

**Applications des Embeddings** :
- Traitement du Langage Naturel (NLP) : Les embeddings de mots (word embeddings) comme Word2Vec, GloVe, et les embeddings de phrase comme ceux générés par BERT, capturent les relations contextuelles entre les mots et les phrases.
- Représentation d'Images : Dans la vision par ordinateur, les embeddings d'images permettent de représenter des images de manière compacte tout en conservant les informations visuelles importantes.
- Recommandation : Les embeddings sont utilisés pour créer des systèmes de recommandation en capturant les similarités entre les utilisateurs et les produits.

**Types d'Embeddings** :
- Word Embeddings : Représentations vectorielles des mots. Exemples : Word2Vec, GloVe.
- Contextual Embeddings : Représentations des mots en tenant compte de leur contexte. Exemples : BERT, GPT.
- Document Embeddings : Représentations des phrases, paragraphes ou documents. Exemples : Doc2Vec, Sentence-BERT.

**Exemple Concret** :
Supposons que nous avons trois mots : "chat", "chien" et "voiture". Dans un espace de mots traditionnel (one-hot encoding), ces mots seraient représentés comme des vecteurs orthogonaux sans lien entre eux. En utilisant des embeddings, "chat" et "chien" auraient des vecteurs proches, car ils sont sémantiquement liés (tous deux sont des animaux), tandis que "voiture" serait plus éloigné.

**Comment sont-ils Appris** ? :
Les embeddings sont généralement appris à partir de grandes quantités de données à l'aide de réseaux de neurones. Par exemple, Word2Vec utilise un réseau de neurones pour prédire les mots environnants d'un mot cible (Skip-gram) ou pour prédire un mot cible à partir de ses mots environnants (CBOW).

En résumé, les embeddings sont des outils puissants en intelligence artificielle qui permettent de représenter des données complexes de manière compacte et significative, facilitant ainsi leur utilisation dans divers algorithmes et applications.

### Document Loader
Le Document Loader en Python est généralement utilisé pour charger et traiter des documents de différentes sources et formats. Il peut être une partie intégrante de bibliothèques ou de frameworks qui manipulent des données textuelles. Voici un aperçu général de son fonctionnement, souvent basé sur l'utilisation de bibliothèques comme LangChain, PyPDF2, Pandas, et autres.

### Chunking
Qu'est-ce que le Chunking ?

Le chunking est une technique en traitement du langage naturel (NLP) qui consiste à diviser un texte en segments plus petits et significatifs appelés chunks. Ces chunks sont souvent des groupes de mots qui forment des unités syntaxiques cohérentes, comme des syntagmes nominaux ou verbaux.

#### Chunking Ids
Chaque chunk peut être identifié par un ID de chunk, qui sert à référencer ou à étiqueter ces segments de manière unique et cohérente.

#### Chunking a la BD (SQLlite)

## Comment l'utiliser

## Chroma
Chroma est une bibliothèque ou un outil souvent utilisé en traitement du langage naturel (NLP) et en machine learning pour la gestion et la manipulation de vecteurs d'embeddings. Les embeddings sont des représentations vectorielles de données (comme des mots, des phrases, ou des documents) qui capturent les relations sémantiques entre ces données.

## Qu'est-ce qu'un rag?
RAG (Retrieval-Augmented Generation) est une technique en traitement du langage naturel (NLP) qui combine la génération de texte avec la récupération d'informations à partir d'une base de données ou d'une source de connaissances externe. Cette méthode améliore la qualité et la précision des réponses générées par des modèles de langage.

Le RAG avec Ollama permet de créer des systèmes de question-réponse puissants et précis en combinant la récupération d'informations pertinentes avec la génération de texte enrichi par des modèles de langage. Cette technique tire parti des forces de chaque composant pour fournir des réponses de haute qualité et pertinentes, en utilisant efficacement les capacités locales de Ollama pour exécuter des modèles LLM.
