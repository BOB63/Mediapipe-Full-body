
"""
Utilizzo modulo MEDIAPIPE per analisi pose mani , corpo e volto.
Il programma utilizza i dati catturati ed esportati su file csv dal programma 
" Mediapipe Rec & Export.py "

Il file 'Rhand_coords.csv' contine colonna 1 la classe ( due,tre,quattro) e a seguire 
le coordinate x,y,z e visibility dei 21 landmark identificati sulla mano destra.

Tale file sara' poi dato in pasto al programma training_pose.py 
per addestrare i modelli di ML

"""


import pandas as pd
#splitta il dataset di input tra training e test
from sklearn.model_selection import train_test_split 

# genera una pipeline di modelli di ML che verranno addestrati contemporaneamente
# per poi scegliere quello che da lo score piu' alto

from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

# importa 4 modelli di ML ( classificatori ) 
# LogisticRegression, 
# RidgeClassifier, 
# RandomForestClassifier
# GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import os
from pathlib import Path

# --- Path Setup ---
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

input_file = Path(path) / 'full_coords.csv'

print(f"Dati input : {input_file}")
print("Esiste file input ?:", input_file.exists())



df = pd.read_csv(input_file)
#df.head()
#df.tail()
#df[df['class']=='due']
X = df.drop('class', axis=1) # features
y = df['class'] # target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

"""
print('pipeline')
print(list(pipelines.values())[0])
"""


fit_models = {}
for algo, model in pipelines.items():
    model.fit(X_train, y_train)  # <-- Fit the model here
    fit_models[algo] = model     # Ocontiene modelli usati
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
print(y_test)    
#print(fit_models['rf'].predict(X_test))

# il file body_language.pkl contiene il modello di ML scelto 

print('Seleziona modello ML da salvare tra: lr (Logistic), rc (RidgeClassifier), rf (RandomForest), gb (GradientBoosting)')
model_type = input('Modello: ').strip()

# Validate user input
if model_type not in fit_models:
    print(f"'Modello ML  '{model_type}' non riconosciuto. Scegli tra: {', '.join(fit_models.keys())}")
else:
    
    output_file = Path(path) / f"{model_type}_full_language.pkl"
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(fit_models[model_type], f)
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")

    print(f"'Modello '{model_type}' salvato con successo in '{output_file}'")

