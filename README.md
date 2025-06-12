Sequenza Utilizzo files :

1 ) Full MPRec & Export v1.py   per  definire classi e registrare coordinate su file csv

2 ) Full training_pose v1.py    per addestramento modelli ML

3 ) full_decode_pose v1.py      per verifica e decodifica pose



full_coords.csv  ,               file contenente coordinate landmarks generato da Full MPRec & Export v1.py

full_language.pkl  ,             file modello ML addestrato generato da Full training_pose v1.py 

NOTA : con w11 installare modulo scikit-learn al posto di scklearn

python -m pip install scikit-learn --user

Per installazione Python , OpenCV e MediaPipe fare riferimento a rispettivi siti web 

For more info see tutorial https://www.youtube.com/watch?v=We1uB79Ci-w&ab_channel=NicholasRenotte

tutti i miei codici derivano da questo tutorial.
