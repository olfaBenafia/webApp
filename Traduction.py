from googletrans import Translator
from mysql.connector import Error
import mysql.connector
import Levenshtein as lev
import pandas as pd
import numpy as np

def compare_Levenshtein(str1, str2):
    str2 = str(str2)
    sorted_str1 = ''.join(sorted(str1.replace(" ", "").lower()))
    sorted_str2 = ''.join(sorted(str2.replace(" ", "").lower()))
    distance = lev.distance(sorted_str1, sorted_str2)
    similarity = 1 - distance / max(len(sorted_str1), len(sorted_str2))
    return similarity
import pandas as pd
import numpy as np

def traductionlatin(word):
  print(word)
  translator = Translator()
  pronunciation = translator.translate(word, src='ar',dest='ar').pronunciation
  return pronunciation

def traductionfr(word):
  translator = Translator()
  pronunciation=''
  word=word.split()
  for w in word:
      pronunciation+=' '+ translator.translate(w, src='ar',dest='fr').text
  return pronunciation

def traduction_ville(ville_arabe):
    df = pd.read_excel("C:/Users/olfab/Downloads/Place_scrap.xlsx")
    similarite = []
    ville_finale=''
    ville_finale_latin=''
    L_arabe = df['Lieux Arabe'].values
    L_arabe = np.append(L_arabe, df['Gouv_Arabe'])
    L_latin = df['Lieux_Français'].values
    L_latin = np.append(L_latin, df['Gouv_Français'])
    combined_data = pd.DataFrame({'Arabe': L_arabe, 'Latin': L_latin})
    liste_ville_arabe=ville_arabe.split()
    for mot in liste_ville_arabe:
        for index, row in combined_data.iterrows():
            ville = row['Arabe']
            ville2 = row['Latin']
            ville = str(ville)
            liste_ville = ville.split()
            liste_ville_latin = ville2.split()
            i = 0
            if i < len(liste_ville):
                similarity_score = compare_Levenshtein(mot, liste_ville[i])
                if similarity_score == 1.0:
                    if liste_ville[i] not in ville_finale:
                        ville_finale += ' ' + liste_ville[i]
                        if liste_ville_latin[i] == 'Gouv_':
                            ville_finale_latin += ' ' + liste_ville_latin[i + 1]
                        else:
                            ville_finale_latin += ' ' + liste_ville_latin[i]
                    break
                i += 1
    if ville_finale == '':
        for ville in combined_data['Arabe']:
            similarity_score = compare_Levenshtein(ville_arabe, ville)
            similarite.append(similarity_score)
        maximum = max(similarite)
        print(maximum)
        indices = []
        if maximum>=0.9:
            for i in range(len(similarite)):
                if similarite[i] == maximum:
                    indices.append(i)
            if 'Gouv_' in L_latin[indices[0]]:
                ville_latin = L_latin[indices[0]].replace('Gouv_', '')
            else:
                ville_latin = L_latin[indices[0]]
            return ville_latin
        else:
            return traductionlatin(ville_arabe)
    else:
        return ville_finale_latin

def latin_ville(d, info):
    if info == 'recto':
        d['place_birth'] = traduction_ville(d['place_birth'])
    else:
        d['ville1'] = traduction_ville(d['ville1'])
        d['ville2'] = traduction_ville(d['ville2'])
        return d


def Traductionfinale(d, info):
    final = []
    for key, value in d.items():
        if value != ' ' and value != '':
            if key == 'birth' or key == 'date_deleb' or key == 'profession':
                trad = traductionfr(value)
                print(value, trad)
            elif key == 'place_birth' or key == 'ville1' or key == 'ville2':
                trad = traduction_ville(value)
            else:
                trad = traductionlatin(value)
                # Fix: Check if trad is None before accessing indices
                if trad is not None and len(trad) >= 2:
                    if trad[0] == 'e' and trad[1] == 'a':
                        trad = trad[0].replace("e", "", 1) + trad[1:]
                elif trad is None:
                    # Fallback: use original value if translation fails
                    print(f"[WARNING] Translation failed for {key}: {value}, using original")
                    trad = value
        else:
            trad = ' '

        # Additional safety check before capitalize
        if trad is not None:
            trad = trad.capitalize()
        else:
            trad = ' '

        final.append(trad)

    print(final)
    if info == 'recto':
        L = ['cin', 'name', 'sur_name', 'sexe', 'father_name', 'Grand_Father_Name', 'Husband_Name', 'birth',
             'place_birth']
    else:
        L = ['nom_mere', 'profession', 'ville1', 'ville2', 'date_deleb', 'ref', 'qr_code']

    dico = dict()
    for i in range(len(L)):
        dico[L[i]] = final[i]
    return dico

def inserer_BD(dict_traduit,upload_time,info):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="cin"
    )
    print('connexion etablie')
    cursor = conn.cursor()
    if info == 'recto':
        sql = "INSERT INTO cin_recto (cin, name, sur_name, sexe, father_name, Grand_Father_Name, Husband_Name, birth, place_birth,upload_time) VALUES (%s, %s, %s,%s, %s, %s,%s, %s, %s,%s)"
        cursor.execute(sql, (dict_traduit['cin'], dict_traduit['name'], dict_traduit['sur_name'], dict_traduit['sexe'], dict_traduit['father_name'] , dict_traduit['Grand_Father_Name'], dict_traduit['Husband_Name'] , dict_traduit['birth'], dict_traduit['place_birth'],upload_time))
    if info == 'verso':
        print('f wesstou',dict_traduit)
        sql = "INSERT INTO cin_verso (nom_mere, profession, ville1, ville2, date_deleb, ref,upload_time) VALUES (%s,%s, %s, %s,%s, %s,%s)"
        cursor.execute(sql, (dict_traduit['nom_mere'], dict_traduit['profession'], dict_traduit['ville1'], dict_traduit['ville2'],dict_traduit['date_deleb'], dict_traduit['ref'],upload_time))
    conn.commit()
    cursor.close()
    conn.close()

'''d_verso = compare_ocrs("C:/Users/rabeb/Digitexe/Appweb1/static/upload/00a9aaa6-55d0-4aef-a5ab-31414746546e.jpg", 'verso')
d_final_verso = Traductionfinale(d_verso, 'verso')
inserer_BD(d_final_verso,'verso')'''
#print(rotation_image_boucle("C:/Users/rabeb/Digitexe/Appweb1/static/upload/26c79fda-5757-4dca-be63-17e900f0da2a.jpeg",'verso'))