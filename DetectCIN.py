import cv2
import math
from PIL import Image
from IPython import display
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import supervision as sv
from ArabicOcr import arabicocr
import numpy as np
import easyocr
import paddle
from paddleocr import PaddleOCR, draw_ocr
from pygrametl.datasources import CSVSource




model_recto = YOLO('C:\\Users\\olfab\\pfe\\AppwebV1\\AppwebV1\\static\\models\\best_RectoCIN.pt')
model_verso = YOLO('C:\\Users\\olfab\\pfe\\AppwebV1\\AppwebV1\\static\\models\\best_VersoCIN.pt')

def modeling(image_path,info) :
    if info == 'recto':
        results = model_recto(image_path)
    else:
        results = model_verso(image_path)
    detections = sv.Detections.from_ultralytics(results[0])
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(scene=cv2.imread(image_path),detections=detections)
    return(detections)
    if not results or results[0] is None:
        print("Erreur : aucun résultat de détection.")
        return None

def calcul_angle(detections) :
  s=0
  if len(detections.xyxy)>0:
    for i in range (len(detections.xyxy)):
      coords = detections.xyxy[i]
      angle = math.degrees(math.atan2(coords[3] - coords[1], coords[2] - coords[0]))
      s+=angle
    angle=s/len(detections.xyxy)
  else:
    angle=0
  return(angle)


def resize_image(image_path):
  image = cv2.imread(image_path)
  new_size = (max(image.shape[0], image.shape[1]), min(image.shape[0], image.shape[1]))
  resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
  return(resized_image,new_size)


def rotate_image(image_path, angle):
  image = cv2.imread(image_path)
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  angle_radians = math.radians(angle)
  rotation_matrix = cv2.getRotationMatrix2D(center, angle,1)
  new_width = int(h * abs(math.sin(angle_radians)) + w * abs(math.cos(angle_radians)))
  new_height = int(h * abs(math.cos(angle_radians)) + w * abs(math.sin(angle_radians)))
  rotation_matrix[0, 2] += (new_width / 2) - center[0]
  rotation_matrix[1, 2] += (new_height / 2) - center[1]
  rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
  return rotated_image


def rotation_image_boucle(image_path,info):
  list_angle=[]
  list_image=[]
  i=1
  a=True
  angle_deg=True
  left=a*angle_deg

  while left==1 :
      image_path = inverse_image(image_path,info)
      detections=modeling(image_path,info)
      angle=calcul_angle(detections)
      print(angle)

      list_image.append(image_path)
      list_angle.append(angle)

      image = cv2.imread(image_path)

      for ang in list_angle:
        if angle>ang:
          a=False

      if angle<25:
        angle_deg=False

      left=a*angle_deg

      if left==0:
        if len(list_angle)==1 :
         angle=list_angle[-1]
        else :
          angle=list_angle[-2]
        image_path=list_image[-1]
        break
      if angle >= 70:
          rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      else:
          rotated_image = rotate_image(image_path, -angle + 5)
      image_array = np.array(rotated_image)  # Assurez-vous que image_rotated est votre tableau numpy
      image = Image.fromarray(image_array, 'RGB')
      imsave='image'+str(i)+'.png'
      image.save(imsave)

      image_path='image'+str(i)+'.png'
      i+= 1

  return(angle,image_path)

def inverse_image(image_path,info):
    image = cv2.imread(image_path)
    detections=modeling(image_path,info)
    somme = 0
    for element in detections.confidence:
        somme += element
    if len(detections.confidence)>0:
      moyenne = somme / len(detections.confidence)
    else :
      moyenne=0
    if moyenne < 0.3 or len(detections.confidence)<6:
      image_rotated = cv2.rotate(image, cv2.ROTATE_180)
    else :
      image_rotated=image
    image_array = np.array(image_rotated)
    image = Image.fromarray(image_array, 'RGB')
    if info=='recto':
        imsave='image_rotated_recto.png'
        image_path = 'image_rotated_recto.png'
    else :
        imsave='image_rotated_verso.png'
        image_path = 'image_rotated_verso.png'
    image.save(imsave)
    return image_path


# Fonction pour effectuer la détection sur une image
def detect_image_arabicocr(img_path,info):
    angle, image_path = rotation_image_boucle(img_path,info)
    # Chargez l'image
    img = cv2.imread(image_path)
    img_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convertissez l'image en format PIL (Pillow)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if info == 'recto':
        results = model_recto(image_path)
    else:
        results = model_verso(image_path)
    detections = sv.Detections.from_ultralytics(results[0])
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(scene=cv2.imread(image_path),detections=detections )
    img_path = 'C:/Users/rabeb/Digitexe/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, annotated_frame)
    i=0
    lines=[]
    for seg in detections.xyxy:
      i+=1
      converted_seg = [int(item) for item in seg]
      x_min, y_min, x_max, y_max = converted_seg
      segment = img[y_min:y_max, x_min:x_max]
      segment_path='C:/Users/olfab/pfe/Appweb1/Appweb1/Links/'+str(i)+'.png'
      cv2.imwrite(segment_path, segment)

      out_image = 'C:/Users/olfab/pfe/Appweb1/Appweb1/Links/out'+str(i)+'.png'
      results = arabicocr.arabic_ocr(segment_path, out_image)
      results_reversed=[]
      for item in results:
          word = item[1]  # Accéder au mot
          reversed_word = word[::-1]  # Inverser le mot
          results_reversed.append(reversed_word)
      lines.append(results_reversed)
      print(lines)
    list_of_strings = [' '.join(sublist) for sublist in lines]
    result_string = '\n'.join(list_of_strings)

    return (result_string)



# Fonction pour effectuer la détection sur une image
def detect_image_easyocr(img_path,info):
    reader = easyocr.Reader(['ar','en'])
    angle, image_path = rotation_image_boucle(img_path,info)
    # Chargez l'image
    img = cv2.imread(image_path)
    img_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convertissez l'image en format PIL (Pillow)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if info == 'recto':
        results = model_recto(image_path)
    else:
        results = model_verso(image_path)
    detections = sv.Detections.from_ultralytics(results[0])
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(scene=cv2.imread(image_path), detections=detections)
    img_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, annotated_frame)
    i = 0
    lines = []
    scores=[]
    for seg in detections.xyxy:
        i += 1
        converted_seg = [int(item) for item in seg]
        x_min, y_min, x_max, y_max = converted_seg
        segment = img[y_min:y_max, x_min:x_max]
        segment_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/Links/' + str(i) + '.png'
        cv2.imwrite(segment_path, segment)

        out_image = 'C:/Users/olfab/pfe/Appweb1/Appweb1/Links/out' + str(i) + '.png'

        results_ocr = reader.readtext(segment)
        somme = 0
        text = ''
        print(results_ocr)
        if results_ocr!=[]:
            for j in range(len(results_ocr)):
                text += ' ' + results_ocr[j][1]
                somme += float(results_ocr[j][2])
            moy = somme / len(results_ocr)
        else:
            text = ' Not detected by ocr'
            moy = 0
        lines.append(text)
        scores.append(moy)
    if info == 'recto':
        L = ['cin', 'name', 'sur_name', 'father_name', 'birth', 'place_birth']
    else:
        L = ['nom_mere', 'profession', 'ville1', 'ville2', 'date_deleb', 'ref', 'qr_code']
    dico = dict()
    # Prendre le max score + ordonner les etiquettes
    for item in L :
      confidence=[]
      position=np.where(detections.data['class_name'] == item)
      print('position',position)
      if len(position[0])>=1:
        for i in position[0][::-1]:
          print('position',detections.data['class_name'][i],detections.confidence[i])
          confidence.append(detections.confidence[i])
        max_score=min(confidence)
        indices=np.where(confidence == max_score)
        ind=indices[0][0]
        dico[item] = [lines[position[0][ind]], scores[position[0][ind]]]
      else:
        dico[item] = ['Not detected', 0]
    return dico

def Decoupage_information (d):
    dict1=d.copy()
    dict2 = dict1.copy()
    print(dict2)
    for key,value in dict1.items():
        if key == 'sur_name':
            if  'بنت' in value:
                words = value.split('بنت')
                dict2['sur_name']=words[0]
                if 'بن' in words[1]:
                    liste_fathers= words[1].split('بن')
                    name_father = liste_fathers[0]
                    dict2['Grand_Father_Name'] = liste_fathers[1]
                else :
                    name_father = words[1]
                    dict2['Grand_Father_Name'] = 'Null'
                dict2['sexe'] = 'Femme'
                liste = dict2['father_name']
                liste = liste.split()
                if liste[0] == 'حرم' or liste[0] == 'رم' or liste[0] == 'م':
                    dict2["Husband_Name"] = ' '.join(liste[1:])
                else:
                    dict2['Husband_Name'] = dict2['father_name']
                dict2['father_name'] = name_father
                liste_keys= ['cin','name','sur_name','sexe','father_name','Grand_Father_Name','Husband_Name','birth','place_birth']
                sorted_keys = [key for key in liste_keys if key in dict2]
                sorted_dict = {key: dict2[key] for key in sorted_keys}
                print(sorted_dict)
                return sorted_dict

        elif key == 'father_name':
            dict2['Husband_Name'] = 'NULL'
            print('value',value)
            words = value.split()
            if words[0] == 'بن' or words[0] == 'ىن' or words[0] == 'ن' or len(words[0])<=2 :
                dict2['sexe']='Homme'
                liste_fathers = value.split('بن')
                name_father = liste_fathers[0]
                dict2['Grand_Father_Name'] = liste_fathers[1]
            elif words[0]=='بنت' or words[0]=='نت' or words[0]=='ت':
                dict2['sexe']='Femme'
                words = value.split()
                words.remove(words[0])
                resultat = ' '.join(words)
                liste_fathers = resultat.split('بن')
                name_father = liste_fathers[0]
                dict2['Grand_Father_Name'] = liste_fathers[1]
            else:
                dict2['sexe']='Not Detected by OCR'
                liste_fathers = value.split('بن')
                name_father = liste_fathers[0]
                dict2['Grand_Father_Name'] = liste_fathers[1]
            dict2['father_name'] = name_father
            liste_keys = ['cin', 'name', 'sur_name', 'sexe', 'father_name', 'Grand_Father_Name', 'Husband_Name','birth', 'place_birth']
            sorted_keys = [key for key in liste_keys if key in dict2]
            sorted_dict = {key: dict2[key] for key in sorted_keys}
            print(sorted_dict)
            return sorted_dict

def detect_image_paddle(img_path,info):
    ocr = PaddleOCR(use_angle_cls=True, lang='ar')
    angle, image_path = rotation_image_boucle( img_path,info)
    # Chargez l'image
    img = cv2.imread(image_path)
    img_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convertissez l'image en format PIL (Pillow)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if info == 'recto':
        results = model_recto(image_path)
    else:
        results = model_verso(image_path)
    detections = sv.Detections.from_ultralytics(results[0])
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(scene=cv2.imread(image_path), detections=detections)
    img_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, annotated_frame)
    i = 0
    lines = []
    scores=[]
    for seg in detections.xyxy:
        i += 1
        converted_seg = [int(item) for item in seg]
        x_min, y_min, x_max, y_max = converted_seg
        segment = img[y_min:y_max, x_min:x_max]
        segment_path = 'C:/Users/olfab/pfe/Appweb1/Appweb1/Links/' + str(i) + '.png'
        cv2.imwrite(segment_path, segment)
        results_ocr = ocr.ocr(segment_path, cls=True)
        text=''
        somme=0
        if results_ocr!=[None]:
            for j in range(len(results_ocr[0])):
                print(results_ocr[0][j][1][0])
                text+=' '+results_ocr[0][j][1][0]
                somme+=float(results_ocr[0][j][1][1])
            text=text[::-1]
            moy=somme/len(results_ocr[0])
        else:
            text = ' Not detected by ocr'
            moy=0
        lines.append(text)
        scores.append(moy)
    if info == 'recto':
        L = ['cin', 'name', 'sur_name', 'father_name', 'birth', 'place_birth']
    else:
        L = ['nom_mere', 'profession', 'ville1', 'ville2', 'date_deleb', 'ref','qr_code']
    dico = dict()
    # Prendre le max score + ordonner les etiquettes
    for item in L:
        confidence = []
        position = np.where(detections.data['class_name'] == item)
        if len(position[0]) >= 1:
            for i in position[0]:
                confidence.append(detections.confidence[i])
            max_score = max(confidence)
            indices = np.where(confidence == max_score)
            ind = indices[0][0]
            dico[item] = [lines[position[0][ind]],scores[position[0][ind]]]
        else:
            dico[item] = ['Not detected',0]
    return dico

def compare_ocrs(path_save,info):
    dict2 = detect_image_easyocr(path_save,info)
    '''dict1 = detect_image_paddle(path_save,info)
    d = dict()
    for key in dict1.keys():
        if key=='birth':
            res = dict2[key][0]
        else:
            maxi = max(dict1[key][1], dict2[key][1])
            print(maxi)
            if maxi == dict1[key][1]:
                res = dict1[key][0]
            else:
                res = dict2[key][0]
        d[key] = res'''
    dict3 = dict()
    for key in dict2.keys():
        dict3[key] = dict2[key][0]
    print('dict1:',dict3)
    print('dict2: ',dict2)
    return(dict3)

import csv
def Traduction():

    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)


import Levenshtein


def comparer_resultats_ocr(resultat1, resultat2):
    distance1 = Levenshtein.distance(resultat1, resultat2)
    print(distance1)
    if distance1 < 15:
        return resultat1
    else:
        return resultat2
import difflib


def compare_ocr(ocr1_result, ocr2_result):
  """Compare deux résultats OCR et retourne le meilleur."""
  # Convertir les résultats en minuscules et supprimer les espaces inutiles
  ocr1_result = ocr1_result.lower().strip()
  ocr2_result = ocr2_result.lower().strip()

  # Calculer la similarité entre les deux résultats
  similarity = difflib.SequenceMatcher(None, ocr1_result, ocr2_result).ratio()
  print(similarity)
  # Si la similarité est supérieure à un seuil prédéfini, conserver le résultat le plus long
  if similarity > 0.8:
    if len(ocr1_result) > len(ocr2_result):
      return ocr1_result
    else:
      return ocr2_result
  else:
    # Si la similarité est inférieure au seuil, retourner le résultat le plus long
    if len(ocr1_result) > len(ocr2_result):
      return ocr1_result
    else:
      return ocr2_result





