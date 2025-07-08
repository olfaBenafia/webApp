import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import cv2
from DetectCIN import rotation_image_boucle, model_recto, model_verso
from PIL import Image
import supervision as sv
import easyocr


def detect_image_paddle(img_path,info):
    ocr = PaddleOCR(use_angle_cls=True, lang='ar')
    angle, image_path = rotation_image_boucle( img_path,info)
    # Chargez l'image
    img = cv2.imread(image_path)
    img_path = 'C:/Users/olfab/pfe/AppwebV1/AppwebV1/static/predict/' + image_path + '.png'
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
    if info == 'recto':
        position1 = np.where(detections.data['class_name'] == 'mat')
    img_path = 'C:/Users/olfab/pfe/AppwebV1/AppwebV1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, annotated_frame)
    i = 0
    lines = []
    scores=[]
    for seg in detections.xyxy:
        i += 1
        converted_seg = [int(item) for item in seg]
        x_min, y_min, x_max, y_max = converted_seg
        segment = img[y_min:y_max, x_min:x_max]
        segment_path = 'C:/Users/olfab/pfe/AppwebV1/AppwebV1/Links/' + str(i) + '.png'
        print('i',i,'position',position1[0][0])
        if i-1==position1[0][0]:
            print('position:  ',detections.data['class_name'][position1[0][0]])
            cv2.imwrite(segment_path, segment)
            seg_image = cv2.imread(segment_path)
            seg_rotated = cv2.rotate(seg_image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(segment_path, seg_rotated)

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
        L = ['adresse','carte_grise_verif','cin','constructeur','dpmc','genre','mat','nom','num_serie','style','type_com','type_const']
    else:
        L = ['nom_mere', 'profession', 'ville1', 'ville2', 'date_deleb', 'ref','qr_code']
    dico = dict()
    # Prendre le max score + ordonner les etiquettes
    for item in L:
        confidence = []
        position = np.where(detections.data['class_name'] == item)
        if item != 'carte_grise_verif':
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


# Fonction pour effectuer la détection sur une image
def detect_image_easyocr(img_path,info):
    reader = easyocr.Reader(['ar','en'])
    angle, image_path = rotation_image_boucle(img_path,info)
    # Chargez l'image
    img = cv2.imread(image_path)
    img_path = 'C:/Users/rabeb/Digitexe/Appweb1/static/predict/' + image_path + '.png'
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
    if info == 'recto':
        position = np.where(detections.data['class_name'] == 'mat')
    img_path = 'C:/Users/olfab/pfe/AppwebV1/AppwebV1/static/predict/' + image_path + '.png'
    cv2.imwrite(img_path, annotated_frame)
    i = 0
    lines = []
    scores=[]
    for seg in detections.xyxy:
        i += 1
        converted_seg = [int(item) for item in seg]
        x_min, y_min, x_max, y_max = converted_seg
        segment = img[y_min:y_max, x_min:x_max]
        segment_path = 'C:/Users/olfab/pfe/AppwebV1/AppwebV1/Links/' + str(i) + '.png'
        cv2.imwrite(segment_path, segment)
        results_ocr = reader.readtext(segment)

        if info == 'recto':
            print('i', i, 'position', position[0][0])
            if i - 1 == position[0][0]:
                print('position:  ', detections.data['class_name'][position[0][0]])
                cv2.imwrite(segment_path, segment)
                seg_image = cv2.imread(segment_path)
                seg_rotated = cv2.rotate(seg_image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(segment_path, seg_rotated)
                results_ocr = reader.readtext(seg_rotated)
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
        L = ['adresse','carte_grise_verif','cin','constructeur','dpmc','genre','mat','nom','num_serie','style','type_com','type_const']
        dico = dict()
        # Prendre le max score + ordonner les etiquettes
        for item in L:
            confidence = []
            position = np.where(detections.data['class_name'] == item)
            if item != 'carte_grise_verif':
                if len(position[0]) >= 1:
                    for i in position[0]:
                        confidence.append(detections.confidence[i])
                    max_score = max(confidence)
                    indices = np.where(confidence == max_score)
                    ind = indices[0][0]
                    dico[item] = [lines[position[0][ind]], scores[position[0][ind]]]
                else:
                    dico[item] = ['Not detected', 0]
    else:
        L = ['CG_verso','carr','cu','cy','date','date_Cre','det','en','ip','ne','np','pf','pt','pv','qc','type']
        dico = dict()
        # Prendre le max score + ordonner les etiquettes
        for item in L:
            confidence = []
            position = np.where(detections.data['class_name'] == item)
            if item != 'CG_verso':
                if len(position[0]) >= 1:
                    for i in position[0]:
                        confidence.append(detections.confidence[i])
                    max_score = max(confidence)
                    indices = np.where(confidence == max_score)
                    ind = indices[0][0]
                    dico[item] = [lines[position[0][ind]], scores[position[0][ind]]]
                else:
                    dico[item] = ['Not detected', 0]
    return dico
path = "C:/Users/olfab/pfe/AppwebV1/AppwebV1/static/upload/03f90046-4f2f-42b1-9766-e403db69201c.jpg"
dict2 = detect_image_easyocr(path,'recto')
print(dict2)
