import os
import requests
url = 'http://localhost:8300/api/new_image/'

def upload_image(image_path):
    with open(image_path, 'rb') as img:
        name_img= os.path.basename(image_path)
        files= {'file': (name_img,img,'multipart/form-data',{'Expires': '0'}) }
        with requests.Session() as s:
            r = s.post(url,files=files)
            print(r.status_code)





def get_faces_from_directory(folder_path):
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for image in images:
        image = folder_path + "/" + image
        upload_image(image)

    # exit()

get_faces_from_directory("/jupyter-notebooks/input_images_old")