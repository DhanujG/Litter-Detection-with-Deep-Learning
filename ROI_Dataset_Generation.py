
import json
import simplejson
import pandas as pds  
import numpy
import numpy as np
from PIL import Image
import math
import random

#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def export_modified_json_dataset(json_file, output_json = "", export_json = False, num_files = 500):
    
    filename = open(json_file,)

    annot= json.load(filename)

    #total_images = 1499
    total_images = num_files


    #print(annotations_json.keys())
    #print(len(annot['scene_annotations']))

    #dict_format = {'image_id': 0,  'flickr_url': '', 'filename': '', 'og_width': 0, 'og_height': 0, 'scene_id': [], 'bboxes':[]}

    custom_dataset = []

    for cur_id in range(0, total_images):

        #print(cur_id)

        temp = {'image_id': 0,  'image_url': '', 'filename': '', 'og_width': 0, 'og_height': 0, 'scene_id': [], 'bboxes':[]}

        temp['image_id'] = cur_id

        for element1 in (elementb for elementb in annot["images"] if elementb["id"] == cur_id):

            temp['image_url'] = element1["flickr_url"]
            temp['filename'] = element1['file_name']
            temp['og_width'] = element1['width']
            temp['og_height'] = element1['height']
        
        for element2 in annot["annotations"]:

            if element2["image_id"] == cur_id:
                temp['bboxes'].append(element2["bbox"])

        for element3 in (elementb for elementb in annot['scene_annotations'] if elementb["image_id"] == cur_id):

             settemp= set(element3["background_ids"])
             temp['scene_id'] = list(settemp)
            

        
        #print(temp)
        custom_dataset.append(temp)

    

    scene_background_ids = [{"id": 0, "name": "Clean"}, {"id": 1, "name": "Indoor, Man-made"}, {"id": 2, "name": "Pavement"}, {"id": 3, "name": "Sand, Dirt, Pebbles"}, {"id": 4, "name": "Trash"}, {"id": 5, "name": "Vegetation"}, {"id": 6, "name": "Water"}]


    

    custom_dataset_json = {"data_subset": custom_dataset}


    if (export_json == True):
        print("outputting...")
        fout = open(output_json,"w")
        fout.write(simplejson.dumps(custom_dataset_json, indent = 4, sort_keys = True))
        fout.close()


    return custom_dataset



def create_custom_boxes(trash_image_annot, output_json = "", export_json = False):

    final_dataset = {'image_ids' : [] , 'boundry_box' : [], 'trash' : [], 'environment' : []}

    for element in trash_image_annot:

        #obtain original dimensions
        width = element['og_width']
        height = element['og_height']
        environment_ids = element['scene_id']

        #      "bbox": [x,y,width,height]
        #add bbox images to data for positive trash
        for box in element['bboxes']:

            x = math.floor(box[0])
            y = math.floor(box[1])
            widthb = box[2]
            heightb = box[3]

            #print(box)

            
            
            #make ROI boxes square
            if (widthb < heightb):
                if (((x + math.ceil(widthb / 2) + (heightb - widthb)) < width)) and (((x - math.ceil(widthb / 2) - (heightb - widthb)) > 0)):
                    widthb = heightb

            if (widthb > heightb):
                if (((y + math.ceil(heightb / 2) + (widthb -heightb )) < height)) and (((y - math.ceil(heightb / 2) - (widthb - heightb)) > 0)):
                    heightb = widthb


            overlaps = False

            dimension_increaser = 0

            while(overlaps == False):
                
                temp = dimension_increaser + 1

                

                if (x + temp + math.ceil(widthb / 2)) > width or (x - temp - math.ceil(widthb / 2)) < 0 or (y - temp - math.ceil(heightb / 2)) < 0 or (y + temp + math.ceil(heightb / 2)) > height:
                    overlaps = True
                    break

                for new_box in element['bboxes']:
                    new_x = new_box[0]
                    new_y = new_box[1]
                    new_widthb = new_box[2]
                    new_heightb = new_box[3]

                    new_x1 = new_box[0] - math.ceil(new_box[2]/2)
                    new_x2 = new_box[0] + math.ceil(new_box[2]/2)
                    new_y1 = new_box[1] - math.ceil(new_box[3]/2)
                    new_y2 = new_box[1] + math.ceil(new_box[3]/2)

                    adj_x2 = (x + temp + math.ceil(widthb / 2))
                    adj_x1 = (x - temp - math.ceil(widthb / 2))
                    adj_y1 = (y - temp - math.ceil(heightb / 2))
                    adj_y2 = (y + temp + math.ceil(heightb / 2))

                    if (new_x != x) and (new_y != y):

                        if ((new_x1 < adj_x1 < new_x2) or (new_x1 < adj_x2 < new_x2)) and ((new_y1 < adj_y1 < new_y2) or (new_y1 < adj_y2 < new_y2)):

                            overlaps = True
                            break

                        if ((adj_x2 - adj_x1) * (adj_y2 - adj_y1)) > ((width*height)/3):
                            overlaps = True
                            break
                
                if overlaps == False:

                    dimension_increaser = temp
            
            final_dataset['boundry_box'].append([x, y, (widthb + dimension_increaser*2), (heightb + dimension_increaser*2)])
            #print([x, y, (widthb + dimension_increaser*2), (heightb + dimension_increaser*2)])

            final_dataset['image_ids'].append(element["image_id"])

            final_dataset['trash'].append(1)

            final_dataset['environment'].append(element['scene_id'])


        #add bbox images for non trash regions

        

        
        goal_height_width = math.floor(min(width, height) / 4)
        
        for box in element['bboxes']:

            counter = 0
            num_pushed = 0
            fail_rate = 0
            box_x = math.floor(box[0])
            box_y = math.floor(box[1])
            widthb = box[2]
            heightb = box[3]
            x1 = box_x - math.floor((box[2])/2)
            x2 = box_x + math.floor((box[2])/2)
            y1 = box_y - math.floor(heightb/2)
            y2 = box_y + math.floor(heightb/2)


            #print(element['image_id'])
            while (counter != 500 and num_pushed != 1 and fail_rate != 1000):
                
                fail_rate = fail_rate + 1
                #print(fail_rate)
                adj_x = random.randint(0, goal_height_width)
                adj_y = random.randint(0, goal_height_width)
                adj_x1 = adj_x - math.floor(goal_height_width/2)
                adj_x2 = adj_x + math.floor(goal_height_width/2)
                adj_y1 = adj_y - math.floor(goal_height_width/2)
                adj_y2 = adj_y + math.floor(goal_height_width/2)

                if not(((x1 < adj_x1 < x2) or (x1 < adj_x2 < x2)) and ((y1 < adj_y1 < y2) or (y1 < adj_y2 < y2))):

                    
                    counter = counter + 1

                    success = True

                    for obox in element['bboxes']:
                        
                        obox_x = math.floor(obox[0])
                        obox_y = math.floor(obox[1])
                        owidthb = box[2]
                        oheightb = box[3]
                        ox1 = obox_x - math.floor((obox[2])/2)
                        ox2 = obox_x + math.floor((obox[2])/2)
                        oy1 = obox_y - math.floor(oheightb/2)
                        oy2 = obox_y + math.floor(oheightb/2)

                        if (obox_x)!= box_x and (obox_y!= box_y):
                            if (((ox1 < adj_x1 < ox2) or (ox1 < adj_x2 < ox2)) and ((oy1 < adj_y1 < oy2) or (oy1 < adj_y2 < oy2))):
                                success = False

                    if success == True:
                        num_pushed = num_pushed + 1

                        final_dataset['boundry_box'].append([adj_x, adj_y, (goal_height_width), (goal_height_width)])
                        #print([x, y, (widthb + dimension_increaser*2), (heightb + dimension_increaser*2)])

                        final_dataset['image_ids'].append(element["image_id"])

                        final_dataset['trash'].append(0)

                        final_dataset['environment'].append(element['scene_id'])
        

    #print(len(final_dataset['image_ids']))

    if (export_json == True):
        #print("outputting...")
        fout = open(output_json,"w")
        fout.write(simplejson.dumps(final_dataset, indent = 2, sort_keys = False))
        fout.close()
    
    return final_dataset


def export_roi_dataset(final_dataset, trash_annotations, output_folder):

    images_pil = []

    for i in range(0, len(final_dataset['image_ids'])) :

        image_id = final_dataset['image_ids'][i] 

        filenamey = trash_annotations[ image_id ]['filename']

        pil_image = Image.open ("./image_data/" +  filenamey)

        #full_image = numpy.array (  pil_image  )

        x1 = final_dataset['boundry_box'][i][0] - math.floor(final_dataset['boundry_box'][i][2]/2)
        x2 = final_dataset['boundry_box'][i][0] + math.floor(final_dataset['boundry_box'][i][2]/2)
        y1 = final_dataset['boundry_box'][i][1] - math.floor(final_dataset['boundry_box'][i][3]/2)
        y2 = final_dataset['boundry_box'][i][1] + math.floor(final_dataset['boundry_box'][i][3]/2)



        if (x1 < 0):
            x1 = 0

        if (y1 < 0):
            y1 = 0


        #if (x2 > ((np.shape(full_image))[0]) - 2):
           # x2 = ((np.shape(full_image))[0]) - 2

        #if (y2 > ((np.shape(full_image))[1]) - 2):
            #y2 = ((np.shape(full_image))[0]) - 2

        if x1 == x2:
            x2 = x1 + 1
        
        if y1 == y2:
            y2 = y1 + 1

        
        #new_im = Image.fromarray(full_image[x1:x2, y1:y2, :])


        
        width, height = pil_image.size

        #print(pil_image.size)
        #print(x1, y2, x2, y1)

        #images_np.append(full_image[x1:x2, y1:y2, :])

        new_im = pil_image.crop((x1, height - y2, x2, height - y1))

        new_im = new_im.resize((1000,1000))

        images_pil.append(new_im)

        environments = [str(x) for x in final_dataset['environment'][i]]

        environments = "".join(environments)

        new_im.save(output_folder + str(i) + "_" + str(image_id) + "_" + str(final_dataset['trash'][i]) + "_" + environments + ".png")

    
    #arr_reshaped = images_np(arr.shape[0], -1)
    

    #np.savetxt("./data/numpy_image_data.txt", arr_reshaped)




    # return image_ids, image_data, trash_labels, environment_labels
    return final_dataset["image_ids"], images_pil, final_dataset["trash"], final_dataset["environment"]



        
def data_runner(json_file, export_json = True, num_files = 500, output_data_subset_json = '', output_new_dataset_json = '', output_data_folder = ''):


    
    
    trash_image_annot = export_modified_json_dataset(json_file = json_file, output_json = output_data_subset_json, export_json=export_json, num_files = num_files)

    print("Created Data Subset Json....")
    print(len(trash_image_annot))

    final_dataset = create_custom_boxes(trash_image_annot, output_json = output_data_subset_json, export_json = export_json)

    print("Created New ROI Box Boundries JSON Data...")
    print(len(final_dataset['image_ids']))


    #all lists -> image_data is list of numpy_arrays
    image_ids, image_data, trash_labels, environment_labels = export_roi_dataset(final_dataset, trash_image_annot, output_folder = output_data_folder)

    print("Created New Image Dataset and Data Variables...")
    print(len(image_data))

    return image_ids, image_data, trash_labels, environment_labels



def main():

    json_file = "./data/annotations.json"

    output_data_subset_json = "./data/custom_498_data.json"

    output_new_dataset_json = "./Generated_Data_JSON/roi_boundries_custom_dataset.json"

    output_data_folder = "./new_data/"

    image_ids, image_data, trash_labels, environment_labels = data_runner(json_file, export_json = True, num_files = 500, output_data_subset_json = output_data_subset_json, output_new_dataset_json = output_new_dataset_json, output_data_folder = output_data_folder )



if __name__ == "__main__":
    main()
