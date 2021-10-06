import ocr_utility
import img_utility
import text_scene_detection_utility
import pandas as pd
import kpi_utility
from IPython.display import clear_output, display

net = text_scene_detection_utility.build_model()

# run entire ocr pipeline
def do_ocr(path, custom_config=r'--oem 1 --psm 8'):

    img_list = img_utility.load_img(path)
    bounding_boxes, b, ret_score_text = text_scene_detection_utility.text_detection(net, img_list, text_threshold=0.8, link_threshold=0.4, low_text=0.4, cuda=True, poly=False, refine_net=None)
    
    if len(bounding_boxes) == 0:
        return '', 100
    
    croped_images, font_color, all_mean_edge_colors, total_mean_edge_colors_all = img_utility.crop_images(bounding_boxes, img_list)
    text, conf = ocr_utility.image_to_text(croped_images, custom_config)
    
    return text, conf, font_color, all_mean_edge_colors
    

def ocr_on_df(df, ds_name, path, img_format):
    
    result_list = []
    
    for i in range(len(df)):
#         try:
        clear_output(wait=True)
        display(i)

        row = df.iloc[i]

        if ds_name=='fb': 
            img_columm = row['img']
            _id = img_columm.split('/')[-1].split('.')[0]
            orig_text = row['text']
            img_name = _id + '.' + img_format
        if ds_name=='memotion':
            img_name = row['image_name']
            _id = img_name.split('.')[0]
            orig_text = row['text_corrected']
        
        if ds_name=='font_color':
            
            _id = 123
            orig_text = 'Dummy'
            ds_name_2 = row['dataset']
            font_color_orig = row['font_color']
            
            path_2 = row['path']
            file_name = row['file_name']
            img_name = file_name


#         text, conf, font_color, all_mean_edge_colors = do_ocr(path_2+img_name)
        text, conf, font_color, all_mean_edge_colors = do_ocr(img_name)


            result = {'ds_name': ds_name, '_id': _id, 'confidence': conf, 'orig_text': orig_text.lower(), 'pred_text': text, 'font_color': font_color, 'all_mean_edge_colors': all_mean_edge_colors}
#         result = {'ds_name': ds_name_2, '_id': _id, 'confidence': conf, 'orig_text': orig_text.lower(), 'pred_text': text, 'font_color': font_color, 'all_mean_edge_colors': all_mean_edge_colors, 'font_color_orig': font_color_orig}
        result_list.append(result)
#         except:
#             continue
            
    result_df = pd.DataFrame(result_list)
    return result_df

def add_columns_on_df(df,columns):
    
    if 'cos_similarity' in columns:
    
        df['cos_similarity'] = df.apply(lambda x: kpi_utility.get_cosine(x['pred_text'], x['orig_text']), axis=1)
    
    return df

