from pdf2image import convert_from_path
import os
from joblib import Parallel, delayed


def pdf_to_img(filename,input_dir, output_dir):
    print("processing "+filename)
    input_path=os.path.join(input_dir,filename)
    output_path=os.path.join(output_dir,filename[:-4]+'.png')
    pages = convert_from_path(input_path,dpi=600)
    for page in pages:
        page.save(output_path,fmt='png')

input_dir="../../raw_data/personnel-records/1954/scans/official_office/"
output_dir="../../raw_data/personnel-records/1954/scans/official_office/"
clean_names = lambda x: [i for i in x if i[0] != '.' and i.endswith('.pdf')]
filenames=sorted(clean_names(os.listdir(input_dir)))

Parallel(n_jobs=-1)(map(delayed(pdf_to_img), filenames, [input_dir]*len(filenames), [output_dir]*len(filenames)))