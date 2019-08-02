# HarvardAHA

Still under developing.

/Preprocessing takes scanned images as input and output row images. The output should avoid blanks, pages numbers (any potential meaningless classes that are regarded as noise). Currently we can accurately detect each page, the next step is to detect each col.

/CNN includes the code for train&test CNN which recognize the class of each row/col images. The parameters hasn't been tuned for PR1956 

/GraphicalModel is used on top of CNN to further improve the classification accuracy by utilizing the information between adjacent rows/cols. Parameters hasn't been tuned for PR1956, it was developed for PR1954.

/OCR is not developed yet, which should take a row/col image as input and output the characters in that image.
