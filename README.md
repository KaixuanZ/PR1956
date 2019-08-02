# HarvardAHA

/Preprocessing is still under developing, which takes scanned images as input and output row/col images. The output should avoid blanks, pages numbers (any potential meaningless classes that are regarded as noise).

/CNN includes the code for train&test CNN which recognize the class of each row/col images.

/GraphicalModel is used on top of CNN to further improve the classification accuracy by utilizing the information between adjacent rows/cols.

/OCR is not developed yet, which should take a row/col image as input and output the characters in that image.

Master branch contains the code that generally works for most books, if you need code for a particular book, go to the correspondent branch.
