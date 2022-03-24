# Shapes-Dataset
Generates a grayscale image dataset of different shapes. Triangles, circles, ellipses, para- and hyperbolas also non solid NGons. The images are saved as a gzip compressed .csv file.


Command Line usage
* required: Argument 1 the word 'run' to enable it.
* optional: Argument 2 choose subset size, default 2000
* optional: Argument 3 Outputfile name

for example this generates 5 * 1000 images:
`python shapes_generation.py run 1000`

For a broader functional and plotting usage see the help text of the file.


---

Secondly there is a TensorFlow model included for classification with an accuracy of 99%+ on a train and test dataset.
Below you see some examples and misclassifications of the model.
The images are from an earlier version of the code where no restrain to minimal size was implemented.

![Examples and misclassification](https://github.com/Daraan/Shapes-Dataset/blob/main/images/Shapes%20Classifier%20Wrong%20Outputs.jpg)

---

Furhter credits to and code partially taken from:
[https://github.com/cctech-labs/ml-2dshapes](https://github.com/cctech-labs/ml-2dshapes)
