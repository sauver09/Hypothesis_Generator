<h1> Hypothesis Generator</h1>

This project provides a UI interface when user can generate several hypothesis inferences of a given dataset using some principle probability tools and techniques.
Flask Framework is being in the backend.

## Steps to Run:
* Make sure you have following libraries installed

  * Flask==1.1.2
  * Jinja2==3.0.1
  * Werkzeug==2.0.1
  * numpy==1.19.2
  * matplotlib==3.3.2
  * seaborn==0.11.0
  * pandas==1.1.3
  * scipy==1.5.2
  
* When using flask framework, we must not ignore the hierarchy of the files and folders. Make sure that you have following directory structure
  
  <pre>
  ├── app.py 
  ├── static
    ├── images
  ├── templates 
    ├── index.html
</pre>

* This project currently covers following Hypthesis testing techniques:<br>
  - <strong> Data cleaning </strong> methods are implemented such as applying Tukey's outlier detection, removing Null values etc
  - <strong> One Sample Hypothesis Tests </strong> such as Wald's, Z and T tests are implemented
  - <strong> Two Population Hypothesis Tests </strong> for Wald's and T tests are implemented
  - <strong> One Sample Kolmogorov–Smirnov Hypothesis Test </strong> is implemented
  - <strong> Two Population Kolmogorov–Smirnov Hypothesis Test </strong> is implemented
  - <strong> Permutation Hypothesis Test </strong> is implemented
  
  
 
## Demo is hosted on :

- http://saurabh2612.pythonanywhere.com/

## Demo Screenshots:

- #### Home Page ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.39.19%20AM.png?raw=true)

- #### View Complete data  ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.39.39%20AM.png?raw=true)

- #### Data Cleaning ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.39.56%20AM.png?raw=true)

- #### One Sample Hypothesis tests ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.40.20%20AM.png?raw=true)

- #### Two Population Hypothesis tests ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.40.54%20AM.png?raw=true)

- #### One Sample K-S tests ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.41.14%20AM.png?raw=true)

- #### Two Population K-S tests ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.41.32%20AM.png?raw=true)

- #### Permutation Test ####
![alt text](https://github.com/sauver09/Hypothesis_Generator/blob/main/screenshots/Screen%20Shot%202021-07-20%20at%202.41.48%20AM.png?raw=true)




