# Conformity Assessments

This repository contains an implementation for (semi-)automated assessments of artificial intelligence (AI) conformity.

# Installation 
The implementation builds on Python 3.8 and depends on several other packages for handling data, modeling, and generating reports.
To install the required packages, run the following:
```
pip install -r requirements.txt
 ```

Besides, the tool needs a running MySQL/MariaDB (compatible) database. Please import the SQL dump from the file conformity_assessment.sql into your SQL database and add the database credentials into flask_app/server.py (lines 12 and following, e.g., db_username, db_password, ...).
# Start Server

## Development Mode
To start the Flask server, go to the flask_app folder with your command line tool and run the server.py function with your Python interpreter. You can now access the conformity assessment tool with your browser on 127.0.0.1:5000.

## WSGI Server
Alternatively, use as WSGI server such as gunicorn and start the Flask app accordingly (e.g., $ unicorn -w 4 -b 0.0.0.0 'server:app').

# Usage
Conformity assessments can be run on the exemplary use case of fraud detection:

- Select the Fraud Detection use case from the landing page
- Select Fraud_Germany_2023.csv as the dataset set file, Fraud_Classifier_XGBoost.sav as the model file, TARGET as the outcome variable, and CODE_GENDER_M as the sensitive attribute. Other values might work as well, but some need minor changes in the source code.
- Answer the questions according to your best knowledge
- Explore the evaluation report 

# Citation
Please consider citing us if you find this helpful for your work:
```
@article{vonZahn.2025,
  title={Navigating AI conformity: a design framework to assess fairness, explainability, and performance},
  author={von Zahn, Moritz and Zacharias, Jan and Lowin, Maximilian and Chen, Johannes and Hinz, Oliver},
  journal={Electronic Markets},
  volume={35},
  number={24},
  year={2025},
  publisher={Springer}
}
 ```
